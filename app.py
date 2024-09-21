from flask import Flask, render_template, request, redirect, url_for, session, jsonify, g
import os
import json
import random
import yaml
import logging
from datetime import datetime
import sqlite3
import faiss

# 환경 변수 및 설정 파일 로드
from dotenv import load_dotenv
load_dotenv()

# 설정 파일 로드
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 로그 설정
log_level = config.get('log_level', 'CRITICAL').upper()
logging.basicConfig(level=getattr(logging, log_level), format='[%(levelname)s] %(message)s')

# Flask 애플리케이션 설정
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

# 비밀번호 설정
CHAT_PASSWORD = os.getenv("CHAT_PASSWORD")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

# 예시 질문 로드
with open('example_questions.json', 'r', encoding='utf-8') as f:
    example_questions_data = json.load(f)
    all_example_questions = example_questions_data.get('questions', [])

# 모델 프리셋 임포트
from model_presets import model_presets

# 벡터 스토어 매니저 임포트
from vector_store_manager import VectorStoreManager

# Clova CompletionExecutor 클래스 임포트
from completion_executor import CompletionExecutor

# 벡터 스토어 관리자 초기화
vector_store_manager = VectorStoreManager()
vector_store_manager.get_vector_store()

# 네이버 클로바 API 설정
CLOVA_API_KEY = os.getenv("CLOVA_API_KEY")
CLOVA_PRIMARY_KEY = os.getenv("CLOVA_PRIMARY_KEY")
CLOVA_REQUEST_ID = os.getenv("CLOVA_REQUEST_ID")
CLOVA_HOST = 'https://clovastudio.stream.ntruss.com'

# 클로바 실행기 초기화
completion_executor = CompletionExecutor(
    host=CLOVA_HOST,
    api_key=CLOVA_API_KEY,
    api_key_primary_val=CLOVA_PRIMARY_KEY,
    request_id=CLOVA_REQUEST_ID
)

# 인덱스 페이지 (비밀번호 입력)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == CHAT_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('chat'))
        else:
            return render_template('index.html', error='비밀번호가 올바르지 않습니다.')
    return render_template('index.html')

# 채팅 페이지
@app.route('/chat')
def chat():
    if not session.get('authenticated'):
        return redirect(url_for('index'))
    
    # 모델 변경 시 세션 초기화
    session['conversation_history'] = []
    
    # 예시 질문 랜덤 선택
    num_questions = 3  # 표시할 질문의 수
    example_questions = random.sample(all_example_questions, num_questions)
    
    return render_template('chat.html', models=model_presets.keys(), example_questions=example_questions)

# 채팅 API 엔드포인트
@app.route('/chat_api', methods=['POST'])
def chat_api():
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    question = data.get('message')

    # 사용자가 "test"라고 입력하면 정해진 테스트용 문장을 반환
    if question.strip().lower() in ["test", "테스트", "ㅅㄷㄴㅅ"]:
        test_response = "이것은 테스트 응답입니다. 인공지능을 사용하지 않았습니다."
        return jsonify({'answer': test_response, 'reset_message': None})
    
    selected_model = data.get('model', 'model1')
    model_preset = model_presets.get(selected_model, model_presets['model1'])

    # 컨텍스트 생성
    context = generate_context(question)

    # 대화 내역 관리
    conversation_history, reset_message = manage_conversation_history(question)

    # 모델에게 보낼 메시지 구성
    messages = construct_messages(model_preset, conversation_history, context)

    # 모델에 요청 보내기
    response = get_model_response(model_preset, messages)

    # 대화 내역에 봇의 응답 추가
    conversation_history.append({'role': 'assistant', 'content': response})
    session['conversation_history'] = conversation_history

    # 채팅 기록 저장
    save_chat_history(question, response)

    # 응답 반환
    return jsonify({'answer': response, 'reset_message': reset_message})

def manage_conversation_history(question):
    """
    대화 내역을 관리하는 함수
    """
    # 대화 내역 초기화 또는 가져오기
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    conversation_history = session['conversation_history']

    # 사용자의 메시지 추가
    conversation_history.append({'role': 'user', 'content': question})

    # 기억력 제한 가져오기
    max_memory_length = int(config.get('max_memory_length', 10))

    # 기억력 초기화 여부 확인
    reset_message = None
    if len(conversation_history) > max_memory_length:
        conversation_history = []
        session['conversation_history'] = conversation_history
        reset_message = '기억력이 초기화되었습니다!'

    return conversation_history, reset_message

def construct_messages(model_preset, conversation_history, context):
    """
    모델에게 전달할 메시지를 구성하는 함수
    """
    # 모델 프리셋의 사전 설정 메시지 가져오기
    preset_text = model_preset['preset_text'].copy()

    # 대화 내역 추가
    messages = preset_text + conversation_history

    # 컨텍스트를 시스템 메시지로 추가
    messages.append({'role': 'system', 'content': f'사전 정보: {context}'})

    return messages

def get_model_response(model_preset, messages):
    """
    모델에게 요청을 보내고 응답을 받는 함수
    """
    # 모델 요청 데이터 구성
    request_data = model_preset['request_data'].copy()
    request_data['messages'] = messages

    # 모델에 요청 보내기
    response = completion_executor.execute(request_data)
    return response

@app.route('/get_example_questions')
def get_example_questions():
    num_questions = 3  # 표시할 질문의 수
    example_questions = random.sample(all_example_questions, num_questions)
    return jsonify({'example_questions': example_questions})

@app.route('/reset_conversation', methods=['POST'])
def reset_conversation():
    session['conversation_history'] = []
    return '', 204

# 관리자 페이지
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if not session.get('admin_authenticated'):
        if request.method == 'POST':
            password = request.form.get('password')
            if password == ADMIN_PASSWORD:
                session['admin_authenticated'] = True
                return redirect(url_for('admin'))
            else:
                return render_template('admin.html', error='비밀번호가 올바르지 않습니다.')
        return render_template('admin.html')

    # 설정 변경 로직
    if request.method == 'POST':
        new_config = request.form.to_dict()
        with open('config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(new_config, f, allow_unicode=True)
        return render_template('admin.html', success='설정이 업데이트되었습니다.', config=new_config)

    return render_template('admin.html', config=config)

@app.route('/admin/chat_history')
def chat_history():
    if not session.get('admin_authenticated'):
        return redirect(url_for('admin'))

    db = get_db()
    cursor = db.cursor()
    cursor.execute('SELECT user_message, bot_response, timestamp FROM chat_history ORDER BY timestamp DESC')
    chat_logs = cursor.fetchall()

    return render_template('chat_history.html', chat_logs=chat_logs)

def generate_context(question):
    """
    사용자 질문에 대한 컨텍스트를 생성하는 함수
    """
    # 질문 임베딩
    question_embedding = vector_store_manager.get_embedding(question).reshape(1, -1)
    faiss.normalize_L2(question_embedding)

    # 유사도 검색
    k = int(config.get('top_k', 5))
    D, I = vector_store_manager.vector_store.index.search(question_embedding, k)

    # 유사도 점수와 문서 매핑
    docs_and_scores = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        doc_id = vector_store_manager.vector_store.index_to_docstore_id[idx]
        doc = vector_store_manager.vector_store.docstore[doc_id]
        docs_and_scores.append((doc, score))

    # 컨텍스트 생성
    max_total_length = int(config.get('max_total_length', 1500))
    total_length = 0
    context = ""

    for idx, (doc, score) in enumerate(docs_and_scores):
        section_content = doc.page_content
        section_title = doc.metadata['title']
        section_length = len(section_content)

        if total_length + section_length > max_total_length:
            # 남은 길이만큼 자름
            section_content = section_content[:max_total_length - total_length]
            section_length = len(section_content)
            if section_length == 0:
                continue

        total_length += section_length
        context += f"# {section_title}\n{section_content}\n---\n"

        # 사용된 문서의 정보 로그 출력
        logging.info(f"유사도 순위: {idx + 1}, 점수: {score}, 제목: {section_title}")
        logging.info(f"내용:\n{section_content}\n")

        if total_length >= max_total_length:
            break

    return context

def save_chat_history(user_message, bot_response):
    """
    채팅 기록을 데이터베이스에 저장하는 함수
    """
    db = get_db()
    cursor = db.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO chat_history (user_message, bot_response, timestamp)
        VALUES (?, ?, ?)
    ''', (user_message, bot_response, timestamp))
    db.commit()

def get_db():
    """
    데이터베이스 연결을 가져오는 함수
    """
    if 'db' not in g:
        g.db = sqlite3.connect('chat_history.db')
    return g.db

@app.teardown_appcontext
def close_db(error):
    """
    애플리케이션 컨텍스트 종료 시 데이터베이스 연결을 닫는 함수
    """
    db = g.pop('db', None)

    if db is not None:
        db.close()

def init_db():
    """
    데이터베이스를 초기화하는 함수
    """
    conn = sqlite3.connect('chat_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message TEXT,
            bot_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
