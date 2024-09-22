from flask import Flask, render_template, request, redirect, url_for, session, jsonify, g
import os
import json
import random
import yaml
import logging
from datetime import datetime
import sqlite3
import faiss
import shutil
import math

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

# 초기 설정 파일 생성 함수
def setup_files():
    templates = {
        'example_questions.json': 'example_questions.template.json',
        'model_presets.json': 'model_presets.template.json'
    }
    for target, template in templates.items():
        if not os.path.exists(target):
            if os.path.exists(template):
                shutil.copy(template, target)
                logging.info(f"{target} 파일이 생성되었습니다. 원하는 설정으로 수정해주세요.")
            else:
                logging.error(f"{template} 파일이 존재하지 않습니다.")

# 초기 설정 실행
setup_files()

# 예시 질문 로드 함수
def load_questions():
    try:
        with open('example_questions.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('questions', [])
    except FileNotFoundError:
        logging.warning("example_questions.json 파일이 없으므로 빈 리스트를 반환합니다.")
        return []
    except json.JSONDecodeError:
        logging.error("example_questions.json 파일이 유효한 JSON 형식이 아닙니다.")
        return []

# 예시 질문 로드
all_example_questions = load_questions()

# 모델 프리셋 로드 함수
def load_model_presets():
    try:
        with open('model_presets.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning("model_presets.json 파일이 없으므로 빈 딕셔너리를 반환합니다.")
        return {}
    except json.JSONDecodeError:
        logging.error("model_presets.json 파일이 유효한 JSON 형식이 아닙니다.")
        return {}

# 모델 프리셋 로드
model_presets = load_model_presets()

# 벡터 스토어 매니저 및 CompletionExecutor 임포트
from vector_store_manager import VectorStoreManager
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
    session.pop('conversation_history', None)
    
    # 예시 질문 랜덤 선택
    num_questions = 3  # 표시할 질문의 수
    example_questions = random.sample(all_example_questions, num_questions) if len(all_example_questions) >= num_questions else all_example_questions
    
    # 모델 정보 전달
    models = {
        model_key: {
            'display_name': model_info.get('display_name', model_key),
            'description': model_info.get('description', ''),
            'avatar_image': model_info.get('avatar_image', 'bot_avatar.png')
        } for model_key, model_info in model_presets.items()
    }
    
    return render_template('chat.html', models=models, example_questions=example_questions)

# 채팅 API 엔드포인트
@app.route('/chat_api', methods=['POST'])
def chat_api():
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    question = data.get('message', '').strip()

    # 사용자가 "test"라고 입력하면 정해진 테스트용 문장을 반환
    if question.lower() in ["test", "테스트", "ㅅㄷㄴㅅ"]:
        test_response = "이것은 테스트 응답입니다. 인공지능을 사용하지 않았습니다."
        return jsonify({'answer': test_response, 'reset_message': None})

    selected_model = data.get('model', 'model1')
    model_preset = model_presets.get(selected_model, model_presets.get('model1', {}))

    if not model_preset:
        return jsonify({'error': 'Model preset not found'}), 400

    try:
        # 컨텍스트 생성
        context = generate_context(question)

        # 대화 내역 관리 (사용자의 메시지 추가)
        conversation_history, reset_required = manage_conversation_history(question)

        # 모델에게 보낼 메시지 구성
        messages = construct_messages(model_preset, conversation_history, context)

        # 모델에 요청 보내기
        response = get_model_response(model_preset, messages)

        # 대화 내역에 봇의 응답 추가
        conversation_history.append({'role': 'assistant', 'content': response})
        session['conversation_history'] = conversation_history

        # 채팅 기록 저장
        save_chat_history(question, response)

        # 기억력 초기화가 필요한 경우 대화 내역 초기화
        reset_message = None
        if reset_required:
            session['conversation_history'] = []
            reset_message = '기억력이 초기화되었습니다!'

        # 응답 반환
        return jsonify({'answer': response, 'reset_message': reset_message})
    except Exception as e:
        logging.error(f"채팅 처리 중 오류 발생: {e}")
        return jsonify({'error': '채팅 처리 중 오류가 발생했습니다.'}), 500

def manage_conversation_history(question):
    """
    대화 내역을 관리하는 함수 (기존 대화 내역 유지, 필요 시 초기화)
    """
    conversation_history = session.get('conversation_history', [])
    conversation_history.append({'role': 'user', 'content': question})

    # 기억력 제한 가져오기
    max_memory_length = int(config.get('max_memory_length', 10))

    # 기억력 초기화 여부 확인
    reset_required = len(conversation_history) > max_memory_length

    # 대화 내역 세션에 저장
    session['conversation_history'] = conversation_history

    return conversation_history, reset_required

def construct_messages(model_preset, conversation_history, context):
    """
    모델에게 전달할 메시지를 구성하는 함수
    """
    # 모델 프리셋의 사전 설정 메시지 가져오기
    preset_text = model_preset.get('preset_text', []).copy()

    # 대화 내역 추가
    messages = preset_text + conversation_history

    # 컨텍스트를 시스템 메시지로 추가
    if context:
        messages.append({'role': 'system', 'content': f'사전 정보: {context}'})

    return messages

def get_model_response(model_preset, messages):
    """
    모델에게 요청을 보내고 응답을 받는 함수
    """
    # 모델 요청 데이터 구성
    request_data = model_preset.get('request_data', {}).copy()
    request_data['messages'] = messages

    # 모델에 요청 보내기
    response = completion_executor.execute(request_data)
    return response

@app.route('/get_example_questions')
def get_example_questions():
    num_questions = 3  # 표시할 질문의 수
    example_questions = random.sample(all_example_questions, num_questions) if len(all_example_questions) >= num_questions else all_example_questions
    return jsonify({'example_questions': example_questions})

@app.route('/reset_conversation', methods=['POST'])
def reset_conversation():
    session.pop('conversation_history', None)
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

        # 정수로 변환 가능한 값들은 정수로 변환
        for key, value in new_config.items():
            try:
                if value.isdigit():  # 정수인지 확인
                    new_config[key] = int(value)
            except ValueError:
                continue  # 정수가 아닌 경우 그대로 둠

        try:
            with open('config.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(new_config, f, allow_unicode=True)
            return render_template('admin.html', success='설정이 업데이트되었습니다.', config=new_config)
        except Exception as e:
            logging.error(f"설정 파일 업데이트 중 오류 발생: {e}")
            return render_template('admin.html', error='설정 업데이트 중 오류가 발생했습니다.', config=config)

    return render_template('admin.html', config=config)

@app.route('/admin/chat_history')
def chat_history():
    if not session.get('admin_authenticated'):
        return redirect(url_for('admin'))

    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT user_message, bot_response, timestamp FROM chat_history ORDER BY timestamp DESC')
        chat_logs = cursor.fetchall()
        return render_template('chat_history.html', chat_logs=chat_logs)
    except Exception as e:
        logging.error(f"채팅 기록 조회 중 오류 발생: {e}")
        return render_template('chat_history.html', chat_logs=[], error='채팅 기록을 불러오는 중 오류가 발생했습니다.')

def generate_context(question, alpha=int(config.get('alpha', 1))):
    """
    사용자 질문에 대한 컨텍스트를 생성하는 함수
    :param question: 사용자 질문 문자열
    :param alpha: 가중치 조정 파라미터 (기본값: 1.0)
    :return: 생성된 컨텍스트 문자열
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
        doc = vector_store_manager.vector_store.docstore.get(doc_id)
        if doc:
            docs_and_scores.append((doc, score))

    if not docs_and_scores:
        logging.warning("유사한 문서를 찾을 수 없습니다.")
        return ""

    # 문서 순위별 가중치 계산 (지수 함수 사용)
    weights = [math.exp(-alpha * rank) for rank in range(len(docs_and_scores))]

    # 가중치 정규화
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # 컨텍스트 생성
    max_total_length = int(config.get('max_total_length', 1500))
    context = ""
    total_length = 0

    for i, (doc, score) in enumerate(docs_and_scores):
        section_content = doc.page_content
        section_title = doc.metadata.get('title', 'Untitled')

        # 할당할 문자 수 계산
        allocated_length = math.floor(normalized_weights[i] * max_total_length)

        # 제목과 구분자의 길이 계산
        title_text = f"# {section_title}\n"
        separator_text = "\n---\n"
        title_length = len(title_text)
        separator_length = len(separator_text)

        # 할당된 길이에서 제목과 구분자 길이를 제외
        content_max_length = allocated_length - title_length - separator_length
        if content_max_length <= 0:
            continue  # 할당된 길이가 제목과 구분자를 포함할 수 없는 경우 건너뜀

        # 내용 자르기 (문장의 중간에서 자르지 않도록)
        truncated_content = truncate_text(section_content, content_max_length)
        actual_length = len(title_text) + len(truncated_content) + len(separator_text)

        if total_length + actual_length > max_total_length:
            # 남은 길이에 맞게 조정
            remaining_length = max_total_length - total_length
            if remaining_length <= len(title_text) + len(separator_text):
                break  # 제목과 구분자를 추가할 공간이 부족하면 종료
            content_max_length = remaining_length - len(title_text) - len(separator_text)
            truncated_content = truncate_text(section_content, content_max_length)
            context += f"{title_text}{truncated_content}{separator_text}"
            total_length += len(title_text) + len(truncated_content) + len(separator_text)
            break  # 최대 길이에 도달했으므로 루프 종료
        else:
            context += f"{title_text}{truncated_content}{separator_text}"
            total_length += actual_length

        # 사용된 문서의 정보 로그 출력
        # logging.info(f"유사도 점수: {score}, 제목: {section_title}, 할당 문자 수: {allocated_length}")
        # logging.info(f"요약 내용:\n{truncated_content}\n")
    # logging.info(f"컨텍스트: {context}\n")

    return context

def truncate_text(text, max_length):
    """
    주어진 최대 길이 내에서 텍스트를 자르되, 가능한 한 문장의 끝에서 자릅니다.
    :param text: 원본 텍스트
    :param max_length: 최대 문자 수
    :return: 잘린 텍스트
    """
    if len(text) <= max_length:
        return text
    # 가능한 마지막 문장 끝 찾기
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    split_pos = max(last_period, last_newline)
    if split_pos != -1:
        return truncated[:split_pos+1]
    else:
        # 문장 끝을 찾지 못하면 그냥 자름
        return truncated

def save_chat_history(user_message, bot_response):
    """
    채팅 기록을 데이터베이스에 저장하는 함수
    """
    try:
        db = get_db()
        cursor = db.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO chat_history (user_message, bot_response, timestamp)
            VALUES (?, ?, ?)
        ''', (user_message, bot_response, timestamp))
        db.commit()
    except Exception as e:
        logging.error(f"채팅 기록 저장 중 오류 발생: {e}")

def get_db():
    """
    데이터베이스 연결을 가져오는 함수
    """
    if 'db' not in g:
        g.db = sqlite3.connect('chat_history.db')
        g.db.row_factory = sqlite3.Row  # 딕셔너리 형태로 가져오기 위해 추가
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
    try:
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
    except Exception as e:
        logging.error(f"데이터베이스 초기화 중 오류 발생: {e}")
    finally:
        conn.close()

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
