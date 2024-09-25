# routes/chat.py
from flask import Blueprint, render_template, request, jsonify, session, current_app, redirect, url_for  # 'redirect' 추가
import random
import logging
from models.vector_store_manager import VectorStoreManager
from models.completion_executor import CompletionExecutor
from utils.conversation import manage_conversation_history
from utils.context import generate_context
from db import save_chat_history

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat')
def chat_page():
    if not session.get('authenticated'):
        return redirect(url_for('auth.index'))
    
    # 모델 변경 시 세션 초기화
    session.pop('conversation_history', None)
    
    # 예시 질문 랜덤 선택
    num_questions = 3  # 표시할 질문의 수
    all_example_questions = current_app.config.get('ALL_EXAMPLE_QUESTIONS', [])
    example_questions = random.sample(all_example_questions, num_questions) if len(all_example_questions) >= num_questions else all_example_questions
    
    # 모델 정보 전달
    model_presets = current_app.config.get('MODEL_PRESETS', {})
    models = {
        model_key: {
            'display_name': model_info.get('display_name', model_key),
            'description': model_info.get('description', ''),
            'avatar_image': model_info.get('avatar_image', 'bot_avatar.png')
        } for model_key, model_info in model_presets.items()
    }
    
    return render_template('chat.html', models=models, example_questions=example_questions)

@chat_bp.route('/chat_api', methods=['POST'])
def chat_api_endpoint():
    if not session.get('authenticated'):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    question = data.get('message', '').strip()

    # 사용자가 "test"라고 입력하면 정해진 테스트용 문장을 반환
    if question.lower() in ["test", "테스트", "ㅅㄷㄴㅅ"]:
        test_response = "이것은 테스트 응답입니다. 인공지능을 사용하지 않았습니다."
        return jsonify({'answer': test_response, 'reset_message': None})

    selected_model = data.get('model', 'model1')
    model_presets = current_app.config.get('MODEL_PRESETS', {})
    model_preset = model_presets.get(selected_model, model_presets.get('model1', {}))

    if not model_preset:
        return jsonify({'error': 'Model preset not found'}), 400

    try:
        # 클로바 실행기와 벡터 스토어 관리자 가져오기
        completion_executor = current_app.config.get('COMPLETION_EXECUTOR')
        vector_store_manager = current_app.config.get('VECTOR_STORE_MANAGER')

        # 컨텍스트 생성
        context = generate_context(question, vector_store_manager)

        # 대화 내역 관리 (사용자의 메시지 추가)
        conversation_history, reset_required = manage_conversation_history(question)

        # 모델에게 보낼 메시지 구성
        messages = construct_messages(model_preset, conversation_history, context)

        # 모델에 요청 보내기
        response = get_model_response(model_preset, messages, completion_executor)

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

def get_model_response(model_preset, messages, completion_executor):
    """
    모델에게 요청을 보내고 응답을 받는 함수
    """
    # 모델 요청 데이터 구성
    request_data = model_preset.get('request_data', {}).copy()
    request_data['messages'] = messages

    # 모델에 요청 보내기
    response = completion_executor.execute(request_data)
    return response
