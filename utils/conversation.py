# utils/conversation.py
from flask import session
from config import MAX_MEMORY_LENGTH
import logging

def manage_conversation_history(question):
    """
    대화 내역을 관리하는 함수 (기존 대화 내역 유지, 필요 시 초기화)
    """
    conversation_history = session.get('conversation_history', [])
    conversation_history.append({'role': 'user', 'content': question})

    # 기억력 제한 가져오기
    max_memory_length = MAX_MEMORY_LENGTH

    # 기억력 초기화 여부 확인
    reset_required = len(conversation_history) > max_memory_length

    # 대화 내역 세션에 저장
    session['conversation_history'] = conversation_history

    return conversation_history, reset_required
