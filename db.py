# db.py
import sqlite3
import logging
from flask import g
from config import config

def get_db():
    """
    데이터베이스 연결을 가져오는 함수
    """
    if 'db' not in g:
        g.db = sqlite3.connect('chat_history.db')
        g.db.row_factory = sqlite3.Row  # 딕셔너리 형태로 가져오기 위해 추가
    return g.db

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
