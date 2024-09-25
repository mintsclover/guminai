# app.py
from flask import Flask
import logging
from config import LOG_LEVEL, SECRET_KEY
from setup import initialize
from db import init_db, close_db
from routes.auth import auth_bp
from routes.chat import chat_bp
from routes.admin import admin_bp
from models.vector_store_manager import VectorStoreManager
from models.completion_executor import CompletionExecutor
from config import CLOVA_HOST, CLOVA_API_KEY, CLOVA_PRIMARY_KEY, CLOVA_REQUEST_ID
from utils.context import generate_context
from utils.conversation import manage_conversation_history
from db import save_chat_history

# 초기 설정 및 로드
all_example_questions, model_presets = initialize()

# 로그 설정
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format='[%(levelname)s] %(message)s')

# Flask 애플리케이션 설정
app = Flask(__name__)
app.secret_key = SECRET_KEY

# 블루프린트 등록
app.register_blueprint(auth_bp)
app.register_blueprint(chat_bp)
app.register_blueprint(admin_bp)

# 벡터 스토어 관리자 초기화
vector_store_manager = VectorStoreManager()
vector_store_manager.get_vector_store()

# 클로바 실행기 초기화
completion_executor = CompletionExecutor(
    host=CLOVA_HOST,
    api_key=CLOVA_API_KEY,
    api_key_primary_val=CLOVA_PRIMARY_KEY,
    request_id=CLOVA_REQUEST_ID
)

# 데이터베이스 초기화
@app.before_first_request
def initialize_database():
    init_db()

# 데이터베이스 연결 종료
app.teardown_appcontext(close_db)

if __name__ == '__main__':
    app.run(debug=True)
