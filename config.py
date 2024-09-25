# config.py
import os
import yaml
from dotenv import load_dotenv

# 환경 변수 및 설정 파일 로드
load_dotenv()

# 설정 파일 로드
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 환경 변수
SECRET_KEY = os.getenv("SECRET_KEY")
CHAT_PASSWORD = os.getenv("CHAT_PASSWORD")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
CLOVA_API_KEY = os.getenv("CLOVA_API_KEY")
CLOVA_PRIMARY_KEY = os.getenv("CLOVA_PRIMARY_KEY")
CLOVA_REQUEST_ID = os.getenv("CLOVA_REQUEST_ID")
CLOVA_HOST = 'https://clovastudio.stream.ntruss.com'

# 기타 설정
LOG_LEVEL = config.get('log_level', 'CRITICAL').upper()
MAX_MEMORY_LENGTH = int(config.get('max_memory_length', 10))
ALPHA = int(config.get('alpha', 1))
TOP_K = int(config.get('top_k', 5))
MAX_TOTAL_LENGTH = int(config.get('max_total_length', 1500))
