import sys
import os

# 애플리케이션의 경로를 추가
sys.path.insert(0, os.path.dirname(__file__))

from app import app as application
from app import init_db  # init_db 함수 가져오기

# 데이터베이스 초기화
init_db()