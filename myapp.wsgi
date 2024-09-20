import sys
import os

# 애플리케이션의 경로를 추가
sys.path.insert(0, os.path.dirname(__file__))

from app import app as application