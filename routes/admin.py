# routes/admin.py
from flask import Blueprint, render_template, request, redirect, url_for, session, current_app
import yaml
import logging
from config import config
from db import get_db

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/admin', methods=['GET', 'POST'])
def admin_page():
    if not session.get('admin_authenticated'):
        return redirect(url_for('auth.admin_login'))

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
            # 업데이트된 설정을 현재 애플리케이션 설정에 반영
            current_app.config.update(new_config)
            return render_template('admin.html', success='설정이 업데이트되었습니다.', config=new_config)
        except Exception as e:
            logging.error(f"설정 파일 업데이트 중 오류 발생: {e}")
            return render_template('admin.html', error='설정 업데이트 중 오류가 발생했습니다.', config=config)

    return render_template('admin.html', config=config)

@admin_bp.route('/admin/chat_history')
def chat_history_page():
    if not session.get('admin_authenticated'):
        return redirect(url_for('auth.admin_login'))

    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute('SELECT user_message, bot_response, timestamp FROM chat_history ORDER BY timestamp DESC')
        chat_logs = cursor.fetchall()
        return render_template('chat_history.html', chat_logs=chat_logs)
    except Exception as e:
        logging.error(f"채팅 기록 조회 중 오류 발생: {e}")
        return render_template('chat_history.html', chat_logs=[], error='채팅 기록을 불러오는 중 오류가 발생했습니다.')
