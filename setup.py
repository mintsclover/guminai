# setup.py
import os
import shutil
import json
import logging

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

def load_json(file_path, default):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"{file_path} 파일이 없으므로 기본값을 반환합니다.")
        return default
    except json.JSONDecodeError:
        logging.error(f"{file_path} 파일이 유효한 JSON 형식이 아닙니다.")
        return default

def initialize():
    setup_files()
    all_example_questions = load_json('example_questions.json', {}).get('questions', [])
    model_presets = load_json('model_presets.json', {})
    return all_example_questions, model_presets
