#!/bin/bash

# example_questions.json 생성
if [ ! -f example_questions.json ]; then
    if [ -f example_questions.template.json ]; then
        cp example_questions.template.json example_questions.json
        echo "example_questions.json 파일이 생성되었습니다. 원하는 질문으로 수정해주세요."
    else
        echo "example_questions.template.json 파일이 존재하지 않습니다."
    fi
fi

# model_presets.json 생성
if [ ! -f model_presets.json ]; then
    if [ -f model_presets.template.json ]; then
        cp model_presets.template.json model_presets.json
        echo "model_presets.json 파일이 생성되었습니다. 원하는 설정으로 수정해주세요."
    else
        echo "model_presets.template.json 파일이 존재하지 않습니다."
    fi
fi

echo "초기 설정이 완료되었습니다."
