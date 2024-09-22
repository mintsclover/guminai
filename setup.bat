@echo off

REM example_questions.json 생성
if not exist "example_questions.json" (
    if exist "example_questions.template.json" (
        copy "example_questions.template.json" "example_questions.json"
        echo example_questions.json 파일이 생성되었습니다. 원하는 질문으로 수정해주세요.
    ) else (
        echo example_questions.template.json 파일이 존재하지 않습니다.
    )
)

REM model_presets.json 생성
if not exist "model_presets.json" (
    if exist "model_presets.template.json" (
        copy "model_presets.template.json" "model_presets.json"
        echo model_presets.json 파일이 생성되었습니다. 원하는 설정으로 수정해주세요.
    ) else (
        echo model_presets.template.json 파일이 존재하지 않습니다.
    )
)

echo 초기 설정이 완료되었습니다.
pause
