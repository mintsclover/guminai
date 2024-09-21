import requests
import json

class CompletionExecutor:
    """
    네이버 클로바 API를 사용하여 모델의 응답을 가져오는 클래스
    """
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def execute(self, completion_request):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        with requests.post(
            f"{self._host}/testapp/v1/chat-completions/HCX-DASH-001",
            headers=headers, json=completion_request, stream=True) as r:
            result_found = False
            for line in r.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("event:result"):
                        result_found = True
                    elif result_found and decoded_line.startswith("data:"):
                        data = json.loads(decoded_line[5:])
                        if "message" in data and "content" in data["message"]:
                            return data["message"]["content"]

        return ""
