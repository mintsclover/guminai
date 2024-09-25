# utils/text_utils.py
def truncate_text(text, max_length):
    """
    주어진 최대 길이 내에서 텍스트를 자르되, 가능한 한 문장의 끝에서 자릅니다.
    :param text: 원본 텍스트
    :param max_length: 최대 문자 수
    :return: 잘린 텍스트
    """
    if len(text) <= max_length:
        return text
    # 가능한 마지막 문장 끝 찾기
    truncated = text[:max_length]
    last_period = truncated.rfind('.')
    last_newline = truncated.rfind('\n')
    split_pos = max(last_period, last_newline)
    if split_pos != -1:
        return truncated[:split_pos+1]
    else:
        # 문장 끝을 찾지 못하면 그냥 자름
        return truncated
