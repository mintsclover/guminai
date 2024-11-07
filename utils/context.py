# utils/context.py
import math
import faiss
import logging
from config import ALPHA, TOP_K, MAX_TOTAL_LENGTH
from utils.text_utils import truncate_text

def generate_context(question, vector_store_manager):
    """
    사용자 질문에 대한 컨텍스트를 생성하는 함수
    :param question: 사용자 질문 문자열
    :param vector_store_manager: VectorStoreManager 인스턴스
    :return: 생성된 컨텍스트 문자열
    """
    # 질문 임베딩
    question_embedding = vector_store_manager.get_embedding(question).reshape(1, -1)
    faiss.normalize_L2(question_embedding)

    # 유사도 검색
    k = TOP_K
    D, I = vector_store_manager.vector_store.index.search(question_embedding, k)

    # 유사도 점수와 문서 매핑
    docs_and_scores = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        doc_id = vector_store_manager.vector_store.index_to_docstore_id[idx]
        doc = vector_store_manager.vector_store.docstore.get(doc_id)
        if doc:
            docs_and_scores.append((doc, score))

    if not docs_and_scores:
        logging.warning("유사한 문서를 찾을 수 없습니다.")
        return ""

    # 문서 순위별 가중치 계산 (지수 함수 사용)
    weights = [math.exp(-ALPHA * rank) for rank in range(len(docs_and_scores))]

    # 가중치 정규화
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # 컨텍스트 생성
    context = ""
    total_length = 0

    for i, (doc, score) in enumerate(docs_and_scores):
        section_content = doc.page_content
        section_title = doc.metadata.get('title', 'Untitled')

        # 할당할 문자 수 계산
        allocated_length = math.floor(normalized_weights[i] * MAX_TOTAL_LENGTH)

        # 제목과 구분자의 길이 계산
        title_text = f"# {section_title}\n"
        separator_text = "\n---\n"
        title_length = len(title_text)
        separator_length = len(separator_text)

        # 할당된 길이에서 제목과 구분자 길이를 제외
        content_max_length = allocated_length - title_length - separator_length
        if content_max_length <= 0:
            continue  # 할당된 길이가 제목과 구분자를 포함할 수 없는 경우 건너뜀

        # 내용 자르기 (문장의 중간에서 자르지 않도록)
        truncated_content = truncate_text(section_content, content_max_length)
        actual_length = len(title_text) + len(truncated_content) + len(separator_text)

        if total_length + actual_length > MAX_TOTAL_LENGTH:
            # 남은 길이에 맞게 조정
            remaining_length = MAX_TOTAL_LENGTH - total_length
            if remaining_length <= len(title_text) + len(separator_text):
                break  # 제목과 구분자를 추가할 공간이 부족하면 종료
            content_max_length = remaining_length - len(title_text) - len(separator_text)
            truncated_content = truncate_text(section_content, content_max_length)
            context += f"{title_text}{truncated_content}{separator_text}"
            total_length += len(title_text) + len(truncated_content) + len(separator_text)
            break  # 최대 길이에 도달했으므로 루프 종료
        else:
            context += f"{title_text}{truncated_content}{separator_text}"
            total_length += actual_length

    return context

def generate_context2(question, vector_store_manager):
    """
    사용자 질문에 대한 컨텍스트를 생성하는 함수
    :param question: 사용자 질문 문자열
    :param vector_store_manager: VectorStoreManager 인스턴스
    :return: 생성된 컨텍스트 문자열
    """

    MAX_TOTAL_LENGTH = 400  # 원하는 최대 길이로 설정
    TOP_K = 1  # 가장 유사한 한 개의 문서를 검색
    SIMILARITY_THRESHOLD = 0.7  # 유사도 임계값

    # 질문 임베딩
    question_embedding = vector_store_manager.get_embedding(question).reshape(1, -1)
    faiss.normalize_L2(question_embedding)

    # 유사도 검색 (k=1로 설정)
    k = TOP_K
    D, I = vector_store_manager.vector_store.index.search(question_embedding, k)

    # 유사도 점수와 문서 매핑
    docs_and_scores = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        doc_id = vector_store_manager.vector_store.index_to_docstore_id[idx]
        doc = vector_store_manager.vector_store.docstore.get(doc_id)
        if doc:
            docs_and_scores.append((doc, score))

    if not docs_and_scores:
        logging.warning("유사한 문서를 찾을 수 없습니다.")
        return ""
    
    # 유사도 임계값 검사
    top_score = docs_and_scores[0][1]
    if top_score < SIMILARITY_THRESHOLD:
        logging.info(f"유사도 {top_score}이(가) 임계값 {SIMILARITY_THRESHOLD}보다 낮아 빈 컨텍스트를 반환합니다.")
        return ""

    # 컨텍스트 생성
    context = ""
    total_length = 0

    for i, (doc, score) in enumerate(docs_and_scores):
        section_content = doc.page_content

        # 할당할 문자 수 계산 (전체 길이를 초과하지 않도록)
        allocated_length = MAX_TOTAL_LENGTH

        # 내용 자르기 (문장의 중간에서 자르지 않도록)
        truncated_content = truncate_text(section_content, allocated_length)
        actual_length = len(truncated_content)

        if total_length + actual_length > MAX_TOTAL_LENGTH:
            # 남은 길이에 맞게 조정
            remaining_length = MAX_TOTAL_LENGTH - total_length
            if remaining_length <= 0:
                break  # 최대 길이에 도달했으므로 루프 종료
            truncated_content = truncate_text(section_content, remaining_length)
            context += truncated_content
            total_length += len(truncated_content)
            break  # 최대 길이에 도달했으므로 루프 종료
        else:
            context += truncated_content
            total_length += actual_length

    return context
