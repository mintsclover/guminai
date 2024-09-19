import os
import json
import pickle
import requests
import numpy as np
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import re
import logging
from functools import lru_cache
import yaml

# 로그 설정
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# 환경 변수 및 설정 파일 로드
load_dotenv()

# 설정 파일 로드
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 네이버 클로바 API 설정
CLOVA_API_KEY = os.getenv("CLOVA_API_KEY")
CLOVA_PRIMARY_KEY = os.getenv("CLOVA_PRIMARY_KEY")
CLOVA_REQUEST_ID = os.getenv("CLOVA_REQUEST_ID")
CLOVA_HOST = 'https://clovastudio.stream.ntruss.com'

VECTOR_STORE_PATH = "vector_store.index"

# Clova CompletionExecutor 클래스: 네이버 클로바 API와 상호작용하기 위한 클래스
class CompletionExecutor:
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

# 벡터 스토어 관리 클래스
class VectorStoreManager:
    def __init__(self, embedding_model_name=None):
        # 임베딩 모델 이름을 설정 파일에서 가져옴
        if embedding_model_name is None:
            embedding_model_name = config.get('embedding_model_name', 'jhgan/ko-sroberta-multitask')
        self.embedding_model_name = embedding_model_name
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.vector_store = None

    @lru_cache(maxsize=None)
    def get_embedding(self, text):
        return np.array(self.embedding_function.embed_query(text), dtype='float32')

    # 벡터 스토어 저장
    def save_vector_store(self, path):
        faiss.write_index(self.vector_store.index, path)
        # 필요한 메타데이터도 함께 저장
        with open('docstore.pkl', 'wb') as f:
            pickle.dump((self.vector_store.docstore, self.vector_store.index_to_docstore_id), f)

    # 벡터 스토어 로드
    def load_vector_store(self, path):
        index = faiss.read_index(path)
        with open('docstore.pkl', 'rb') as f:
            docstore, index_to_docstore_id = pickle.load(f)
        self.vector_store = FAISS(
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            embedding_function=self.get_embedding
        )

    # 문서 전처리: 특수 기호 제거 및 내용 정제
    def preprocess_document(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

            # 파일명에서 제목 추출
            title = os.path.splitext(os.path.basename(file_path))[0]
            title_parts = title.split('__')
            if len(title_parts) >= 3:
                title = title_parts[2]
            else:
                title = title_parts[-1]

            # 제목에 '틀'이 포함된 문서 제외
            if '틀' in title:
                return None

            # 내용이 없는 문서 제외
            if not text.strip():
                return None

            # 특수 기호 및 불필요한 문법 제거
            text = re.sub(r'<table[^>]*>', '', text)  # <table ...> 제거
            text = re.sub(r'#\w+', '', text)  # #색코드 제거
            text = re.sub(r'<#\w+>', '', text)  # <#색코드> 제거
            text = re.sub(r'<[wcr][0-9]+>', '', text)  # <w숫자>, <c숫자>, <r숫자> 제거
            text = text.replace('{br}', '')  # {br} 제거
            text = re.sub(r'\+\d+', '', text)  # +숫자 제거
            text = re.sub(r'\{include:틀:[^\}]*\}', '', text)  # {include:틀:XXXX} 제거
            text = text.replace('{', '').replace('}', '')
            text = text.replace('[', '').replace(']', '')
            text = text.replace('<', '').replace('>', '')
            text = text.replace('|', '')
            text = re.sub(r'https?://\S+', '', text)  # URL 제거
            text = re.sub(r'#+', '', text)  # '#' 기호 제거
            text = re.sub(r'[\*\=\-]', '', text)  # '*', '=', '-' 기호 제거
            text = re.sub(r'\s+', ' ', text)  # 연속된 공백 제거

            # 문서를 '#'로 분할하여 섹션별로 저장
            sections = re.split(r'\n#+\s*', text)
            # 헤딩 추출
            headings = re.findall(r'\n(#+\s*.*)', text)
            headings = [re.sub(r'#+\s*', '', h).strip() for h in headings]

            # 섹션과 헤딩 매핑
            content_sections = []
            for i, section_content in enumerate(sections):
                if i == 0:
                    if section_content.strip():
                        content_sections.append(('Introduction', section_content.strip()))
                else:
                    heading = headings[i - 1]
                    content_sections.append((heading, section_content.strip()))

            # 임베딩 계산
            embeddings = []
            weights = []

            # 제목 임베딩
            title_embedding = np.array(self.embedding_function.embed_query(title), dtype='float32')
            embeddings.append(title_embedding)
            weights.append(0.5)

            # 개요 섹션 확인
            overview_index = next((i for i, (h, _) in enumerate(content_sections) if h == '개요'), None)
            if overview_index is not None:
                overview_content = content_sections[overview_index][1]
                overview_embedding = np.array(self.embedding_function.embed_query(overview_content), dtype='float32')
                embeddings.append(overview_embedding)
                weights.append(0.3)
                # 개요 섹션 제거
                content_sections.pop(overview_index)
            else:
                overview_content = None

            # 남은 가중치 계산
            remaining_weight = 1.0 - sum(weights)
            num_sections = len(content_sections)
            if num_sections > 0:
                # 섹션 순서에 따라 가중치 분배 (역수)
                total_inverse = sum(1 / (i + 1) for i in range(num_sections))
                for idx, (heading, content) in enumerate(content_sections):
                    weight = remaining_weight * ((1 / (idx + 1)) / total_inverse)
                    section_embedding = np.array(self.embedding_function.embed_query(content), dtype='float32')
                    embeddings.append(section_embedding)
                    weights.append(weight)
            else:
                # 섹션이 없는 경우 제목 가중치 증가
                weights[0] += remaining_weight

            # 가중치 합산
            combined_embedding = np.zeros_like(embeddings[0])
            for emb, weight in zip(embeddings, weights):
                combined_embedding += emb * weight

            # 임베딩 벡터 정규화
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

            # Document 생성 (섹션 정보 저장)
            doc = Document(page_content=text)
            doc.metadata = {
                'title': title,
                'sections': content_sections,
                'embedding': combined_embedding
            }
            return doc

    # 마크다운 파일로부터 벡터 스토어 생성
    def create_vector_store_from_markdown(self, files, folder_path):
        docs = []

        # 파일 목록이 제공되지 않은 경우 폴더 내의 모든 .md 파일 사용
        if not files:
            files = [f for f in os.listdir(folder_path) if f.endswith('.md') and '틀' not in f]

        # 각 마크다운 파일을 전처리하고 Document로 생성
        for file in files:
            file_path = os.path.join(folder_path, file)
            doc = self.preprocess_document(file_path)
            if doc:  # 유효한 문서만 추가
                docs.append(doc)

        if not docs:
            raise ValueError("유효한 문서가 없습니다.")

        # 임베딩 추출
        embeddings = [doc.metadata['embedding'] for doc in docs]
        embeddings = np.array(embeddings, dtype='float32')
        faiss.normalize_L2(embeddings)

        # FAISS 인덱스 생성 (코사인 유사도 사용)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # 벡터 스토어 생성
        self.vector_store = FAISS(
            index=index,
            docstore={str(i): docs[i] for i in range(len(docs))},
            index_to_docstore_id={i: str(i) for i in range(len(docs))},
            embedding_function=self.embedding_function.embed_query
        )

    # 벡터 스토어 가져오기 함수
    def get_vector_store(self, files=None, folder_path=None):
        if os.path.exists(VECTOR_STORE_PATH):
            logging.info("벡터 스토어를 로드합니다.")
            self.load_vector_store(VECTOR_STORE_PATH)
        else:
            logging.info("벡터 스토어를 생성합니다.")
            # 설정 파일에서 files와 folder_path를 가져옴
            if folder_path is None:
                folder_path = config.get('folder_path', 'wiki')
            if files is None:
                files = config.get('files', None)
            if not files:
                # 폴더에서 .md 파일 목록 가져오기
                files = [f for f in os.listdir(folder_path) if f.endswith('.md') and '틀' not in f]
            self.create_vector_store_from_markdown(files, folder_path)
            self.save_vector_store(VECTOR_STORE_PATH)
        return self.vector_store

# 클로바 API를 사용하여 질문에 답변 생성
def ask_clova(question, context, executor):
    # 프롬프트 생성
    prompt = f"질문: {question}\n답변을 위한 보조 자료: {context}"
    preset_text = [
        {
            "role": "system",
            "content": "당신은 수민국 수민특별시 성내구 시작동의 한 주민입니다. 시작동에 대해 자부심이 넘치고, 옆 구인 성동구를 싫어합니다. 기본적으로 반말을 사용하며 친절하지만, 성동구를 칭찬하는 이야기가 나오면 인상을 찌푸립니다."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    request_data = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 256,
        'temperature': 0.5,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }
    response = executor.execute(request_data)
    return response

# 챗봇과 사용자 간의 대화 함수
def chat_with_resident(vector_store_manager, executor, max_total_length=1500):
    print("챗봇: 안녕하세요! 가상 국가에 대해 궁금한 점이 있나요?")
    while True:
        question = input("나: ")
        if question.lower() == "종료":
            print("챗봇: 대화를 종료합니다.")
            break

        try:
            # 질문 임베딩
            question_embedding = vector_store_manager.get_embedding(question).reshape(1, -1)
            faiss.normalize_L2(question_embedding)

            # 유사도 검색
            k = config.get('top_k', 5)
            D, I = vector_store_manager.vector_store.index.search(question_embedding, k)

            # 유사도 점수와 문서 매핑
            docs_and_scores = []
            for idx, score in zip(I[0], D[0]):
                if idx == -1:
                    continue
                doc_id = vector_store_manager.vector_store.index_to_docstore_id[idx]
                doc = vector_store_manager.vector_store.docstore[doc_id]
                docs_and_scores.append((doc, score))

            # 문서 매핑 및 컨텍스트 생성
            max_total_length = config.get('max_total_length', 1500)
            alpha = config.get('alpha', 1.0) # 유사도 점수의 영향력 조절
            total_length = 0

            # 유사도 점수와 문서 길이를 기반으로 할당된 길이 계산
            similarity_scores = np.array([score for doc, score in docs_and_scores])
            doc_lengths = np.array([len(doc.page_content) for doc, score in docs_and_scores])
            length_factors = doc_lengths / doc_lengths.sum()

            adjusted_scores = similarity_scores ** alpha
            allocated_lengths = (adjusted_scores / adjusted_scores.sum()) * max_total_length

            context = ""

            for idx, (doc, score) in enumerate(docs_and_scores):
                allocated_length = int(allocated_lengths[idx])

                # 섹션별로 내용을 추가하되, 섹션 중간에서 자르지 않음
                sections = doc.metadata.get('sections', [])
                content = ""
                length_used = 0
                for heading, section_content in sections:
                    section_length = len(section_content)
                    if length_used + section_length <= allocated_length:
                        content += f"#{heading}\n{section_content}\n"
                        length_used += section_length
                    else:
                        # 섹션이 전체 할당된 길이를 초과하는 경우
                        if length_used == 0:
                            # 할당된 길이만큼 섹션 내용을 자름
                            truncated_content = section_content[:allocated_length]
                            content += f"#{heading}\n{truncated_content}\n"
                            length_used += len(truncated_content)
                        break

                if length_used == 0:
                    continue  # 이 문서에서 추가된 내용이 없음

                total_length += length_used
                if total_length > max_total_length:
                    break  # 전체 컨텍스트 길이 초과

                context += f"{content}\n---\n"

                # 사용된 문서의 전문 로그 출력
                logging.info(f"유사도 순위: {idx + 1}, 점수: {score}, 제목: {doc.metadata['title']}")
                logging.info(f"내용:\n{content}\n")

            if context:
                # 클로바를 사용하여 답변 생성
                answer = ask_clova(question, context, executor)
                print(f"챗봇: {answer}")
            else:
                print("챗봇: 죄송합니다. 해당 질문에 대한 답변을 찾을 수 없습니다.")

        except Exception as e:
            logging.error(f"오류가 발생했습니다: {e}")
            print("챗봇: 죄송합니다. 현재 서비스를 제공할 수 없습니다.")

if __name__ == "__main__":
    # 벡터 스토어 매니저 초기화
    vector_store_manager = VectorStoreManager()

    # 벡터 스토어 생성 또는 로드
    vector_store_manager.get_vector_store()

    # 클로바 Completion Executor 초기화
    completion_executor = CompletionExecutor(
        host=CLOVA_HOST,
        api_key=CLOVA_API_KEY,
        api_key_primary_val=CLOVA_PRIMARY_KEY,
        request_id=CLOVA_REQUEST_ID
    )

    # 챗봇 대화 시작
    chat_with_resident(vector_store_manager, completion_executor, max_total_length=config.get('max_total_length', 1500))