import os
import numpy as np
import pickle
import faiss
import re
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from functools import lru_cache

# 설정 파일 로드
import yaml
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

VECTOR_STORE_PATH = config.get('vector_store_path', 'vector_store.index')

class VectorStoreManager:
    """
    벡터 스토어를 관리하는 클래스
    """
    def __init__(self, embedding_model_name=None):
        if embedding_model_name is None:
            embedding_model_name = config.get('embedding_model_name', 'jhgan/ko-sroberta-multitask')
        self.embedding_model_name = embedding_model_name
        self.embedding_function = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.vector_store = None

    @lru_cache(maxsize=None)
    def get_embedding(self, text):
        return np.array(self.embedding_function.embed_query(text), dtype='float32')

    def save_vector_store(self, path):
        """
        벡터 스토어를 파일로 저장하는 함수
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        faiss.write_index(self.vector_store.index, path)
        with open('docstore.pkl', 'wb') as f:
            pickle.dump((self.vector_store.docstore, self.vector_store.index_to_docstore_id), f)

    def load_vector_store(self, path):
        """
        벡터 스토어를 파일에서 로드하는 함수
        """
        index = faiss.read_index(path)
        with open('docstore.pkl', 'rb') as f:
            docstore, index_to_docstore_id = pickle.load(f)
        self.vector_store = FAISS(
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            embedding_function=self.embedding_function
        )

    def preprocess_document(self, file_path):
        """
        문서를 전처리하여 Document 객체로 변환하는 함수
        """
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
                return []

            # 내용이 없는 문서 제외
            if not text.strip():
                return []

            # 특수 기호 및 불필요한 문법 제거
            text = self.clean_text(text)

            # 섹션 분할
            pattern = re.compile(r'^(#+)\s*(.*)', re.MULTILINE)
            matches = pattern.finditer(text)

            sections = []
            last_index = 0
            headings = []

            for match in matches:
                start, end = match.span()
                if last_index < start:
                    content = text[last_index:start]
                    sections.append((headings.copy(), content.strip()))
                level = len(match.group(1))
                heading_text = match.group(2).strip()

                # 현재 헤딩 레벨에 맞게 headings 리스트 조정
                if len(headings) >= level:
                    headings = headings[:level-1]
                headings.append(heading_text)
                last_index = end

            # 마지막 섹션 추가
            if last_index < len(text):
                content = text[last_index:]
                sections.append((headings.copy(), content.strip()))

            # 각 섹션별로 Document 생성
            documents = []
            for heading_list, section_content in sections:
                if not section_content.strip():
                    continue

                # 섹션 제목 생성 (헤딩들을 연결)
                section_title = title + '의 ' + ' - '.join(heading_list)

                # 특수 기호 및 불필요한 문법 제거
                section_content = re.sub(r'#+', '', section_content)
                # section_content_clean = self.clean_text(section_content)

                # 제목과 내용에 각각 0.5의 가중치를 부여하여 임베딩 생성
                title_embedding = np.array(self.embedding_function.embed_query(section_title), dtype='float32')
                content_embedding = np.array(self.embedding_function.embed_query(section_content), dtype='float32')
                combined_embedding = 0.5 * title_embedding + 0.5 * content_embedding
                combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

                # Document 생성
                doc = Document(page_content=section_content)
                doc.metadata = {
                    'title': section_title,
                    'source': title,  # 원래 문서의 제목
                    'embedding': combined_embedding
                }
                documents.append(doc)

            return documents

    def clean_text(self, text):
        """
        특수 기호 및 불필요한 문법을 제거하는 함수
        """
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
        # text = re.sub(r'#+', '', text)  # '#' 기호 제거
        text = re.sub(r'[\*\=\-]', '', text)  # '*', '=', '-' 기호 제거
        text = re.sub(r'\s+', ' ', text)  # 연속된 공백 제거
        return text

    def create_vector_store_from_markdown(self, files, folder_path):
        """
        마크다운 파일로부터 벡터 스토어를 생성하는 함수
        """
        all_docs = []

        # 파일 목록이 제공되지 않은 경우 폴더 내의 모든 .md 파일 사용
        if not files:
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.md') and '틀' not in f]

        # 파일 목록 디버깅 로그 출력
        logging.info(f"처리할 파일 목록: {files}")

        # 각 마크다운 파일을 전처리하고 Document로 생성
        for file in files:
            file_path = os.path.join(folder_path, file)
            if not os.path.isfile(file_path):
                logging.warning(f"파일이 존재하지 않습니다: {file_path}")
                continue
            try:
                docs = self.preprocess_document(file_path)
                all_docs.extend(docs)  # 여러 Document를 추가
            except Exception as e:
                logging.error(f"파일 처리 중 오류 발생: {file_path}, 오류: {e}")

        if not all_docs:
            raise ValueError("유효한 문서가 없습니다.")

        # 임베딩 추출
        embeddings = [doc.metadata['embedding'] for doc in all_docs]
        embeddings = np.array(embeddings, dtype='float32')
        faiss.normalize_L2(embeddings)

        # FAISS 인덱스 생성 (코사인 유사도 사용)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        # 벡터 스토어 생성
        self.vector_store = FAISS(
            index=index,
            docstore={str(i): all_docs[i] for i in range(len(all_docs))},
            index_to_docstore_id={i: str(i) for i in range(len(all_docs))},
            embedding_function=self.embedding_function
        )

    def get_vector_store(self, files=None, folder_path=None, force_create=False):
        """
        벡터 스토어를 가져오는 함수
        """
        if not force_create and os.path.exists(VECTOR_STORE_PATH):
            logging.info("벡터 스토어를 로드합니다.")
            self.load_vector_store(VECTOR_STORE_PATH)
        else:
            logging.info("벡터 스토어를 생성합니다.")
            # 설정 파일에서 files와 folder_path를 가져옴
            if folder_path is None:
                folder_path = config.get('folder_path', 'wiki')
            if files is None or files == 'None':
                files = None  # files가 'None' 문자열인 경우 실제 None으로 설정
            if not files:
                # 폴더에서 .md 파일 목록 가져오기
                files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.md') and '틀' not in f]
            self.create_vector_store_from_markdown(files, folder_path)
            self.save_vector_store(VECTOR_STORE_PATH)
        return self.vector_store
