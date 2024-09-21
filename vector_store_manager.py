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
        title = title_parts[2] if len(title_parts) >= 3 else title_parts[-1]

        logging.debug(f"추출된 제목: {title}")

        if not text.strip():
            logging.debug("문서 내용이 비어있어 문서를 제외합니다.")
            return []

        # `{toc}`로 텍스트 분할
        toc_split = text.split('{toc}', 1)
        text_before_toc = toc_split[0]
        text_after_toc = toc_split[1] if len(toc_split) == 2 else ''

        logging.debug("`{toc}` 이전의 텍스트를 처리합니다.")

        # Include 내용 추출 및 처리
        include_pattern = re.compile(r'\{include:틀:([^\}]+)\}')
        includes = include_pattern.findall(text_before_toc)
        logging.debug(f"추출된 include: {includes}")

        include_contents = []
        for include in includes:
            include_content = self.get_include_content(include)
            if include_content:
                include_contents.append(include_content)
                logging.debug(f"포함된 내용 추가: {include_content[:30]}...")

        # 표 추출 및 처리
        table_pattern = re.compile(r'^\|.*\n(?:\|.*\n)*', re.MULTILINE)
        table_matches = table_pattern.findall(text_before_toc)
        logging.debug(f"추출된 표의 개수: {len(table_matches)}")

        table_contents = []
        for table_text in table_matches:
            table_cleaned = self.clean_table(table_text)
            table_contents.append(table_cleaned)
            logging.debug(f"표 내용 추가: {table_cleaned[:30]}...")

        # "개요" 섹션 추출 (메인 Document에 포함)
        overview_pattern = re.compile(r'#\s*개요\s*(.*?)(?=\n#|$)', re.DOTALL)
        overview_match = overview_pattern.search(text_after_toc)
        overview_content = overview_match.group(1).strip() if overview_match else ''
        if overview_content:
            logging.debug(f"개요 섹션 내용 추출: {overview_content[:30]}...")

        # Include 내용, 표, 개요 섹션 내용을 모두 결합하여 메인 Document 생성
        combined_content = "\n".join(include_contents + table_contents + [overview_content])

        documents = []
        if combined_content.strip():
            cleaned_content = self.clean_text(combined_content)
            title_embedding = np.array(self.embedding_function.embed_query(title), dtype='float32')
            content_embedding = np.array(self.embedding_function.embed_query(cleaned_content), dtype='float32')
            combined_embedding = 0.5 * title_embedding + 0.5 * content_embedding
            combined_embedding /= np.linalg.norm(combined_embedding)

            main_doc = Document(page_content=cleaned_content)
            main_doc.metadata = {
                'title': title,
                'source': title,
                'embedding': combined_embedding
            }
            documents.append(main_doc)
            logging.debug(f"메인 Document 생성 완료: 제목={title}")

        # Include와 표를 원본 텍스트에서 제거
        text_before_toc = include_pattern.sub('', text_before_toc)
        text_before_toc = table_pattern.sub('', text_before_toc)

        # 나머지 텍스트 처리 (개요는 이미 메인 Document에 포함됨)
        text = text_before_toc + '\n{toc}\n' + text_after_toc
        text = self.clean_text(text)

        # 섹션 분할 및 Document 생성 (개요 제외)
        sections = self.split_into_sections(text, title)
        documents.extend(sections)

        return documents

    def get_include_content(self, include_name):
        """
        포함된 내용을 실제로 로드하는 함수
        """
        # Include 파일의 이름 패턴에 맞게 경로 설정
        include_file_pattern = re.compile(r'\d+__\d+__틀-.+\.md')
        directory = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리

        for file in os.listdir(directory):
            if include_file_pattern.match(file) and include_name in file:
                include_file_path = os.path.join(directory, file)
                with open(include_file_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
        logging.warning(f"포함된 파일을 찾을 수 없습니다: 틀-{include_name}.md")
        return f"포함된 내용: {include_name}"

    def clean_text(self, text):
        """
        특수 기호 및 불필요한 문법을 제거하는 함수
        """
        text = re.sub(r'<table[^>]*>', '', text)  # <table ...> 제거
        text = re.sub(r'#\w+', '', text)  # #색코드 제거
        text = re.sub(r'<#\w+>', '', text)  # <#색코드> 제거
        text = re.sub(r'<[wcr][0-9]+>', '', text)  # <w숫자>, <c숫자>, <r숫자> 제거
        text = text.replace('{br}', ' ')  # {br}을 공백으로 대체
        text = re.sub(r'\+\d+', '', text)  # +숫자 제거
        text = re.sub(r'\{include:틀:[^\}]*\}', '', text)  # {include:틀:XXXX} 제거
        text = text.replace('{', '').replace('}', '')
        text = text.replace('[', '').replace(']', '')
        text = text.replace('<', '').replace('>', '')
        text = text.replace('|', '')
        text = re.sub(r'https?://\S+', '', text)  # URL 제거
        text = re.sub(r'[\*\=\-]', '', text)  # '*', '=', '-' 기호 제거
        text = re.sub(r'[ \t]+', ' ', text)  # 연속된 스페이스 및 탭만 제거, 줄 바꿈은 유지
        return text

    def clean_table(self, table_text):
        """
        표를 텍스트로 변환하는 함수
        """
        rows = table_text.strip().split('\n')
        cleaned_rows = []
        for row in rows:
            cells = row.strip('|').split('|')
            cleaned_cells = [self.clean_text(cell.strip()) for cell in cells]
            if len(cleaned_cells) >= 2:
                key = cleaned_cells[0]
                value = cleaned_cells[1]
                cleaned_rows.append(f"{key}: {value}")
            else:
                cleaned_rows.append(" ".join(cleaned_cells))
        return '\n'.join(cleaned_rows)

    def split_into_sections(self, text, title):
        """
        텍스트를 섹션별로 분할하고 Document 리스트를 반환하는 함수
        '개요' 섹션은 이미 메인 Document에 포함되어 있으므로 제외
        """
        pattern = re.compile(r'^\s*(#+)\s*(.*)', re.MULTILINE)
        matches = list(pattern.finditer(text))

        sections = []
        headings = []

        for i, match in enumerate(matches):
            heading_level = len(match.group(1))
            heading_text = match.group(2).strip()

            # '개요' 섹션은 건너뜀
            if heading_text == '개요':
                continue

            # 현재 헤딩 레벨에 맞게 headings 리스트 조정
            if len(headings) >= heading_level:
                headings = headings[:heading_level - 1]
            else:
                # If heading levels are skipped, fill with empty strings
                while len(headings) < heading_level - 1:
                    headings.append('')

            headings.append(heading_text)

            # 제목 생성
            section_title = f"{title}"
            for h in headings:
                if h:
                    section_title += f"의 {h}"

            # 내용 추출
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            if not content:
                continue

            # 특수 기호 및 불필요한 문법 제거
            content = self.clean_text(content)

            # 임베딩 생성
            title_embedding = np.array(self.embedding_function.embed_query(section_title), dtype='float32')
            content_embedding = np.array(self.embedding_function.embed_query(content), dtype='float32')
            combined_embedding = 0.5 * title_embedding + 0.5 * content_embedding
            combined_embedding /= np.linalg.norm(combined_embedding)

            # Document 생성
            doc = Document(page_content=content)
            doc.metadata = {
                'title': section_title,
                'source': title,
                'embedding': combined_embedding
            }
            sections.append(doc)
            logging.debug(f"섹션 Document 생성: 제목={section_title}")

        return sections

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
