import os
import numpy as np
from vector_store_manager import VectorStoreManager  # 클래스 임포트

# 시뮬레이션을 위한 코드 정의
class DocumentSimulator:
    def __init__(self, embedding_function):
        self.vector_store_manager = VectorStoreManager(embedding_model_name=None)  # VectorStoreManager 인스턴스 생성
        self.vector_store_manager.embedding_function = embedding_function  # 임베딩 함수 설정

    def simulate_preprocessing(self, file_path):
        # VectorStoreManager의 preprocess_document 호출
        documents = self.vector_store_manager.preprocess_document(file_path)
        
        # 결과 출력
        if not documents:
            print("No documents were generated. Please check the preprocessing steps.")
        for i, doc in enumerate(documents, 1):
            print(f"Document {i}:")
            print(f"Title: {doc.metadata['title']}")
            print(f"Source: {doc.metadata['source']}")
            print(f"Content: {doc.page_content}")
            print(f"Embedding (first 5 elements): {doc.metadata['embedding'][:5]}")  # 임베딩의 앞 5개 요소만 출력
            print("-" * 50)

# 수정된 임베딩 함수
class DummyEmbeddingFunction:
    def embed_query(self, text):
        # 임베딩 크기를 10으로 고정
        embedding = [ord(char) for char in text][:10]  # 첫 10개의 문자로 제한
        # 임베딩 벡터의 크기를 10으로 맞추기 위해, 부족한 경우 0으로 채움
        return embedding + [0] * (10 - len(embedding))

# 샘플 파일을 테스트할 수 있는 코드
if __name__ == "__main__":
    # 예시 파일 경로 설정 (이 경로를 실제 파일 경로로 교체해야 함)
    sample_file_path = "wiki/28696__45964__수덕궁.md"
    
    # 임베딩 함수 객체 생성
    embedding_function = DummyEmbeddingFunction()

    # 시뮬레이터 객체 생성 및 전처리 시뮬레이션 실행
    simulator = DocumentSimulator(embedding_function)
    simulator.simulate_preprocessing(sample_file_path)
