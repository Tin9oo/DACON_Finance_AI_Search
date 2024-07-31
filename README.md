# DACON_Finance_AI_Search
# 사용 라이브러리 및 용도
# 단계별 역할
## 1. 라이브러리 설치
## 2. 라이브러리 임포트
## 3. PDF 처리 및 벡터 데이터베이스 생성
# 개발 로그
## 1. cpu로 코드 전환
### 라이브러리 설치 변경
`pytorch`, `faiss` `cpu` 버전 설치
```python
# m1 macbook 전용 (cpu)
!pip install torch
!pip install faiss-cpu
```
### 벡더 db 생성 시 임베딩 모델 파라미터 변경
`JuggingFaceEmbeddings()`에 `model_dwargs`를 `cuda`에서 `cpu`로 변경
```python
def create_vector_db(chunks, model_path="intfloat/multilingual-e5-small"):
    """FAISS DB 생성"""
    # 임베딩 모델 설정
    # model_kwargs = {'device': 'cuda'}
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # FAISS DB 생성 및 반환
    db = FAISS.from_documents(chunks, embedding=embeddings)
    return db
```
### llm pipeline 생성 시 모델 로드 파라미터 변경
`AutoModelForCasualLM.from_pretrained()`에 `quantization_config`를 전달하지 않고 `device_map`을 `auto`에서 `cpu`로 변경
```python
def setup_llm_pipeline():
    # 4비트 양자화 설정 (생략)
    # 모델 ID (생략)
    # 토크나이저 로드 및 설정 (생략)

    # 모델 로드 및 양자화 설정 적용
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # quantization_config=bnb_config,
        # device_map="auto",
        device_map="cpu",
        trust_remote_code=True )

    # HuggingFacePipeline 객체 생성 (생략)

    return hf
```
## 2. colab으로 이전