# DACON_Finance_AI_Search
# 사용 라이브러리 및 용도

# 단계별 역할
## 1. PDF 처리
```python
doc = fitz.open(file_path)
```
PDF 파일을 엽니다.

```python
text = ''
for page in doc:
    text += page.get_text()
```
모든 페이지의 텍스트를 추출합니다.

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunk_temp = splitter.split_text(text)
```
텍스트를 chunk 단위로 분할합니다.
> chunk_size: 
chunk_overlab: 

```python
chunks = [Document(page_content=t) for t in chunk_temp]
```
나눈 조각들을 Document 객체 리스트로 생성하여 chunks로 사용합니다.

## 2. 벡터 데이터베이스 만들기
```python
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
```
임베딩에 전달할 파라미터를 선정합니다.
> 임베딩 정규화란?

```python
embeddings = HuggingFaceEmbeddings(model_name=model_path, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
```
chunk를 벡터로 변환하기 위한 임베딩을 생성합니다.
> 임베딩이란?

```python
db = FAISS.from_documents(chunks, embedding=embeddings)
```
PDF 파일에서 추출한 텍스트 조각들을 벡터로 변환하여 FAISS 벡터 데이터베이스를 만듭니다.

## 3. PDF 문서 처리
```python
def normalize_path(path):
    return unicodedata.normalize('NFC', path)
```
경로 문자열을 유니코드 정규화합니다.

```python
normalized_path = normalize_path(path)
full_path = os.path.normpath(os.path.join(base_directory, normalized_path.lstrip('./'))) if not os.path.isabs(normalized_path) else normalized_path
pdf_title = os.path.splitext(os.path.basename(full_path))[0]
```
경로를 정규화하고 파일명을 추출합니다.

```python
chunks = process_pdf(full_path)
db = create_vector_db(chunks)
retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 8})
```
각 PDF 문서에 대한 벡터 데이터베이스와 리트리버를 생성합니다.

## 4. LLM 파이프라인 설정
```python
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model_id = "beomi/llama-2-ko-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.use_default_system_prompt = False
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
text_generation_pipeline = pipeline(model=model, tokenizer=tokenizer, task="text-generation", temperature=0.2, return_full_text=False, max_new_tokens=128)
hf = HuggingFacePipeline(pipeline=text_generation_pipeline)
```
Llama-2-ko-7b 모델을 로드하고, 4비트 양자화를 설정하여 파이프라인을 구성합니다.

## 5. 질문에 대한 답변 생성
```python
def normalize_string(s):
    return unicodedata.normalize('NFC', s)
```
소스 문자열을 유니코드 정규화합니다.

```python
def format_docs(docs):
    context = ""
    for doc in docs:
        context += doc.page_content
        context += '\n'
    return context
```
검색된 문서의 텍스트를 하나의 문자열로 포맷팅합니다.

```python
source = normalize_string(row['Source'])
question = row['Question']
normalized_keys = {normalize_string(k): v for k, v in pdf_databases.items()}
retriever = normalized_keys[source]['retriever']
template = """
다음 정보를 바탕으로 질문에 답하세요:
{context}
질문: {question}
답변:
"""
prompt = PromptTemplate.from_template(template)
rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
full_response = rag_chain.invoke(question)
results.append({"Source": row['Source'], "Source_path": row['Source_path'], "Question": question, "Answer": full_response})
```
각 질문에 대한 관련 문서를 검색하고, LLM을 사용하여 답변을 생성합니다.

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
## 2. colab으로 이전 후 첫 제출
### google drive mount 및 경로 변경
### 제출 성능
0.2476