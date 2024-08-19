# DACON_Finance_AI_Search
# 사용 라이브러리 및 용도
## PyMuPDF (`fitz`)
PDF 파일을 읽고 텍스트를 추출하는 라이브러리입니다. PDF 문서에서 텍스트 데이터를 추출하여 자연어 처리 작업에 사용할 수 있도록 합니다.

## Transformers (`transformers`)
자연어 처리 모델을 쉽게 사용할 수 있도록 해주는 라이브러리입니다. 사전 학습된 언어 모델을 사용하여 텍스트 임베딩, 텍스트 생성, 텍스트 분류 등의 작업을 수행합니다.

## Accelerate (`accelerate`)
모델 학습 및 추론을 위한 멀티-GPU 및 TPU 지원을 쉽게 구현할 수 있도록 돕는 라이브러리입니다. 대규모 모델의 효율적인 학습 및 추론을 지원합니다.

## Langchain (`langchain`)
다양한 언어 모델과 자연어 처리 도구를 결합하여 강력한 애플리케이션을 구축할 수 있게 해주는 프레임워크입니다. 언어 모델을 활용한 응용 프로그램을 쉽게 만들고, 데이터베이스와 결합하여 검색과 생성 작업을 통합합니다.

## FAISS (`faiss`)
대규모 벡터 검색을 효율적으로 수행하는 라이브러리입니다. 대규모 데이터에서 유사한 벡터를 빠르게 검색하여 관련 정보를 찾습니다.

## Sentence-Transformer (`sentence-transformer`)
문장을 임베딩 벡터로 변환하는 라이브러리입니다. 문장 간의 유사도를 계산하거나 검색 작업을 수행합니다.

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

# KIL
## 청크로 분할하는게 뭐지?
청크로 분할하는 것은 긴 텍스트를 작은 조각으로 나누는 것을 의미합니다. 이렇게 하는 이유는 긴 텍스트를 한 번에 처리하기 어려울 때, 작은 조각으로 나누어 처리하기 위함입니다. 각 청크는 독립적으로 검색되고 처리될 수 있어 효율적입니다.

> chunk_size: 각 청크의 최대 길이
> chunk_overlab: 각 청크 간의 겹치는 부분의 길이를 지정

## 임베딩을 정규화 하는게 뭐야?
임베딩을 정규화 하는 것은 벡터의 크기를 조정하여 일정한 크기로 만드는 것을 의미합니다. 이는 벡터의 길이가 일정해져서, 벡터 간의 비교가 더 공정하고 일관되게 이루어지도록 합니다. 주로 코사인 유사도를 계산할 때 용이합니다.

## FAISS 벡터 데이터베이스가 뭐고 왜 쓰는거지?
FAISS(Facebook AI Similarity Search)는 대규모 벡터 데이터베이스를 효율적으로 검색하기 위한 라이브러리입니다. 벡터 검색을 통해 문서 간의 유사도를 빠르게 계산하고, 관련 있는 문서를 찾을 수 있게 합니다. 이렇게 하면 질문에 대해 관련성이 높은 문서를 빠르게 찾을 수 있습니다.

## 문자열을 유니코드로 정규화 하는 이유는?
문자열을 유니코드로 정규화하는 것은 텍스트를 표준 형태로 변환하는 것입니다. 이렇게 하면 같은 문자라도 다른 방식으로 표현된 것을 동일하게 처리할 수 있습니다.

## 리트리버가 뭐야? 각 PDF 문서에 대한 정보를 따로 저장하는건가?
리트리버는 질문에 대해 관련 문서를 검색하는 역할을 합니다. 각 PDF 문서의 정보를 미리 벡터 데이터베이스에 저장해 두고, 사용자가 질문을 하면 그 질문과 관련된 문서를 데이터베이스에서 찾아내는 역할을 합니다. 각 PDF 문서에 대한 벡터 데이터베이스와 리트리버를 따로 생성하고 저장하여 필요할 때 검색에 사용합니다.

## 벡터 데이터베이스
벡터 데이터베이스의 역할은 텍스트 데이터를 벡터로 변환하여 저장하고, 유사한 벡터를 효율적으로 검색하는 것입니다. 질문이나 문서와 유사한 문서를 찾기 위해 텍스트를 벡터로 변환하고, 이를 기반으로 유사도를 계산하여 관련 문서를 찾습니다.

## LLM 파이프라인을 구성하는 이유와 용도는?
LLM 파이프라인을 구성하는 이유는 질문에 대해 자연스러운 답변을 생성하기 위해서입니다. 파이프라인은 질문을 입력으로 받아 관련 정보를 찾아내고, 이를 기반으로 답변을 생성하는 일련의 과정을 자동화 합니다. 이렇게 하면 복잡한 질문에 대해 인공지능 모델이 자동으로 답변을 생성할 수 있습니다.

## 4비트 양자화는?
4비트 양자화는 모델의 가중치를 4비트로 표현하여 모델의 크기를 줄이고, 연산 속도를 높이는 방법입니다. 이렇게 하면 메모리 사용량을 줄이고, 모델을 더 효율적으로 실행할 수 있습니다. 특히, 큰 모델을 작은 디바이스에서 실행할 때 유용합니다.

## 문자열을 포맷팅하는 이유는?
문자열을 포맷팅하는 이유는 검색된 문서나 데이터를 보기 좋게 정리하고, 인공지능 모델이 이해하기 쉽게 만들기 위해서입니다. 예를 들어, 여러 문서에서 검색된 텍스트를 하나의 문자열로 연결하여 모델에 입력하면, 모델이 질문에 대해 더 잘 이해하고 답변할 수 있습니다.

## 임베딩 모델
임베딩 모델은 텍스트 데이터를 고차원 벡터 공간의 벡터로 변환하는 역할을 합니다. 이 벡터는 텍스트의 의미를 수치적으로 표현한 것입니다. 이를 통해 문장이나 단어 간의 유사도를 계산할 수 있습니다.

### 과정
1. 텍스트 전처리 (토크나이제이션, 토큰 인덱싱, 패딩과 마스킹)
2. 벡터 생성 (모델 입력, 벡터 추출)
3. 벡터 정규화

## FAISS 대체품
FAISS는 대규모 벡터 검색에 최적화된 라이브러리입니다. 그러나 다른 벡터 검색 라이브러리들도 존재합니다.

1. Annoy (Approximate Nearest Neighbors Oh Yeah)
Annoy는 Spotify에서 개발한 라이브러리로, 높은 차원의 벡터 공간에서 근사 최근접 이웃 검색을 수행합니다. 메모리 내에서 동작하며, 검색 속도가 빠릅니다. 또한, 데이터셋을 디스크에 저장하고 로드할 수 있습니다. 추천 시스템이나 유사한 아이템 검색에 사용됩니다.

2. HNSWlib (Hierarchical Navigable Small World)
HNSWlib는 비슷한 검색 문제를 해결하기 위한 C++ 라이브러리로, 파이썬 바인딩을 제공합니다. 고차원 데이터에서 매우 빠르고 정확한 최근접 이웃 검색을 지원합니다. 이미지 검색이나 텍스트 유사도 검색에 사용됩니다.

3. ScaNN (Scalable Nearest Neighbors)
Google에서 개발한 ScaNN은 대규모 및 고차원 벡터 검색에 최적화된 라이브러리입니다. 높은 검색 정확도와 빠른 검색 속도를 제공합니다. TensorFlow와 통합되어 머신러닝 워크플로우에 쉽게 적용할 수 있습니다. 텍스트 검색이나 이미지 검색에 사용됩니다.

4. NMSLIB (Non-Metric Space Library)
NMSLIB는 비유클리드 공간에서의 최근접 이웃 검색을 위한 라이브러리입니다. 다양한 거리 측정 방법을 지원하며, 매우 빠른 검색 속도를 제공합니다. 추천 시스템이나 유사한 문서 검색에 사용됩니다.

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