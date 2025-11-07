# Phase 2 Preparation Guide (Days 5-10)

> **목표**: Phase 2 개발을 시작하기 전에 꼭 필요한 준비 사항들을 정리한 문서입니다.
> **기간**: 2025-11-08 이후, Phase 2 시작 전
> **상태**: 준비 필수 ✅

---

## 🎯 Phase 2 개요

**Phase 2: Feature Development & Testing (Days 5-10)**

- **Day 5**: Testing & Debugging (현재 FastAPI 구조 검증)
- **Day 6-7**: Ingest Pipeline (데이터셋 다운로드 + 이미지 전처리)
- **Day 8-9**: Search API & Indexing (검색 엔드포인트 + FAISS 인덱싱)
- **Day 10**: Performance Optimization (캐싱, 배치 최적화)

---

## ✅ 필수 준비사항

### 1. OpenAI API Key 설정

**왜 필요한가?**
- OpenAI Embeddings API (text-embedding-3-small): 쿼리/이미지 캡셔닝을 벡터로 변환
- Day 8-9에서 검색 기능을 구현할 때 필수

**준비 방법:**

```bash
# 1. OpenAI 계정 생성 (https://platform.openai.com/)
# 2. API 키 발급 (https://platform.openai.com/account/api-keys)
# 3. 크레딧 충전 ($5 이상 권장)

# 4. 프로젝트에 .env 파일 생성
cd C:\work\project\image-search
echo OPENAI_API_KEY=sk-proj-xxxxx > .env

# 5. .env 파일 무시 (Git에 커밋되지 않도록)
# .gitignore에 이미 .env가 추가되어 있음
```

**비용 예상:**
- text-embedding-3-small: $0.00002 per 1K tokens
- 테스트용 1,000개 이미지 임베딩: ~$0.02

### 2. app/data 폴더 생성

```bash
# 데이터 저장 폴더 생성
mkdir C:\work\project\image-search\app\data

# 폴더 구조:
# app/data/
#   ├── raw/              (다운로드한 원본 이미지)
#   ├── processed/        (전처리된 이미지)
#   ├── embeddings.pkl    (FAISS 인덱스 저장)
#   └── images.db         (SQLite 데이터베이스)
```

### 3. Python 환경 검증

```bash
# 가상환경 활성화
cd C:\work\project\image-search
venv\Scripts\activate

# 핵심 패키지 임포트 테스트
python -c "import fastapi, openai, faiss, torch, transformers; print('OK')"

# FastAPI 서버 정상 작동 확인
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
# http://localhost:8000/docs 접속 가능한지 확인
```

---

## 📚 권장 준비사항

### 1. 문서 읽기

**순서대로 읽기:**

1. **ARCHITECTURE.md** (5분)
   - 시스템 전체 구조 이해
   - 데이터 흐름 파악

2. **ROADMAP.md** (5분)
   - 3주 개발 일정 확인
   - Day 5-10 작업 범위 이해

3. **IMPLEMENTATION_DECISIONS.md** (3분)
   - 선택된 기술 스택 이유
   - 마이그레이션 경로 확인

### 2. OpenAI Embeddings 이해

```python
# 간단한 테스트 코드
from openai import OpenAI

client = OpenAI(api_key="sk-proj-xxxxx")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="빨간색 여름 드레스"
)

embedding = response.data[0].embedding
print(f"Embedding 차원: {len(embedding)}")  # 1536차원
```

### 3. FAISS 인덱싱 이해

```python
# FAISS 기본 사용법
import faiss
import numpy as np

# 인덱스 생성 (1536차원 - OpenAI embedding 크기)
index = faiss.IndexFlatL2(1536)

# 더미 벡터 추가
dummy_vectors = np.random.random((10, 1536)).astype('float32')
index.add(dummy_vectors)

# 검색
query_vector = np.random.random((1, 1536)).astype('float32')
distances, indices = index.search(query_vector, k=5)
print(f"상위 5개 결과 인덱스: {indices[0]}")
```

### 4. SQL 쿼리 기본기

Day 8-9에서 이미지 메타데이터를 조회할 때 필요:

```python
from sqlalchemy.orm import Session
from app.common.database import Image

# 기본 조회
images = db.query(Image).filter(
    Image.attributes['color'].astext == 'red'  # JSON 필터링
).limit(10).all()

# 임베딩 기반 검색 후 메타데이터 조회
image_ids = [1, 2, 3]  # FAISS 검색 결과
images = db.query(Image).filter(Image.id.in_(image_ids)).all()
```

---

## 🔧 Phase 2 특화 준비

### 1. 데이터셋 다운로드 준비 (Day 6)

**사용 데이터셋:**
- **Fashion-MNIST**: 70,000개 이미지 (28x28 흑백)
  - 다운로드: https://github.com/zalandoresearch/fashion-mnist
  - 크기: ~12MB

- **Polyvore**: 패션 코디네이션 데이터셋 (선택사항)
  - 다운로드: https://github.com/xthan/polyvore-outfits
  - 크기: ~2GB (선택사항)

**준비:**
```bash
# 스크립트 위치
app/scripts/download_sample_images.py

# 실행 예상:
# python app/scripts/download_sample_images.py
# 결과: app/data/raw/ 폴더에 이미지 저장
```

### 2. 이미지 처리 파이프라인 준비 (Day 7)

**생성될 파일들:**
```
app/ingest/
  ├── preprocessor.py      (이미지 크기 조정, 정규화)
  ├── captioner.py         (하드코딩된 캡셔닝)
  ├── extractor.py         (속성 추출)
  ├── palette_extractor.py (색상 팔레트 추출)
  └── pipeline.py          (통합 파이프라인)
```

**알아야 할 개념:**
- PIL/Pillow: 이미지 처리
- 색상 팔레트: K-means 클러스터링으로 주요 색상 추출
- 속성: 7개 필드 (color, material, style, season, brand, size, pattern)

### 3. 검색 API 준비 (Day 8-9)

**생성될 파일들:**
```
app/index/
  ├── embedder.py          (OpenAI API 호출)
  ├── faiss_index.py       (FAISS 인덱스 관리)
  └── build.py             (배치 인덱싱)

app/api/
  └── search.py            (새 라우터)
```

**생성될 엔드포인트:**
- POST `/search/text` - 텍스트 검색
- GET `/asset/{id}` - 이미지 메타데이터 조회
- POST `/feedback` - 검색 결과 피드백

### 4. 성능 최적화 준비 (Day 10)

**최적화 항목:**
- 임베딩 캐싱 (Redis 또는 로컬 파일)
- 배치 임베딩 처리 (1회에 여러 이미지)
- DB 인덱싱 (image.id, feedback.query)
- 검색 성능 벤치마크

---

## 📋 사전 체크리스트

Phase 2를 시작하기 전에 다음을 확인하세요:

### 환경 설정
- [ ] OpenAI API Key 발급 및 .env 파일에 저장
- [ ] app/data 폴더 생성
- [ ] Python 가상환경 활성화
- [ ] 모든 패키지 설치 확인 (`pip list | grep -E "openai|faiss|fastapi"`)

### 코드 검증
- [ ] FastAPI 서버 실행 테스트 (`python -m uvicorn app.api.main:app`)
- [ ] 헬스 체크 엔드포인트 확인 (GET http://localhost:8000/health)
- [ ] DB 초기화 테스트 (app/data/images.db 생성 확인)
- [ ] 모든 임포트 테스트 (`python -c "from app import *"`)

### 문서 이해
- [ ] ARCHITECTURE.md 읽음
- [ ] ROADMAP.md 읽음
- [ ] IMPLEMENTATION_DECISIONS.md 읽음

### 데이터셋 준비
- [ ] Fashion-MNIST 다운로드 방법 확인
- [ ] app/data/raw 폴더 준비
- [ ] 디스크 공간 확인 (최소 5GB 권장)

### Git 상태
- [ ] 로컬 변경사항 커밋 완료
- [ ] main 브랜치 업데이트 (`git pull origin main`)
- [ ] 새 브랜치 생성 준비 (`git checkout -b feature/phase2-ingest`)

---

## 🚀 Phase 2 시작 전 최종 확인

```bash
# 1. 환경 검증
python -c "
from app.common.database import init_db, get_db
from app.common.config import Config
from openai import OpenAI
print('[OK] All imports successful')
"

# 2. 서버 시작 테스트
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
# Ctrl+C로 종료

# 3. .env 파일 확인
cat .env  # OPENAI_API_KEY 확인

# 4. 데이터 폴더 확인
ls -la app/data/

# 5. Git 상태 확인
git status
git log --oneline -5
```

**모두 통과하면 Phase 2 준비 완료!** ✅

---

## 📞 문제 해결

### OpenAI API Key 오류
```
AuthenticationError: Incorrect API key provided
```
→ .env 파일의 API Key 확인 및 크레딧 확인

### FAISS 설치 오류
```
ModuleNotFoundError: No module named 'faiss'
```
→ `pip install faiss-cpu` 재실행

### 포트 8000 이미 사용 중
```
OSError: [Errno 48] Address already in use
```
→ 다른 포트 사용: `--port 8001`

### DB 연결 오류
```
sqlite3.DatabaseError: database disk image is malformed
```
→ `app/data/images.db` 삭제 후 재생성

---

## 📝 참고사항

- Phase 2는 Day 5부터 Day 10까지 약 1주일 소요 예정
- Day 5는 현재 코드 검증 (문제 없으면 빠르게 진행)
- 데이터셋 다운로드는 인터넷 속도에 따라 시간 소요 가능
- OpenAI API 비용은 테스트 범위 내에서 ~$1 이하
- 모든 준비가 완료되면 즉시 Day 5 개발 시작 가능

---

**작성일**: 2025-11-08
**상태**: Phase 2 준비 가이드 v1.0
**다음 단계**: Phase 2 시작 시 이 가이드 참고하여 준비사항 확인
