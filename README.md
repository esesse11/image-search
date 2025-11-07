# 🖼️ Image Search System

AI 기반 이미지 검색 플랫폼 | Text-to-Image | Semantic Search

**특징:**
- 🔍 텍스트 검색으로 유사 이미지 발견 (OpenAI Embeddings)
- 🎨 색상, 재질, 스타일 기반 필터링
- ⚡ FAISS/Qdrant 기반 빠른 벡터 검색
- 🚀 저예산 배포 (로컬 + 프리티어 클라우드)
- 🐍 Docker 불필요 (순수 Python)

---

## 📋 목차

1. [시스템 요구사항](#시스템-요구사항)
2. [빠른 시작](#빠른-시작)
3. [프로젝트 구조](#프로젝트-구조)
4. [사용법](#사용법)
5. [API 문서](#api-문서)
6. [배포](#배포)
7. [FAQ](#faq)

---

## 시스템 요구사항

- **Python**: 3.9+
- **OS**: Windows (WSL2) / Linux / macOS
- **메모리**: 최소 4GB (권장 8GB)
- **디스크**: 10GB+ (이미지 데이터)
- **인터넷**: OpenAI API 요청용

---

## 빠른 시작

### 1️⃣ 저장소 클론 및 폴더 이동

```bash
git clone https://github.com/esesse11/image-search.git
cd image-search
```

### 2️⃣ 가상 환경 생성 및 활성화

```bash
# Windows/WSL2
python -m venv .venv
.venv\Scripts\activate  # Windows CMD
source .venv/bin/activate  # WSL2/Linux/macOS

# 또는 uv 사용 (더 빠름)
uv venv .venv
source .venv/bin/activate
```

### 3️⃣ 의존성 설치

```bash
pip install -r requirements.txt

# 또는 uv 사용
uv pip install -r requirements.txt
```

### 4️⃣ 환경 변수 설정

```bash
cp .env.example .env
# .env 파일 편집 - OPENAI_API_KEY 필수!
# Windows: notepad .env
# Linux/macOS: nano .env
```

**필수 설정:**
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxx  # OpenAI API 키
EMBED_MODEL=text-embedding-3-small
INDEX_TYPE=faiss
DB_PATH=./app/data/images.db
```

### 5️⃣ 초기화 (선택사항)

```bash
# 폴더 및 DB 생성
python -m app.scripts.init_db

# 샘플 데이터 다운로드 (옵션)
python -m app.scripts.download_sample_images
```

### 6️⃣ API 서버 실행

```bash
# Terminal 1: FastAPI 서버
uvicorn app.api.main:app --reload --port 8000

# 또는
python -m app.api.main
```

**접속**: http://localhost:8000/docs (Swagger UI)

### 7️⃣ UI 실행 (다른 터미널)

```bash
# Terminal 2: Streamlit UI
streamlit run app/ui/app.py

# 또는
python -m streamlit run app/ui/app.py
```

**접속**: http://localhost:8501

---

## 프로젝트 구조

```
image-search/
├── app/
│   ├── api/                 # FastAPI 검색 API
│   │   ├── main.py         # 메인 서버
│   │   ├── search.py       # 검색 엔드포인트
│   │   ├── assets.py       # 이미지 메타 엔드포인트
│   │   └── feedback.py     # 피드백 수집
│   │
│   ├── ui/                  # Streamlit 사용자 인터페이스
│   │   ├── app.py          # 메인 UI
│   │   ├── pages/          # 멀티 페이지 (상세보기, 설정 등)
│   │   └── components/     # UI 컴포넌트
│   │
│   ├── ingest/              # 이미지 수집 및 전처리
│   │   ├── pipeline.py     # 메인 파이프라인
│   │   ├── downloader.py   # 이미지 다운로드
│   │   ├── preprocessor.py # 리사이징, 정규화
│   │   ├── captioner.py    # OpenAI Vision
│   │   └── extractor.py    # 속성 추출
│   │
│   ├── index/               # 벡터 인덱싱
│   │   ├── build.py        # 인덱싱 스크립트
│   │   ├── faiss_index.py  # FAISS 래퍼
│   │   ├── qdrant_index.py # Qdrant 래퍼
│   │   └── embedder.py     # OpenAI 임베딩
│   │
│   ├── data/                # 데이터 폴더
│   │   ├── raw_images/     # 원본 이미지
│   │   ├── processed_images/ # 전처리된 이미지
│   │   ├── thumbnails/     # 썸네일
│   │   ├── images.db       # SQLite 메타DB
│   │   ├── index.faiss     # FAISS 인덱스
│   │   └── caches/         # API 응답 캐시
│   │
│   ├── scripts/             # 유틸리티 스크립트
│   │   ├── init_db.py      # DB 초기화
│   │   ├── download_sample_images.py
│   │   ├── index_batch.py  # 배치 인덱싱
│   │   └── evaluate.py     # 검색 성능 평가
│   │
│   └── common/              # 공유 모듈
│       ├── config.py       # 설정 로더
│       ├── logger.py       # 로깅
│       ├── models.py       # Pydantic 모델
│       └── database.py     # SQLAlchemy 세션
│
├── training/                # 파인튜닝 (선택)
│   ├── make_jsonl.py       # JSONL 생성
│   ├── finetune.py         # 파인튜닝 실행
│   └── train.jsonl         # 학습 데이터
│
├── config.yaml             # 통합 설정 파일
├── requirements.txt        # Python 의존성
├── .env.example           # 환경 변수 템플릿
├── .gitignore             # Git 제외 파일
├── README.md              # 이 파일
├── ARCHITECTURE.md        # 아키텍처 상세
└── ROADMAP.md             # 개발 계획

```

---

## 사용법

### 🔍 텍스트 검색

```python
import requests

# 검색 API 호출
response = requests.post("http://localhost:8000/search/text", json={
    "query": "빨간색 겨울 코트",
    "w_caption": 0.6,      # 캡션 가중치
    "w_attrs": 0.3,        # 속성 가중치
    "top_k": 20
})

results = response.json()
# [{
#   "id": "img_001",
#   "score": 0.95,
#   "caption": "...",
#   "attributes": {"color": "red", "season": "winter"},
#   "thumbnail_url": "..."
# }, ...]
```

### 📊 이미지 인덱싱 (배치)

```bash
# 로컬 이미지 폴더 인덱싱
python -m app.index.build \
  --input ./app/data/raw_images \
  --use faiss \
  --model text-embedding-3-small

# 진행 상황 확인
# Processing: 100%|████████| 1000/1000 [3:45<00:00]
```

---

## API 문서

### POST `/search/text`

텍스트 쿼리로 유사 이미지 검색

**Request:**
```json
{
  "query": "파란색 여름 드레스",
  "w_caption": 0.6,
  "w_attrs": 0.3,
  "filters": {
    "brand": "Nike",
    "season": "summer"
  },
  "top_k": 20
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "img_001",
      "score": 0.92,
      "caption": "파란색 면 드레스",
      "attributes": {
        "color": ["blue"],
        "material": ["cotton"],
        "season": "summer"
      },
      "thumbnail_url": "/images/thumbnails/img_001.jpg"
    }
  ],
  "execution_time_ms": 45
}
```

### GET `/asset/{id}`

이미지 메타 정보 조회

**Response:**
```json
{
  "id": "img_001",
  "caption": "파란색 면 드레스",
  "attributes": {...},
  "palette": ["#4A90E2", "#FFFFFF"],
  "size": [512, 512],
  "source_url": "https://...",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### POST `/feedback`

검색 결과에 대한 피드백 (relevance 학습용)

**Request:**
```json
{
  "query": "파란색 드레스",
  "image_id": "img_001",
  "relevance": 1  # 0 또는 1
}
```

더 많은 엔드포인트는 `/docs` (Swagger UI) 참고.

---

## 배포

### 🏃 로컬 실행 (개발)

```bash
# Terminal 1: API
uvicorn app.api.main:app --reload --port 8000

# Terminal 2: UI
streamlit run app/ui/app.py --server.port 8501
```

### 🚀 VM 배포 (프로덕션)

**권장 플랫폼:**
- **AWS Lightsail** (1-2$/월 초기 3개월 무료)
- **Oracle Cloud Free Tier** (무료 2개 OCPU VM)
- **Naver Cloud Micro** (₩5,500/월)

**배포 절차 (예: Oracle VM):**

```bash
# 1. SSH 접속
ssh -i key.pem ubuntu@your-vm-ip

# 2. 저장소 클론
git clone https://github.com/esesse11/image-search.git
cd image-search

# 3. 환경 설정
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# .env 설정
nano .env  # OPENAI_API_KEY 입력

# 4. systemd 서비스 등록 (API)
sudo tee /etc/systemd/system/image-search-api.service > /dev/null <<EOF
[Unit]
Description=Image Search API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/image-search
ExecStart=/home/ubuntu/image-search/.venv/bin/uvicorn app.api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable image-search-api
sudo systemctl start image-search-api

# 5. Nginx 리버스 프록시 (선택)
# config 예시 포함 (DEPLOYMENT.md 참고)
```

---

## FAQ

### Q1. OpenAI API 비용이 얼마나 드나요?

**예상 비용 (월):**
- Embeddings (text-embedding-3-small): ~$0.02/1M tokens
- Vision (gpt-4-vision): ~$0.03/image (4K 토큰 기준)
- Fine-tuning: ~$3/hour

**예:** 1,000개 이미지 인덱싱 → ~$30-50

### Q2. 로컬에서 무료로 테스트할 수 있나요?

네! 다음 방법들이 있습니다:

1. **Mock Embeddings** (테스트 용)
   ```bash
   # .env에서
   MOCK_EMBEDDINGS=true
   ```

2. **로컬 모델** (느리지만 무료)
   ```yaml
   # config.yaml
   embedding:
     provider: "local"  # sentence-transformers
     model: "all-MiniLM-L6-v2"
   ```

3. **OpenAI Free Trial** (초기 $5 크레딧)

### Q3. FAISS vs Qdrant 뭐가 낫나요?

| 기준 | FAISS | Qdrant |
|------|-------|--------|
| 설정 | 간단 | 복잡 |
| 속도 | 빠름 | 빠름 |
| 메모리 | 적음 | 많음 |
| 배포 | 파일 기반 | 서버 필요 |
| 추천 | ✅ 초기/소규모 | 대규모 |

**초기 구현:** FAISS 권장

### Q4. 이미지는 몇 개까지 지원하나요?

- **FAISS**: CPU로 1M+ 개 가능 (메모리 의존)
- **Qdrant**: 거의 무제한 (디스크 크기에 따라)

**메모리 계산:**
- 1,000개 이미지 = ~6MB (FAISS)
- 10,000개 = ~60MB
- 100,000개 = ~600MB

### Q5. Windows에서 실행 가능한가요?

네! WSL2 또는 native Python 모두 지원.

```bash
# Native Windows (cmd.exe)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload
```

---

## 🤝 기여

버그 리포트, 기능 제안은 GitHub Issues로!

---

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

---

## 📞 연락처

- **Issues**: https://github.com/esesse11/image-search/issues
- **Email**: esesse11@naver.com

---

## 🗺️ 다음 단계

1. 로컬에서 `README.md` 단계 따라 설치 & 실행
2. `ARCHITECTURE.md` 읽고 전체 아키텍처 이해
3. `ROADMAP.md`에서 개발 계획 확인
4. 첫 번째 이미지 인덱싱 및 검색 테스트

**Happy searching!** 🚀

