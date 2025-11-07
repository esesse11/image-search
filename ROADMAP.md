# 🗺️ Development Roadmap

**프로젝트명**: Image Search System
**타임라인**: 3주 (Phase 1~3)
**예산**: ~$50-100 (OpenAI API, 프리티어 VM 활용)
**배포 타겟**: WSL2 로컬 → AWS Lightsail / Oracle Free Tier

---

## 📅 Timeline Overview

```
Week 1: Foundation (기초 구축)
├─ Day 1-2: Project Initialization ✅ (완료)
├─ Day 3-4: Core Modules (API, Ingest, Index)
└─ Day 5: Testing & Debugging

Week 2: Feature Development (기능 개발)
├─ Day 6-7: Ingest Pipeline (Captioning, Attributes)
├─ Day 8-9: Search API (FAISS Integration)
└─ Day 10: Performance Optimization

Week 3: Polish & Deployment (완성 & 배포)
├─ Day 11-12: Streamlit UI
├─ Day 13: Fine-tuning Setup (Optional)
├─ Day 14: Local Testing
└─ Day 15: Production Deployment

```

---

## Phase 1: Foundation (Days 1-5) - 진행 중 ✅

### ✅ Day 1-2: Project Initialization (완료)

**산출물:**
- [x] 폴더 구조 생성
- [x] config.yaml 작성 (모든 설정 통합)
- [x] requirements.txt (모든 의존성)
- [x] .env.example (환경 변수 템플릿)
- [x] .gitignore (Git 제외 파일)
- [x] README.md (설치/사용 가이드)
- [x] ARCHITECTURE.md (상세 아키텍처)
- [x] ROADMAP.md (이 파일)

**다음 체크:**
```bash
cd image-search
ls -la
# config.yaml, requirements.txt, .env.example, README.md ✓
```

---

### 🔨 Day 3-4: Core Modules (예정)

#### Task 1: FastAPI 기본 구조

**파일:**
- `app/api/main.py` - FastAPI 애플리케이션
- `app/api/search.py` - 검색 엔드포인트
- `app/api/assets.py` - 이미지 메타 엔드포인트
- `app/api/feedback.py` - 피드백 수집

**구현:**
```python
# app/api/main.py
from fastapi import FastAPI
from app.api import search, assets, feedback

app = FastAPI(title="Image Search API")
app.include_router(search.router)
app.include_router(assets.router)
app.include_router(feedback.router)

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**테스트:**
```bash
uvicorn app.api.main:app --reload
# 브라우저: http://localhost:8000/docs
```

#### Task 2: Database 설정

**파일:**
- `app/common/database.py` - SQLAlchemy 설정
- `app/common/models.py` - ORM 모델

**구현:**
```python
# app/common/models.py
from sqlalchemy import Column, String, JSON, DateTime
from datetime import datetime

class Image(Base):
    __tablename__ = "images"

    id = Column(String, primary_key=True)
    caption = Column(String)
    attributes = Column(JSON)  # {"color": ["blue"], ...}
    palette = Column(JSON)     # ["#4A90E2", "#FFFFFF"]
    embedding = Column(JSON)   # [0.1, -0.2, ...]
    created_at = Column(DateTime, default=datetime.utcnow)

class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    query = Column(String)
    image_id = Column(String)
    relevance = Column(Integer)  # 0 or 1
```

#### Task 3: Config Loader

**파일:**
- `app/common/config.py` - 설정 로더

**구현:**
```python
# app/common/config.py
from pydantic import BaseSettings
import yaml

class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    db_path: str = "./app/data/images.db"
    ...

    class Config:
        env_file = ".env"

def load_config(path: str = "config.yaml"):
    with open(path) as f:
        yaml_config = yaml.safe_load(f)
    return yaml_config
```

**체크리스트:**
```
- [ ] app/api/main.py 작성 및 테스트
- [ ] app/common/models.py 작성
- [ ] app/common/database.py 작성
- [ ] app/common/config.py 작성
- [ ] FastAPI /health 엔드포인트 테스트
- [ ] SQLite 테이블 생성 테스트
```

---

### 🧪 Day 5: Testing & Debugging

**테스트 항목:**
```bash
# 1. API 정상 작동
curl http://localhost:8000/health
# {"status": "ok"}

# 2. DB 연결
python -c "from app.common.database import SessionLocal; db = SessionLocal(); print('✓ DB OK')"

# 3. 설정 로드
python -c "from app.common.config import load_config; c = load_config(); print(c['api'])"

# 4. .env 파일 확인
cat .env  # OPENAI_API_KEY 설정 확인

# 5. 모델 검증
python -c "from app.common.models import ImageAttributes; a = ImageAttributes(color=['blue']); print(a)"
```

**산출물:**
- ✅ FastAPI 기본 구조 완성
- ✅ SQLite DB 초기화 + 7개 속성 모델
- ✅ 설정 시스템 통합
- ✅ Git 커밋 & 푸시

---

## Phase 2: Feature Development (Days 6-10) - 예정

### 🖼️ Day 6-7: Ingest Pipeline

#### Task 1: Image Preprocessing

**파일:** `app/ingest/preprocessor.py`

```python
class Preprocessor:
    def process_image(self, image_path: str):
        # 1. 이미지 로드 (PIL)
        img = Image.open(image_path)

        # 2. 리사이징 (512x512)
        img.thumbnail((512, 512))

        # 3. 정규화
        # (0-255) → (0-1)

        # 4. 썸네일 생성 (256x256)

        return processed_img
```

**테스트:**
```bash
python -m app.ingest.preprocessor --test
```

#### Task 2: Image Captioning

**파일:** `app/ingest/captioner.py`

```python
from openai import OpenAI

class Captioner:
    def generate(self, image_path: str):
        # OpenAI Vision API 호출
        # 입력: 이미지
        # 출력: "파란색 면 드레스, 여름 시즌"

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-vision",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "Describe this clothing item in Korean..."}
                    ]
                }
            ]
        )
        return response.choices[0].message.content
```

**예상 비용:** 100개 이미지 × $0.03/image = $3

#### Task 3: Attribute Extraction

**파일:** `app/ingest/extractor.py`

```python
class Extractor:
    def extract(self, caption: str, image_path: str):
        # 캡션에서 structured attributes 추출
        # 예: "파란색 면 드레스" → {
        #   "color": ["blue"],
        #   "material": ["cotton"],
        #   "style": ["dress"]
        # }

        # 방법 1: LLM (더 정확)
        # 방법 2: 규칙 기반 (빠름)
```

#### Task 4: Color Palette

**파일:** `app/ingest/palette.py`

```python
class PaletteExtractor:
    def extract_palette(self, image_path: str, num_colors: int = 5):
        # K-means clustering으로 주요 색상 추출
        # PIL + scikit-learn 사용

        from sklearn.cluster import KMeans
        import colorsys

        colors = KMeans(n_clusters=num_colors).fit(pixels)
        hex_colors = [self._rgb_to_hex(c) for c in colors.cluster_centers_]
        return hex_colors
```

**체크리스트:**
```
- [ ] app/ingest/preprocessor.py 작성
- [ ] app/ingest/captioner.py 작성 (OpenAI Vision)
- [ ] app/ingest/extractor.py 작성
- [ ] app/ingest/palette.py 작성
- [ ] 통합 pipeline.py 작성
- [ ] 샘플 10개 이미지로 테스트
```

---

### 🔍 Day 8-9: Search API & Indexing

#### Task 1: Embedder

**파일:** `app/index/embedder.py`

```python
from openai import OpenAI

class Embedder:
    def embed(self, text: str):
        # OpenAI text-embedding-3-small
        # 입력: "파란색 여름 드레스"
        # 출력: [0.1, -0.2, ..., 0.8] (1536 dims)

        client = OpenAI()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]):
        # 배치 처리 (더 효율적)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]
```

**예상 비용:** 1,000개 이미지 인덱싱 = ~1M tokens ≈ $0.02

#### Task 2: FAISS Index

**파일:** `app/index/faiss_index.py`

```python
import faiss
import numpy as np

class FAISSIndex:
    def __init__(self, dim=1536):
        self.index = faiss.IndexFlatIP(dim)  # Inner Product
        self.id_map = {}

    def add(self, vectors: np.ndarray, ids: List[str]):
        # 벡터 추가
        self.index.add(vectors)
        self.id_map = {i: id for i, id in enumerate(ids)}

    def search(self, query_vector: np.ndarray, k: int = 20):
        # 검색
        scores, indices = self.index.search(
            query_vector.reshape(1, -1), k
        )
        return [(self.id_map[i], s) for i, s in zip(indices[0], scores[0])]

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)
```

#### Task 3: Build Script

**파일:** `app/index/build.py`

```bash
python -m app.index.build \
  --input ./app/data/raw_images \
  --use faiss \
  --batch-size 100
```

#### Task 4: Search Endpoint

**파일:** `app/api/search.py` (업데이트)

```python
@router.post("/search/text")
async def search_text(query: SearchQuery):
    # 1. 쿼리 벡터화
    query_vector = embedder.embed(query.query)

    # 2. FAISS 검색
    scores, image_ids = index.search(query_vector, k=40)

    # 3. 메타 로드
    images = db.get_images(image_ids)

    # 4. 스코어링
    results = []
    for img in images:
        score = calculate_score(img, query.w_caption, query.w_attrs)
        results.append({
            "id": img.id,
            "score": score,
            "caption": img.caption,
            "attributes": img.attributes
        })

    # 5. 정렬 및 필터링
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {"results": results[:20]}
```

**체크리스트:**
```
- [ ] app/index/embedder.py 작성
- [ ] app/index/faiss_index.py 작성
- [ ] app/index/build.py 작성
- [ ] app/api/search.py 구현
- [ ] 검색 API 테스트
  curl -X POST http://localhost:8000/search/text \
    -H "Content-Type: application/json" \
    -d '{"query": "파란색 드레스", "top_k": 10}'
```

---

### ⚡ Day 10: Performance Optimization

**최적화 항목:**

1. **임베딩 캐싱**
   ```python
   # 이미 계산한 벡터는 캐시에서 로드
   cache = JSONCache("./app/data/embeddings_cache")
   ```

2. **배치 처리**
   ```python
   # 100개씩 묶어서 API 호출
   for batch in chunks(images, 100):
       embeddings = embedder.embed_batch(batch)
   ```

3. **인덱스 최적화**
   ```python
   # GPU 사용 (있으면)
   index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
   ```

4. **DB 인덱싱**
   ```sql
   CREATE INDEX idx_attributes ON images(attributes);
   ```

---

## Phase 3: Polish & Deployment (Days 11-15) - 예정

### 🎨 Day 11-12: Streamlit UI

**파일 구조:**
```
app/ui/
├── app.py              # 메인 페이지
├── pages/
│   ├── detail.py       # 상세보기
│   └── history.py      # 검색 히스토리
└── components/
    ├── search_bar.py
    ├── filters.py
    └── result_cards.py
```

**기본 구현:**
```python
# app/ui/app.py
import streamlit as st
from app.ui.components import SearchBar, ResultCards

st.set_page_config(page_title="Image Search", layout="wide")

# 1. 검색 입력
query = st.text_input("🔍 검색", placeholder="파란색 여름 드레스")

# 2. 필터
col1, col2, col3 = st.columns(3)
color_filter = col1.multiselect("색상", ["빨강", "파랑", "검정"])
season_filter = col2.multiselect("시즌", ["봄", "여름", "가을", "겨울"])

# 3. 검색 실행
if query:
    results = api_client.search(
        query=query,
        filters={
            "color": color_filter,
            "season": season_filter
        }
    )

    # 4. 결과 표시
    ResultCards(results).display()
```

**실행:**
```bash
streamlit run app/ui/app.py
```

### 🧑‍🏫 Day 13: Fine-tuning Setup ⭐ (필수 - 선택됨)

**목적:**
- 사내 표준 캡션 포맷 학습 (SFT)
- 검색 재정렬 선호도 학습 (DPO)

**파일:**
```
training/
├── make_jsonl.py      # DB/피드백 → JSONL 변환
├── finetune.py        # OpenAI Fine-tuning API 실행
├── evaluate.py        # 파인튜닝 모델 평가
├── sft_data.jsonl     # SFT 학습 데이터
└── dpo_data.jsonl     # DPO 학습 데이터
```

**SFT (Supervised Fine-Tuning):**
```python
# training/make_jsonl.py - SFT 데이터 생성
from app.common.database import SessionLocal
from app.common.models import Image

db = SessionLocal()
sft_data = []

for image in db.query(Image).all():
    sft_data.append({
        "messages": [
            {"role": "user", "content": image.caption},
            {"role": "assistant", "content": f"Color: {image.attributes['color']} | Material: {image.attributes['material']} | Style: {image.attributes['style']}"}
        ]
    })

# 500+ 샘플 생성 및 sft_data.jsonl로 저장
```

**DPO (Direct Preference Optimization):**
```python
# 검색 피드백으로 선호도 데이터 생성
# feedback 테이블에서 relevance=1 (좋음) vs relevance=0 (나쁨)
# → dpo_data.jsonl 생성
```

**파인튜닝 실행:**
```bash
python training/finetune.py \
  --model gpt-3.5-turbo \
  --training-file sft_data.jsonl \
  --validation-file dpo_data.jsonl \
  --epochs 3

# 비용: ~$10-20 (SFT + DPO)
```

**체크리스트:**
```
- [ ] training/make_jsonl.py 작성
- [ ] training/finetune.py 작성
- [ ] training/evaluate.py 작성
- [ ] SFT 데이터 생성 (500+ 샘플)
- [ ] DPO 데이터 생성 (100+ 쌍)
- [ ] 파인튜닝 실행
- [ ] 성능 비교 (기본 vs 파인튜닝)
```

### ✅ Day 14: Local Testing

**테스트 체크리스트:**
```
- [ ] API 모든 엔드포인트 테스트
  - POST /search/text
  - GET /asset/{id}
  - POST /feedback

- [ ] UI 모든 기능 테스트
  - 검색 입력
  - 필터 적용
  - 결과 표시

- [ ] 성능 테스트
  - 검색 응답 시간 < 100ms
  - 메모리 사용량 < 2GB

- [ ] 에러 처리
  - 없는 쿼리
  - API 에러
  - DB 에러
```

### 🚀 Day 15: Production Deployment

**배포 옵션:**

#### 옵션 1: AWS Lightsail (권장)
```bash
# 1. VM 생성 (Ubuntu 22.04, 1GB RAM, $4/월)
# 2. SSH 접속
ssh -i key.pem ubuntu@ip

# 3. 저장소 클론
git clone https://github.com/esesse11/image-search.git
cd image-search

# 4. 환경 설정
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 5. .env 설정
echo "OPENAI_API_KEY=sk-..." > .env

# 6. systemd 서비스 등록
sudo cp deployment/image-search-api.service /etc/systemd/system/
sudo systemctl enable image-search-api
sudo systemctl start image-search-api

# 7. Nginx 설정 (리버스 프록시)
sudo cp deployment/nginx.conf /etc/nginx/sites-available/
sudo systemctl restart nginx

# 8. 모니터링
curl http://localhost:8000/health
```

#### 옵션 2: Oracle Cloud Free Tier
```bash
# 비슷한 절차
# 무료 2개 OCPU, 12GB RAM VM 제공
```

#### 옵션 3: 로컬 WSL2 (개발용)
```bash
# 그냥 계속 사용
uvicorn app.api.main:app --host 0.0.0.0
streamlit run app/ui/app.py
```

**최종 산출물:**
- ✅ 모든 소스 코드
- ✅ 배포 가이드 (DEPLOYMENT.md)
- ✅ API 문서 (Swagger /docs)
- ✅ 사용자 가이드 (README.md)
- ✅ 개발자 가이드 (ARCHITECTURE.md)

---

## 📊 Success Criteria

| 항목 | 목표 | 달성 여부 |
|------|------|---------|
| 검색 응답 시간 | < 100ms | ❓ |
| 검색 정확도 | > 0.8 (MRR) | ❓ |
| API 안정성 | 99.9% 가용성 | ❓ |
| 메모리 사용 | < 2GB | ❓ |
| 배포 비용 | < $50/월 | ❓ |
| 코드 커버리지 | > 80% | ❓ |

---

## 🔄 Feedback Loop

1. **주간 리뷰 (매주 금요일)**
   - 완료된 작업 확인
   - 이슈 및 블로커 파악
   - 다음주 계획 조정

2. **사용자 피드백**
   - /feedback 엔드포인트로 검색 품질 모니터링
   - 개선안 적용

3. **성능 메트릭**
   - API 응답 시간
   - 검색 정확도 (MRR, NDCG)
   - 사용자 피드백 정확도

---

## 📝 Notes

- **불확실한 항목들은 진행하며 결정**
  - 캡셔닝: OpenAI Vision vs BLIP (비용/성능 트레이드)
  - 속성 추출: LLM vs 규칙 기반 (정확도/속도)
  - 인덱스: FAISS vs Qdrant (단순성/확장성)

- **초기 우선순위**
  1. 기본 검색 기능 (쿼리 → 결과)
  2. UI 완성
  3. 배포
  4. 성능 최적화 (나중)
  5. 파인튜닝 (선택사항)

---

**시작 일자:** TBD
**예상 완료일:** 3주 후
**마지막 업데이트:** 2025-11-08

