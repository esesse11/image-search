# 🎯 Implementation Decisions

**작성일**: 2025-11-08
**상태**: 확정 (Phase 1 Day 3 시작)

---

## 선택된 구현 방향

### 1️⃣ 이미지 데이터 소스: **공개 데이터셋** ✅

**선택**: A - 공개 데이터셋 자동 다운로드

**구현 계획:**
- **Fashion-MNIST** 또는 **Polyvore Outfit Composition Dataset** 사용
- `app/scripts/download_sample_images.py` 작성
- 초기 개발용: 100-500개 샘플
- 배포 전: 전체 데이터셋으로 확장 가능

**장점:**
- 재현 가능 (누구나 같은 데이터로 테스트)
- 라이선스 명확 (오픈 데이터)
- 자동화 가능

**예시:**
```bash
python app/scripts/download_sample_images.py --dataset polyvore --count 200
```

---

### 2️⃣ 캡셔닝 방식: **테스트 하드코딩** ✅

**선택**: A - 초기는 하드코딩, 나중에 OpenAI Vision으로 전환

**구현 계획:**

#### Phase 1 (초기, 현재):
```python
# app/ingest/captioner.py - 테스트 모드
def generate_caption(image_id: str):
    # 하드코딩된 캡션 (테스트용)
    captions = {
        "img_001": "Blue cotton summer dress",
        "img_002": "Red leather winter jacket",
        ...
    }
    return captions.get(image_id, "Unknown item")
```

**비용**: $0
**속도**: 즉시
**정확도**: 낮음 (테스트 목적)

#### Phase 3 (배포 전):
```python
# 같은 인터페이스, OpenAI Vision으로 전환
from openai import OpenAI

def generate_caption(image_path: str):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4-vision",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Describe this clothing..."}
            ]
        }]
    )
    return response.choices[0].message.content
```

**비용**: $0.03/image
**속도**: ~10초/image
**정확도**: 높음

**전환 방법:**
- `config.yaml`의 `captioning.provider` 변경
- `.env`에 `OPENAI_API_KEY` 추가
- 구현은 동일 (인터페이스 통일)

---

### 3️⃣ 벡터 인덱스: **FAISS 유지** ✅

**선택**: A - FAISS (변경 없음)

**이유:**
- 설정 간단 (파일 기반)
- CPU 기반 (GPU 불필요)
- 초기/중규모 데이터 최적 (1M 레코드)
- 배포 용이

**확장 계획 (향후):**
- 100M+ 레코드 필요시 → Qdrant 마이그레이션
- 현재 FAISS API로 인터페이스 통일하면 쉬운 전환 가능

```python
# app/index/base.py - 추상 인터페이스
class IndexBase:
    def add(self, vectors, ids): pass
    def search(self, query, k): pass
    def save(self, path): pass
    def load(self, path): pass

# FAISS, Qdrant 모두 이 인터페이스 구현
```

---

### 4️⃣ 검색 필터/속성: **7개 확대** ✅

**선택**: B - 기본 4개 + 추가 3개

**최종 속성:**

| # | 속성 | 타입 | 값 예시 |
|---|------|------|--------|
| 1 | **color** | List[str] | ["blue", "red"] |
| 2 | **material** | List[str] | ["cotton", "leather"] |
| 3 | **style** | List[str] | ["casual", "formal"] |
| 4 | **season** | List[str] | ["summer", "winter"] |
| 5 | **brand** | str | "Nike" |
| 6 | **size** | str | "M" |
| 7 | **pattern** | List[str] | ["striped", "solid"] |
| 8 | **price_range** | str | "$50-100" |

**데이터 모델:**
```python
# app/common/models.py
class ImageAttributes(BaseModel):
    color: List[str] = []
    material: List[str] = []
    style: List[str] = []
    season: List[str] = []
    brand: Optional[str] = None
    size: Optional[str] = None
    pattern: List[str] = []
    price_range: Optional[str] = None
```

**DB 스키마:**
```python
# app/common/models.py (SQLAlchemy)
class Image(Base):
    ...
    attributes = Column(JSON)  # 위의 구조를 JSON으로 저장
```

**검색 API 필터:**
```python
# POST /search/text
{
    "query": "blue summer dress",
    "filters": {
        "color": ["blue"],
        "season": ["summer"],
        "brand": "Nike",
        "price_range": "$0-50"
    },
    "top_k": 20
}
```

---

### 5️⃣ 파인튜닝: **초기부터 포함** ✅

**선택**: B - 처음부터 파인튜닝 지원

**목적:**
- 사내 표준 캡션 포맷 학습
- 검색 재정렬 선호도 학습 (DPO)

**구현 계획:**

#### Phase 2에 추가:
```
training/
├── make_jsonl.py       # DB → JSONL 변환
├── sft_data.jsonl      # Supervised Fine-Tuning 데이터
├── dpo_data.jsonl      # Direct Preference Optimization 데이터
├── finetune.py         # 파인튜닝 실행
└── evaluate.py         # 성능 평가
```

#### SFT (Supervised Fine-Tuning) 예시:
```json
// training/sft_data.jsonl
{
  "messages": [
    {"role": "user", "content": "Blue cotton summer dress from Nike"},
    {"role": "assistant", "content": "Color: Blue | Material: Cotton | Season: Summer | Brand: Nike"}
  ]
}
```

#### DPO (Direct Preference Optimization) 예시:
```json
// training/dpo_data.jsonl
{
  "prompt": "Search: blue summer dress",
  "chosen": "img_001 (score: 0.95) - Perfect match",
  "rejected": "img_002 (score: 0.60) - Wrong color"
}
```

#### 실행:
```bash
# JSONL 생성
python training/make_jsonl.py --db ./app/data/images.db

# 파인튜닝 (OpenAI API)
python training/finetune.py --model gpt-3.5-turbo --data training/sft_data.jsonl

# 평가
python training/evaluate.py --model ft:gpt-3.5-turbo:...
```

**비용 예상:**
- SFT: 1,000개 샘플 × $0.08/1K tokens = ~$8
- DPO: 500개 쌍 × $0.003/pair = ~$1.50
- 총: ~$10 (선택사항)

**타이밍:**
- Phase 2 중반: 기초 검색 완성 후
- 피드백 데이터 수집 시작
- Phase 3: 파인튜닝 실행

---

### 6️⃣ API 인증: **초기 없음** ✅

**선택**: A - 초기에는 없음 (배포 전 추가)

**구현 계획:**

#### Phase 1-3 (개발):
```python
# app/api/main.py
@app.get("/search/text")
async def search_text(query: SearchQuery):
    # 인증 없음, 누구나 접근 가능
    return search_results
```

#### Phase 4 (배포 전):
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.get("/search/text")
async def search_text(query: SearchQuery, credentials = Depends(security)):
    # API Key 검증
    if credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return search_results
```

**API Key 관리:**
```python
# .env
API_KEYS=key1,key2,key3

# config.yaml
api:
  require_auth: false  # Phase 1-3: false, Phase 4: true
  valid_keys: ${API_KEYS}
```

---

## 📊 구현 일정 영향도

### Phase 1-2 (Day 1-10): 영향 없음 ✅
- 속성 7개 구조만 다름 (기술적 변경 없음)
- 파인튜닝은 별도 모듈 (선택적)

### Phase 3 (Day 11-15): 영향 있음
- 파인튜닝 섹션 추가 (Day 13)
- 데이터 수집/준비 필요
- JSONL 생성 스크립트 추가

### 예상 추가 작업:
- `training/make_jsonl.py`: 100줄
- `training/finetune.py`: 150줄
- `training/evaluate.py`: 100줄
- **총 추가 시간**: Phase 3에 +1-2일

---

## 🔄 마이그레이션 경로

### 캡셔닝:
```
Phase 1-2: 하드코딩
    ↓
Phase 3: config.yaml에서 provider 변경
    ↓
openai 선택 시 자동으로 OpenAI Vision 사용
```

### 인덱싱:
```
Phase 1-3: FAISS
    ↓
필요시: config.yaml에서 index.type 변경
    ↓
qdrant 선택 시 Qdrant 사용
```

### 인증:
```
Phase 1-3: 없음
    ↓
배포 전: config.yaml에서 api.require_auth = true
    ↓
자동으로 HTTPBearer 보안 활성화
```

---

## 📝 구현 체크리스트

### Day 3-4: FastAPI 기본 구조
- [ ] `app/api/main.py` - API 앱 초기화
- [ ] `app/api/search.py` - 검색 엔드포인트 (stub)
- [ ] `app/api/assets.py` - 이미지 메타 (stub)
- [ ] `app/api/feedback.py` - 피드백 (stub)
- [ ] `app/common/database.py` - SQLAlchemy + 7개 속성 모델
- [ ] `app/common/config.py` - 설정 로더

### Day 5: 테스트
- [ ] FastAPI /health 테스트
- [ ] DB 연결 테스트
- [ ] 속성 모델 검증
- [ ] Git 커밋

### Day 6-7: Ingest
- [ ] `app/scripts/download_sample_images.py` 작성
- [ ] 공개 데이터셋 다운로드 테스트
- [ ] `app/ingest/preprocessor.py` 작성
- [ ] `app/ingest/captioner.py` (하드코딩 모드)

### Day 8-9: Search & Index
- [ ] `app/index/embedder.py` (OpenAI)
- [ ] `app/index/faiss_index.py`
- [ ] `app/index/build.py` 배치 스크립트
- [ ] `/search/text` 엔드포인트 구현

### Day 13: Fine-tuning (Phase 3)
- [ ] `training/make_jsonl.py`
- [ ] `training/finetune.py`
- [ ] `training/evaluate.py`

---

## 🎓 참고 문서

- **ARCHITECTURE.md**: 전체 시스템 다이어그램
- **ROADMAP.md**: 3주 일정 (Day 1-15)
- **README.md**: 사용자 가이드
- **config.yaml**: 모든 설정 통합

---

**다음 단계**: Day 3-4 FastAPI 구현 시작! 🚀

