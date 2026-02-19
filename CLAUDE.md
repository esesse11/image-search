# image-search

## 프로젝트 개요
FastAPI 기반 이미지 검색 시스템. 이미지 임베딩(OpenAI) + FAISS 벡터 검색 + Streamlit UI.

## 현재 상태 (2026-02-19 기준)

### 완성된 것 (Phase 1 Day 1-4)
- 프로젝트 구조 및 설정 (config.yaml, .env.example, requirements.txt)
- FastAPI scaffold: app/api, common, data, index, ingest, ui, scripts
- ARCHITECTURE.md, ROADMAP.md, IMPLEMENTATION_DECISIONS.md
- PHASE2_PREPARATION.md (다음 단계 가이드)

### 아직 안 된 것 (Phase 2~3)
- OpenAI API 키 설정 (.env 미생성)
- 데이터셋 다운로드 및 이미지 전처리 (Day 6-7)
- FAISS 인덱싱 및 검색 API (Day 8-9)
- Streamlit UI (Day 11-12)
- 배포 (Day 14-15)

## 폴더 구조
```
app/
├── api/      # FastAPI 라우터
├── common/   # 공유 유틸리티
├── data/     # 데이터 모델
├── index/    # FAISS 인덱싱
├── ingest/   # 이미지 수집/전처리
├── scripts/  # CLI 스크립트
└── ui/       # Streamlit UI
training/     # 파인튜닝 (옵션)
```

## 실행 방법
```bash
cd C:\work\image-search
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.api.main:app --reload   # FastAPI 서버
streamlit run app/ui/app.py         # UI (Phase 2 이후)
```

## Phase 2 시작 전 필수 준비
1. OpenAI API 키 발급 → `.env` 파일에 `OPENAI_API_KEY=sk-proj-...` 저장
2. `app/data/` 하위 폴더 생성 (raw/, processed/)
3. Day 5: 현재 FastAPI 구조 테스트 확인

## 기술적 결정사항
- 임베딩: OpenAI text-embedding-3-small ($0.00002/1K tokens)
- 벡터DB: FAISS (로컬, 무료)
- DB: SQLite (images.db)
- 배포 타겟: WSL2 로컬 → AWS Lightsail / Oracle Free Tier

## 알려진 이슈
- `.venv/` 내 Windows 인코딩 이슈 일부 수정됨 (print문 관련)
- `app/data/` 폴더 미생성 상태 → Phase 2 시작 전 생성 필요

## 다음 세션 작업 목록 (우선순위 순)
1. `.env` 파일 생성 (OpenAI API 키 설정)
2. Day 5: FastAPI 구조 테스트 (`uvicorn` 실행 후 `/docs` 확인)
3. Day 6-7: 이미지 데이터셋 다운로드 및 전처리 파이프라인 구현
4. Day 8-9: FAISS 인덱싱 및 검색 API 구현
