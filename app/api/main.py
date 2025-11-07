"""FastAPI Main Application"""

import os
import sys
from datetime import datetime
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# 상위 디렉토리를 Python path에 추가 (import 문제 해결)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.common.database import get_db, init_db, DBHelper
from app.common.models import HealthResponse

# FastAPI 앱 초기화
app = FastAPI(
    title="Image Search API",
    description="AI-powered image search system",
    version="0.1.0"
)

# CORS 설정 (개발 단계)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용 (개발용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB 초기화
@app.on_event("startup")
async def startup_event():
    """앱 시작 시 DB 초기화"""
    init_db()
    print("✓ Application started")


@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료 시"""
    print("✓ Application shutdown")


# 기본 엔드포인트
@app.get("/", tags=["Root"])
async def root():
    """루트 경로"""
    return {
        "name": "Image Search API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(db: Session = Depends(get_db)):
    """헬스 체크 엔드포인트"""
    try:
        # DB 연결 테스트
        image_count = DBHelper.count_images(db)
        feedback_count = DBHelper.count_feedbacks(db)

        return HealthResponse(
            status="ok",
            version="0.1.0",
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "version": "0.1.0"
        }


# TODO: 검색 엔드포인트 (Day 8-9)
# TODO: 이미지 메타 엔드포인트 (Day 6-7)
# TODO: 피드백 엔드포인트 (Day 8)


# 라우터 임포트 (나중에 추가)
# from app.api import search, assets, feedback
# app.include_router(search.router)
# app.include_router(assets.router)
# app.include_router(feedback.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
