"""Pydantic Models - Data Structures"""

from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime


# Request Models
class SearchQuery(BaseModel):
    """텍스트 검색 요청"""
    query: str
    w_caption: float = 0.6
    w_attrs: float = 0.3
    filters: Dict[str, any] = {}
    top_k: int = 20


class FeedbackData(BaseModel):
    """검색 피드백 데이터"""
    query: str
    image_id: str
    relevance: int  # 0 or 1


# Response Models
class ImageResult(BaseModel):
    """검색 결과 이미지"""
    id: str
    score: float
    caption: str
    attributes: Dict[str, any] = {}
    thumbnail_url: Optional[str] = None
    palette: List[str] = []


class SearchResponse(BaseModel):
    """검색 응답"""
    results: List[ImageResult]
    execution_time_ms: float


class AssetDetail(BaseModel):
    """이미지 상세 정보"""
    id: str
    caption: str
    attributes: Dict[str, any]
    palette: List[str]
    size: tuple
    source_url: Optional[str]
    created_at: datetime


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    version: str = "0.1.0"
    timestamp: datetime = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "version": "0.1.0"
            }
        }
