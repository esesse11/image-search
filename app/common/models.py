"""Pydantic Models - Data Structures"""

from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime


# Attribute Models (7개 속성)
class ImageAttributes(BaseModel):
    """이미지 속성 (7개 필드)"""
    # 다중값 속성 (List)
    color: List[str] = []  # ["blue", "red"]
    material: List[str] = []  # ["cotton", "leather"]
    style: List[str] = []  # ["casual", "formal"]
    season: List[str] = []  # ["summer", "winter"]
    pattern: List[str] = []  # ["striped", "solid"]

    # 단일값 속성
    brand: Optional[str] = None  # "Nike"
    size: Optional[str] = None  # "M", "L"
    price_range: Optional[str] = None  # "$0-50", "$50-100"


# Request Models
class SearchQuery(BaseModel):
    """텍스트 검색 요청"""
    query: str
    w_caption: float = 0.6  # 캡션 가중치
    w_attrs: float = 0.3  # 속성 가중치
    filters: Dict[str, Any] = {}  # 필터 (color, brand, season 등)
    top_k: int = 20  # 상위 k개 반환


class FeedbackData(BaseModel):
    """검색 피드백 데이터"""
    query: str
    image_id: str
    relevance: int  # 0 (비관련) or 1 (관련)


# Response Models
class ImageResult(BaseModel):
    """검색 결과 이미지"""
    id: str
    score: float  # 유사도 점수 (0-1)
    caption: str  # 이미지 설명
    attributes: ImageAttributes = ImageAttributes()
    thumbnail_url: Optional[str] = None
    palette: List[str] = []  # 주요 색상 (hex)


class SearchResponse(BaseModel):
    """검색 응답"""
    results: List[ImageResult]
    execution_time_ms: float
    total_results: int = 0


class AssetDetail(BaseModel):
    """이미지 상세 정보"""
    id: str
    caption: str
    attributes: ImageAttributes
    palette: List[str]
    size: tuple  # (height, width)
    source_url: Optional[str]
    created_at: datetime


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str  # "ok"
    version: str = "0.1.0"
    timestamp: datetime = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "version": "0.1.0"
            }
        }


# Fine-tuning Models (파인튜닝 데이터)
class SFTTrainingData(BaseModel):
    """Supervised Fine-Tuning 데이터"""
    messages: List[Dict[str, str]]  # [{"role": "user", "content": "..."}, ...]


class DPOTrainingData(BaseModel):
    """Direct Preference Optimization 데이터"""
    prompt: str  # 검색 쿼리
    chosen: str  # 선호하는 결과
    rejected: str  # 선호하지 않는 결과
