"""Database Module - SQLAlchemy ORM Setup"""

import os
from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from app.common.models import ImageAttributes

# DB URL
DB_PATH = os.getenv("DB_PATH", "./app/data/images.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# SQLAlchemy Engine & Session
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite 멀티스레딩
    echo=False  # SQL 로깅 비활성화
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ORM Models
class Image(Base):
    """이미지 메타데이터 테이블"""

    __tablename__ = "images"

    id = Column(String, primary_key=True, index=True)
    caption = Column(String)  # 이미지 설명
    attributes = Column(JSON)  # ImageAttributes (7개 필드)
    palette = Column(JSON)  # 주요 색상 리스트 ["#FF0000", "#00FF00"]
    embedding = Column(JSON)  # 벡터 [0.1, -0.2, ..., 0.8]
    source_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Image id={self.id} caption={self.caption[:30]}...>"


class Feedback(Base):
    """검색 피드백 테이블"""

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String)  # 검색 쿼리
    image_id = Column(String, index=True)  # 관련 이미지 ID
    relevance = Column(Integer)  # 0 (비관련) 또는 1 (관련)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Feedback query={self.query} relevance={self.relevance}>"


# DB 초기화
def init_db():
    """테이블 생성"""
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created")


# Dependency for FastAPI
def get_db() -> Session:
    """FastAPI에서 DB 세션 의존성으로 사용"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# DB 유틸리티
class DBHelper:
    """DB 작업 헬퍼"""

    @staticmethod
    def add_image(db: Session, image: Image) -> Image:
        """이미지 추가"""
        db.add(image)
        db.commit()
        db.refresh(image)
        return image

    @staticmethod
    def get_image(db: Session, image_id: str) -> Image:
        """이미지 조회"""
        return db.query(Image).filter(Image.id == image_id).first()

    @staticmethod
    def get_images(db: Session, limit: int = 100, offset: int = 0) -> list:
        """이미지 목록 조회"""
        return db.query(Image).limit(limit).offset(offset).all()

    @staticmethod
    def add_feedback(db: Session, feedback: Feedback) -> Feedback:
        """피드백 추가"""
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        return feedback

    @staticmethod
    def get_feedbacks(db: Session, query: str = None) -> list:
        """피드백 조회"""
        query_obj = db.query(Feedback)
        if query:
            query_obj = query_obj.filter(Feedback.query == query)
        return query_obj.all()

    @staticmethod
    def delete_image(db: Session, image_id: str) -> bool:
        """이미지 삭제"""
        image = db.query(Image).filter(Image.id == image_id).first()
        if image:
            db.delete(image)
            db.commit()
            return True
        return False

    @staticmethod
    def count_images(db: Session) -> int:
        """이미지 개수"""
        return db.query(Image).count()

    @staticmethod
    def count_feedbacks(db: Session) -> int:
        """피드백 개수"""
        return db.query(Feedback).count()


# 초기화 (앱 시작 시)
if not os.path.exists(DB_PATH):
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    init_db()
