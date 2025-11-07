"""Config Module - Load and manage configuration"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """설정 관리 클래스"""

    _config: Dict[str, Any] = {}

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> Dict[str, Any]:
        """config.yaml 파일 로드"""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            cls._config = yaml.safe_load(f)

        return cls._config

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """설정 값 조회 (dot notation 지원)

        예: Config.get("api.port") → 8000
        """
        if not cls._config:
            cls.load()

        keys = key.split(".")
        value = cls._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default


def get_config() -> Dict[str, Any]:
    """FastAPI에서 사용할 config 반환"""
    return Config.load()


# 초기 로드
try:
    Config.load()
except FileNotFoundError:
    print("[WARN] config.yaml not found. Using defaults.")
    Config._config = {}
