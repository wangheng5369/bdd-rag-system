"""
配置文件加载器
"""

import os
from pathlib import Path
from typing import Optional
import yaml


def load_config(config_path: Optional[str] = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_api_key(config: dict) -> str:
    """
    获取 API Key

    优先级: config > 环境变量 OPENAI_API_KEY > 环境变量 ANTHROPIC_API_KEY
    """
    return config.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")


def get_model(config: dict) -> str:
    """获取模型名称"""
    return config.get("model", "MiniMax-M3")


def get_llm_config(config: dict) -> dict:
    """
    获取 LLM 配置

    Returns:
        dict: {
            "provider": "openai" | "anthropic",
            "base_url": str,
            "api_key": str,
            "model": str,
            "max_tokens": int,
        }
    """
    provider = config.get("llm_provider", "openai")
    base_url = config.get("base_url", "") or os.environ.get("LLM_BASE_URL", "")
    api_key = config.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")

    # 兼容旧环境变量
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    model = config.get("model", "MiniMax-M3")
    max_tokens = config.get("max_tokens", 4096)

    return {
        "provider": provider,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "max_tokens": max_tokens,
    }
