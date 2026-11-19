"""Configuration (environment variables or .env override every value)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # LLM backend (provider-agnostic, see app/llm)
    llm_provider: str = "ollama"            # ollama | openai | anthropic
    llm_model: str = "qwen2.5:7b"
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_api_key: str = ""
    llm_timeout: float = 240.0
    llm_max_retries: int = 1

    # Upload limits
    max_upload_mb: float = 10.0
    max_pages: int = 20
    max_chars_for_llm: int = 24000          # longer documents are truncated with a warning

    # OCR
    ocr_enabled: bool = True
    ocr_min_chars_per_page: int = 30        # pages with less extractable text are OCR'd
    ocr_dpi: int = 200

    # Validation tolerances
    amount_tolerance_abs: float = 0.02
    amount_tolerance_rel: float = 0.005

    log_dir: Path = ROOT / "logs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
