"""Central configuration. Every value can be overridden with an environment variable or .env."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # --- LLM (generation) -------------------------------------------------
    llm_provider: str = "ollama"            # ollama | openai | anthropic
    llm_model: str = "qwen2.5:7b"
    llm_base_url: str = "http://127.0.0.1:11434"   # 127.0.0.1 not localhost: avoids ~2s IPv6 fallback on Windows
    llm_api_key: str = ""
    llm_timeout: float = 180.0
    llm_max_retries: int = 2

    # --- Embeddings -------------------------------------------------------
    embed_provider: str = "ollama"          # ollama | openai
    embed_model: str = "embeddinggemma"
    embed_base_url: str = "http://127.0.0.1:11434"
    embed_api_key: str = ""
    embed_query_prefix: str = "task: search result | query: "
    embed_doc_prefix: str = "title: none | text: "
    embed_on_cpu: bool = True               # Ollama only: keep the small embedding model on CPU so it
                                            # does not evict the LLM from a small GPU (4 GB VRAM)

    # --- Chunking ---------------------------------------------------------
    chunk_size: int = 900                   # characters (~200-230 tokens)
    chunk_overlap: int = 150                # characters carried into the next chunk

    # --- Retrieval --------------------------------------------------------
    retrieval_mode: str = "hybrid"          # dense | bm25 | hybrid
    top_k: int = 5
    candidate_k: int = 20                   # per-retriever candidates before fusion
    bm25_weight: float = 0.3                # BM25 weight in reciprocal rank fusion (swept 1.0/0.5/0.3, see EVALUATION.md)
    min_dense_score: float = 0.30           # calibrated: answerable >= 0.40, off-topic <= 0.24 (see EVALUATION.md)
    relative_score_cutoff: float = 0.75     # keep chunks scoring >= 75% of the best dense score
    near_duplicate_threshold: float = 0.95  # cosine above which two chunks are the same content

    # --- Conversation -----------------------------------------------------
    history_turns: int = 4

    # --- API / storage ----------------------------------------------------
    max_upload_mb: int = 20
    max_files_per_upload: int = 20
    data_dir: Path = ROOT / "data"


@lru_cache
def get_settings() -> Settings:
    return Settings()
