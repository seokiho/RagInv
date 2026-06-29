"""
Settings loaded from .env
All RAG parameters are here — change them without touching the code.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── LLM (Ollama) ────────────────────────────────────────
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "exaone3.5:7.8b"
    # ollama_model: str = "llama3.1:8b"

    # ── Embeddings ──────────────────────────────────────────
    # Best free Korean embedding model for legal/investigative text
    embed_model: str = "jhgan/ko-sroberta-multitask"
    embed_dim: int = 768   # 1536=OpenAI ada-002(구DB), 768=ko-sroberta(현재)

    # ── Vector DB (FAISS) ───────────────────────────────────
    db_path: str = "data/db"
    law_db_path: str = "data/law_db.json"

    # ── Retriever ───────────────────────────────────────────
    top_k: int = 5          # final docs returned to LLM (legacy research mode)
    fetch_k: int = 20       # candidates before MMR reranking
    mmr_lambda: float = 0.7 # 0=pure diversity, 1=pure relevance
    min_score: float = 0.62 # cosine score threshold — below this → "not found"
                            # 정상 법률 질문 ≥ 0.65, 쓰레기/비도메인 ≤ 0.62 (실험 기반)

    # ── 법령 우선 검색 ──────────────────────────────────────────
    top_law_k: int = 4      # 프롬프트에 넣을 법령 조문 수
    top_case_k: int = 2     # 프롬프트에 넣을 판례 수
    law_fetch_k: int = 60   # 법령 확보를 위한 FAISS 초기 후보 수

    # ── Chunker ─────────────────────────────────────────────
    chunk_size: int = 500
    chunk_overlap: int = 100
    # "section" | "fixed" | "hierarchical"
    chunk_strategy: str = "section"

    # ── Research / paper mode ───────────────────────────────
    research_log_dir: str = "results"


settings = Settings()
