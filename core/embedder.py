"""
Embedder — supports both OpenAI (for existing DB) and Korean sentence-transformers.

The existing db/ was built with OpenAI text-embedding-ada-002 (dim=1536).
To query it, we MUST use the same model.

If you rebuild the index fresh with scripts/ingest.py, you can switch to
Korean sentence-transformers (free, dim=768) — just update embed_model in .env.

Auto-detection:
  dim=1536 → OpenAI embeddings  (requires OPENAI_API_KEY in .env)
  dim=768  → jhgan/ko-sroberta-multitask (free, local)
"""
from __future__ import annotations

import os
from typing import List

import numpy as np


# ── OpenAI embedder (no LangChain) ──────────────────────────────────────
class OpenAIEmbedder:
    """
    Direct OpenAI embeddings API.
    Model: text-embedding-ada-002 (dim=1536, same as existing DB)
    """
    MODEL = "text-embedding-ada-002"

    def __init__(self, api_key: str | None = None):
        import openai
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", "").strip()
        )
        print(f"[Embedder] Using OpenAI {self.MODEL} (dim=1536)")

    def embed(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            resp = self.client.embeddings.create(model=self.MODEL, input=batch)
            vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            # Normalise for cosine similarity
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / (norms + 1e-10)
            all_vecs.append(vecs)
        return np.vstack(all_vecs)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


# ── Korean sentence-transformers embedder ───────────────────────────────
class KoreanEmbedder:
    """
    Free local embedding model, best for Korean.
    Use this when building a NEW index from scratch.
    model: jhgan/ko-sroberta-multitask  (dim=768)
    """
    _instance: "KoreanEmbedder | None" = None

    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        import torch
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(f"[Embedder] Loading {model_name} on {device} …", flush=True)
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 1024
        print("[Embedder] Ready ✓", flush=True)

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return vecs.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @classmethod
    def get(cls, model_name: str = "jhgan/ko-sroberta-multitask") -> "KoreanEmbedder":
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance


# ── Factory: auto-detect based on DB dimension ──────────────────────────
def get_embedder(cfg=None):
    """
    Return the right embedder based on DB dimension.
      dim=1536 → OpenAIEmbedder  (existing DB)
      dim=768  → KoreanEmbedder  (new index)
    """
    if cfg is None:
        from .config import settings as cfg

    db_dim = cfg.embed_dim

    # Try to auto-detect from actual DB if it exists
    import os, faiss as _faiss
    idx_path = os.path.join(cfg.db_path, "index.faiss")
    if os.path.exists(idx_path):
        idx = _faiss.read_index(idx_path)
        db_dim = idx.d

    if db_dim == 1536:
        return OpenAIEmbedder()
    else:
        return KoreanEmbedder.get(cfg.embed_model)
