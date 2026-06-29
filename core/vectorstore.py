"""
Raw FAISS vector store — no LangChain.

Native format files:
  data/db/index.faiss   ← FAISS binary index
  data/db/docs.pkl      ← list of {text, source, section, ...} dicts

The migrate_db.py script converts the old LangChain db/ → this format.
"""
from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np


Document = Dict[str, Any]  # {"text": ..., "source": ..., "section": ..., ...}


class FAISSStore:
    """
    Thin wrapper around a faiss.Index.
    Vectors must be L2-normalised before adding (use KoreanEmbedder with
    normalize_embeddings=True).  Inner-product search then == cosine similarity.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim
        # IndexFlatIP → exact inner-product (cosine for normalised vecs)
        self.index: faiss.Index = faiss.IndexFlatIP(dim)
        self.documents: List[Document] = []

    # ── Write ───────────────────────────────────────────────────────────
    def add(self, vectors: np.ndarray, documents: List[Document]) -> None:
        """Add N vectors + their matching documents."""
        assert len(vectors) == len(documents), "vectors and docs must be same length"
        self.index.add(vectors.astype(np.float32))
        self.documents.extend(documents)

    # ── Read ────────────────────────────────────────────────────────────
    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[Document, float, int]]:
        """
        Returns list of (document, cosine_score, faiss_index_id).
        score ∈ [-1, 1]; higher is more similar.
        """
        q = query_vec.reshape(1, -1).astype(np.float32)
        D, I = self.index.search(q, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx >= 0:
                results.append((self.documents[idx], float(score), int(idx)))
        return results

    def get_vector(self, faiss_id: int) -> np.ndarray:
        """Reconstruct the stored vector at position faiss_id."""
        return self.index.reconstruct(faiss_id)

    def __len__(self) -> int:
        return self.index.ntotal

    # ── Persistence ─────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "docs.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        print(f"[FAISSStore] Saved {len(self)} docs to {path}")

    @classmethod
    def load(cls, path: str) -> "FAISSStore":
        index_path = os.path.join(path, "index.faiss")
        docs_path = os.path.join(path, "docs.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}.\n"
                "Run:  python scripts/migrate_db.py --src db/ --dst data/db/\n"
                "or:   python scripts/ingest.py --input data/raw/ --output data/db/"
            )

        store = cls.__new__(cls)
        store.index = faiss.read_index(index_path)
        store.dim = store.index.d

        with open(docs_path, "rb") as f:
            store.documents = pickle.load(f)

        print(f"[FAISSStore] Loaded {len(store)} docs from {path} (dim={store.dim})")
        return store
