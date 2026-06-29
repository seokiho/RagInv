"""
MMR (Maximal Marginal Relevance) retriever — no LangChain.

Ported from Prj_RAG/MMR_Opt.py, made cleaner and LangChain-free.

MMR score:
    score_i = λ · relevance(q, d_i) − (1−λ) · max_{j∈S} similarity(d_i, d_j)

λ=1 → pure relevance ranking (like top-k)
λ=0 → pure diversity (spread out results)
λ=0.7 → default: mostly relevant, somewhat diverse
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any

import numpy as np

from .vectorstore import FAISSStore

Document = Dict[str, Any]
RetrievedDoc = Tuple[Document, float]   # (doc, cosine_score)


# ── Pure MMR algorithm ──────────────────────────────────────────────────
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def mmr_select(
    query_vec: np.ndarray,
    candidate_vecs: List[np.ndarray],
    k: int,
    lambda_val: float = 0.7,
) -> List[int]:
    """
    Return indices of k documents selected by MMR.
    Works on pre-normalised or un-normalised vectors.
    """
    if not candidate_vecs:
        return []

    k = min(k, len(candidate_vecs))
    relevances = [_cosine(query_vec, v) for v in candidate_vecs]
    remaining = list(range(len(candidate_vecs)))
    selected: List[int] = []

    while len(selected) < k and remaining:
        if not selected:
            # First pick: highest relevance
            best = max(remaining, key=lambda i: relevances[i])
        else:
            best_score = -float("inf")
            best = remaining[0]
            for i in remaining:
                max_sim = max(_cosine(candidate_vecs[i], candidate_vecs[j]) for j in selected)
                score = lambda_val * relevances[i] - (1 - lambda_val) * max_sim
                if score > best_score:
                    best_score = score
                    best = i
        selected.append(best)
        remaining.remove(best)

    return selected


# ── Retriever class ─────────────────────────────────────────────────────
class Retriever:
    """
    Wraps FAISSStore + MMR.

    Usage:
        retriever = Retriever(store, top_k=5, fetch_k=20, mmr_lambda=0.7)
        results = retriever.retrieve(query_vec)   # → [(doc, score), ...]
    """

    def __init__(
        self,
        store: FAISSStore,
        top_k: int = 5,
        fetch_k: int = 20,
        mmr_lambda: float = 0.7,
    ):
        self.store = store
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.mmr_lambda = mmr_lambda

    def retrieve(self, query_vec: np.ndarray) -> List[RetrievedDoc]:
        """
        1. Fetch fetch_k candidates from FAISS (cosine similarity).
        2. Rerank top_k of them with MMR.
        Returns list of (document, cosine_score), length ≤ top_k.
        """
        candidates = self.store.search(query_vec, self.fetch_k)
        if not candidates:
            return []

        docs   = [d for d, _, _   in candidates]
        scores = [s for _, s, _   in candidates]
        f_ids  = [i for _, _, i   in candidates]
        vecs   = [self.store.get_vector(i) for i in f_ids]

        selected_idx = mmr_select(query_vec, vecs, self.top_k, self.mmr_lambda)

        return [(docs[i], scores[i]) for i in selected_idx]

    # ── Research helpers ───────────────────────────────────────────────
    def retrieve_with_meta(self, query_vec: np.ndarray) -> List[Dict]:
        """
        Like retrieve(), but returns dicts with extra metrics for paper logging.
        """
        candidates = self.store.search(query_vec, self.fetch_k)
        if not candidates:
            return []

        docs   = [d for d, _, _ in candidates]
        scores = [s for _, s, _ in candidates]
        f_ids  = [i for _, _, i in candidates]
        vecs   = [self.store.get_vector(i) for i in f_ids]

        selected_idx = mmr_select(query_vec, vecs, self.top_k, self.mmr_lambda)

        # Relevance of selected docs against all candidates (for diversity metric)
        results = []
        for rank, i in enumerate(selected_idx):
            results.append({
                "doc": docs[i],
                "cosine_score": scores[i],
                "faiss_id": f_ids[i],
                "mmr_rank": rank,
                # Avg pairwise similarity with already-selected docs (diversity measure)
                "intra_sim": float(
                    np.mean([_cosine(vecs[i], vecs[j]) for j in selected_idx if j != i])
                ) if len(selected_idx) > 1 else 0.0,
            })

        return results
