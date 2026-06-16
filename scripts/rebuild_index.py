"""
docs.pkl 의 텍스트로 FAISS 인덱스를 한국어 임베더로 재빌드.

Usage:
    python scripts/rebuild_index.py
    python scripts/rebuild_index.py --input data/db/docs_v2.pkl
    python scripts/rebuild_index.py --db data/db --batch 128
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import settings
from core.embedder import KoreanEmbedder
from core.vectorstore import FAISSStore


def rebuild(db_path: str, batch_size: int, input_pkl: str | None = None) -> None:
    # --input 지정 시 그 파일 사용, 아니면 db_path/docs_v2.pkl 또는 docs.pkl 순으로 시도
    if input_pkl:
        docs_path = input_pkl
    else:
        v2 = os.path.join(db_path, "docs_v2.pkl")
        docs_path = v2 if os.path.exists(v2) else os.path.join(db_path, "docs.pkl")

    index_path = os.path.join(db_path, "index.faiss")
    backup_path = os.path.join(db_path, "index.faiss.bak")

    # ── 1. Load docs ─────────────────────────────────────────────
    print(f"[rebuild] 로드 중: {docs_path} …")
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)

    law_cnt  = sum(1 for d in docs if d.get("doc_type") == "law")
    case_cnt = sum(1 for d in docs if d.get("doc_type") == "case")
    print(f"[rebuild] 총 {len(docs):,}개  (법령 {law_cnt:,}  판례 {case_cnt:,})")

    # ── 2. Embed ──────────────────────────────────────────────────
    embedder = KoreanEmbedder.get(settings.embed_model)
    store = FAISSStore(dim=768)

    all_vecs = []
    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding"):
        batch = docs[i: i + batch_size]
        texts = [d.get("text") or d.get("page_content", "") for d in batch]
        vecs = embedder.embed(texts)
        all_vecs.append(vecs)

    all_vecs_np = np.vstack(all_vecs)
    store.add(all_vecs_np, docs)

    # ── 3. Backup old index & save new ───────────────────────────
    if os.path.exists(index_path):
        os.rename(index_path, backup_path)
        print(f"[rebuild] 기존 인덱스 백업 → {backup_path}")

    store.save(db_path)

    # docs_v2.pkl 을 사용했다면 docs.pkl 도 교체
    if docs_path != os.path.join(db_path, "docs.pkl"):
        import shutil
        dst_docs = os.path.join(db_path, "docs.pkl")
        shutil.copy2(docs_path, dst_docs)
        print(f"[rebuild] {os.path.basename(docs_path)} → docs.pkl 로 복사")

    print(f"[rebuild] 완료! 새 인덱스 → {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",    default="data/db",  help="DB 경로")
    parser.add_argument("--input", default=None,       help="사용할 docs pkl 파일 (기본: docs_v2.pkl 또는 docs.pkl)")
    parser.add_argument("--batch", type=int, default=64, help="임베딩 배치 크기")
    args = parser.parse_args()
    rebuild(args.db, args.batch, args.input)
