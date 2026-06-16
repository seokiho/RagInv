"""
One-time migration: LangChain FAISS format  →  RagInv native format.

LangChain stores:
  db/index.faiss   ← raw FAISS binary (can read directly with faiss.read_index)
  db/index.pkl     ← LangChain InMemoryDocstore + index_to_docstore_id mapping

RagInv native format:
  data/db/index.faiss   ← same FAISS binary (copied as-is)
  data/db/docs.pkl      ← plain Python list of dicts

Usage:
    python scripts/migrate_db.py --src db/ --dst data/db/

After this runs once you can remove langchain* from requirements.txt.
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def migrate(src_dir: str, dst_dir: str) -> None:
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    faiss_src = src / "index.faiss"
    pkl_src   = src / "index.pkl"

    if not faiss_src.exists():
        raise FileNotFoundError(f"index.faiss not found in {src_dir}")
    if not pkl_src.exists():
        raise FileNotFoundError(f"index.pkl not found in {src_dir}")

    # ── 1. Copy the raw FAISS binary unchanged ───────────────────────
    faiss_dst = dst / "index.faiss"
    shutil.copy2(faiss_src, faiss_dst)
    print(f"[migrate] Copied index.faiss → {faiss_dst}")

    # ── 2. Unpack LangChain pickle into plain list ───────────────────
    print("[migrate] Loading LangChain pickle …")
    with open(pkl_src, "rb") as f:
        lc_data = pickle.load(f)

    # LangChain FAISS pickle — two known formats:
    #   New (tuple):  (InMemoryDocstore, {int: uuid_str})
    #   Old (dict):   {"docstore": ..., "index_to_docstore_id": ...}

    if isinstance(lc_data, tuple) and len(lc_data) == 2:
        # ── Tuple format (most common in newer LangChain) ──────────────
        docstore, idx_to_id = lc_data
        store_dict = docstore._dict
        print(f"[migrate] Tuple format: {len(idx_to_id)} index entries, "
              f"{len(store_dict)} docs in docstore")

    elif isinstance(lc_data, dict) and "index_to_docstore_id" in lc_data:
        # ── Dict format (older LangChain) ──────────────────────────────
        idx_to_id  = lc_data["index_to_docstore_id"]
        docstore   = lc_data["docstore"]
        store_dict = docstore._dict
        print(f"[migrate] Dict format: {len(idx_to_id)} index entries")

    else:
        print("[migrate] Pickle is not a recognised LangChain format — treating as plain list")
        docs = lc_data if isinstance(lc_data, list) else []
        idx_to_id = None   # skip the loop below

    if idx_to_id is not None:
        n = len(idx_to_id)
        docs = []
        for i in range(n):
            uid    = idx_to_id.get(i)
            lc_doc = store_dict.get(uid)
            if lc_doc is None:
                docs.append({"text": "", "source": "", "section": ""})
                continue
            # LangChain Document: .page_content + .metadata
            meta = lc_doc.metadata if hasattr(lc_doc, "metadata") else {}
            docs.append({
                "text":    lc_doc.page_content,
                "source":  meta.get("source",
                           meta.get("law_name",
                           meta.get("unique_id", ""))),
                "section": meta.get("section",
                           meta.get("article_num",
                           meta.get("chapter", ""))),
                # Keep all original metadata fields for research queries
                **{k: v for k, v in meta.items()
                   if k not in ("source", "section")},
            })
        print(f"[migrate] Extracted {len(docs)} documents")

    # ── 3. Save native docs.pkl ──────────────────────────────────────
    docs_dst = dst / "docs.pkl"
    with open(docs_dst, "wb") as f:
        pickle.dump(docs, f)
    print(f"[migrate] Saved docs.pkl ({len(docs)} docs) → {docs_dst}")

    # ── 4. Quick sanity check ────────────────────────────────────────
    import faiss
    idx = faiss.read_index(str(faiss_dst))
    assert idx.ntotal == len(docs), (
        f"FAISS has {idx.ntotal} vectors but docs has {len(docs)} entries! "
        "Something is wrong."
    )
    print(f"[migrate] ✓ Sanity check passed: {idx.ntotal} vectors = {len(docs)} docs")

    # ── 5. Print sample ──────────────────────────────────────────────
    if docs:
        sample = docs[0]
        print(f"\n[migrate] Sample doc[0]:")
        print(f"  source:  {sample.get('source', '')}")
        print(f"  section: {sample.get('section', '')}")
        print(f"  text:    {sample.get('text', '')[:120]} …")

    print(f"\n[migrate] Done! Native DB at: {dst_dir}")
    print("You can now delete langchain* from requirements.txt.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="db/",      help="Source LangChain DB dir")
    parser.add_argument("--dst", default="data/db/", help="Target native DB dir")
    args = parser.parse_args()
    migrate(args.src, args.dst)
