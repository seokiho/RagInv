"""
Ingest raw text documents → FAISS index (for building a NEW index from scratch).

NOT needed if you already have db/ from Prj_RAG — use migrate_db.py instead.

Usage:
    python scripts/ingest.py --input data/raw/ --output data/db/

Supported input files in data/raw/:
  *.txt    plain text
  *.json   list of {"content": "...", "source": "..."} objects

Chunking strategies (set CHUNK_STRATEGY in .env):
  section      (default) split on Korean legal section headers (제X조, 제X절 etc.)
  fixed        fixed-size sliding window
  hierarchical section-first, then fixed if section is too large
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import settings
from core.embedder import KoreanEmbedder
from core.vectorstore import FAISSStore


# ── Section-aware chunker ────────────────────────────────────────────────
# Korean legal document patterns
_SECTION_RE = re.compile(
    r"(제\s*\d+\s*(?:조|절|장|편|항|호)[\s.\)])",
    re.UNICODE,
)


def chunk_section(text: str, source: str) -> list[dict]:
    """Split on Korean legal section markers, yield document dicts."""
    parts = _SECTION_RE.split(text)
    chunks = []
    current_header = ""
    for part in parts:
        if _SECTION_RE.fullmatch(part.strip()):
            current_header = part.strip()
        elif part.strip():
            body = (current_header + " " + part).strip()
            # Sub-split if still too large
            if len(body) > settings.chunk_size * 2:
                for sub in chunk_fixed(body, source, current_header):
                    chunks.append(sub)
            else:
                chunks.append({
                    "text":    body[: settings.chunk_size * 2],
                    "source":  source,
                    "section": current_header,
                })
    return chunks or [{"text": text[:settings.chunk_size], "source": source, "section": ""}]


def chunk_fixed(text: str, source: str, section: str = "") -> list[dict]:
    """Sliding-window fixed-size chunks."""
    size, overlap = settings.chunk_size, settings.chunk_overlap
    chunks = []
    for i in range(0, len(text), size - overlap):
        body = text[i: i + size].strip()
        if body:
            chunks.append({"text": body, "source": source, "section": section})
    return chunks


def chunk_text(text: str, source: str, strategy: str = "section") -> list[dict]:
    if strategy == "section":
        return chunk_section(text, source)
    elif strategy == "fixed":
        return chunk_fixed(text, source)
    else:  # hierarchical
        raw = chunk_section(text, source)
        result = []
        for c in raw:
            if len(c["text"]) > settings.chunk_size:
                result.extend(chunk_fixed(c["text"], source, c["section"]))
            else:
                result.append(c)
        return result


# ── Ingest ───────────────────────────────────────────────────────────────
def ingest(input_dir: str, output_dir: str) -> None:
    embedder = KoreanEmbedder.get(settings.embed_model)
    store    = FAISSStore(dim=settings.embed_dim)

    all_chunks: list[dict[str, Any]] = []
    input_path = Path(input_dir)

    for fpath in sorted(input_path.glob("*.txt")):
        print(f"  Parsing {fpath.name} …")
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        all_chunks.extend(chunk_text(text, fpath.name, settings.chunk_strategy))

    for fpath in sorted(input_path.glob("*.json")):
        print(f"  Parsing {fpath.name} …")
        data = json.loads(fpath.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                content = item.get("content") or item.get("text") or str(item)
                src = item.get("source", fpath.name)
                all_chunks.extend(chunk_text(content, src, settings.chunk_strategy))

    if not all_chunks:
        print("No files found in", input_dir)
        return

    print(f"\n[ingest] {len(all_chunks)} chunks from {input_dir}")

    # Embed in batches
    batch_size = 64
    all_vecs = []
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding"):
        batch = all_chunks[i: i + batch_size]
        vecs  = embedder.embed([c["text"] for c in batch])
        all_vecs.append(vecs)

    all_vecs_np = np.vstack(all_vecs)
    store.add(all_vecs_np, all_chunks)
    store.save(output_dir)
    print(f"[ingest] Done — index at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    default="data/raw",  help="Directory of .txt/.json files")
    parser.add_argument("--output",   default="data/db",   help="FAISS output directory")
    args = parser.parse_args()
    ingest(args.input, args.output)
