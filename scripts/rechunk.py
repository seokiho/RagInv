"""
법령 조문 통합 + 판례 source 수정 → data/db/docs_v2.pkl

문제:
  - 법령: 하나의 조문(unique_id)이 항/호 단위로 최대 85개 청크로 쪼개짐
  - 판례: source 필드가 비어있어 출처 불명
변경사항:
  - 법령: unique_id 기준으로 청크 병합 → 조문 단위 통합
           1,200자 초과 시 항(①②③) 경계로 재분리
  - 판례: source = "법원명 사건명", doc_type = "case"
  - 법령: doc_type = "law"

Usage:
    python scripts/rechunk.py
    python scripts/rechunk.py --input data/db/docs.pkl --output data/db/docs_v2.pkl
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

MAX_ARTICLE_CHARS = 1200

# 항 마커: ①②③...⑮  (전각 원문자)
_HANG_RE = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]")


# ── Law article merger ────────────────────────────────────────────────────

def _split_by_hang(full_text: str, header: str, max_size: int) -> List[str]:
    """
    full_text 안의 항(①②...) 경계로 분리.
    header(조문 제목)를 각 청크에 포함.
    max_size 안에서 여러 항을 하나로 묶음.
    """
    positions = [m.start() for m in _HANG_RE.finditer(full_text)]
    if not positions:
        return [full_text]

    article_header = full_text[: positions[0]].strip()
    if not article_header:
        article_header = header

    # 항 단위 파트 수집
    parts: List[str] = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(full_text)
        parts.append(full_text[pos:end].strip())

    chunks: List[str] = []
    current = article_header

    for part in parts:
        candidate = current + "\n" + part if current else part
        if current != article_header and len(candidate) > max_size:
            chunks.append(current)
            current = article_header + "\n" + part
        else:
            current = candidate

    if current.strip():
        chunks.append(current)

    return chunks if chunks else [full_text]


def merge_law_docs(law_docs: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    unique_id 기준으로 법령 청크를 조문 단위로 병합.
    같은 고유ID의 청크들이 docs 리스트에서 삽입 순서로 나타남을 이용.
    """
    uid_order: List[str] = []
    uid_groups: Dict[str, List[Dict]] = defaultdict(list)

    for doc in law_docs:
        uid = doc.get("unique_id", "")
        if uid not in uid_groups:
            uid_order.append(uid)
        uid_groups[uid].append(doc)

    merged: List[Dict] = []
    stats = {"single": 0, "merged": 0, "split_large": 0}

    for uid in uid_order:
        chunks = uid_groups[uid]
        base = chunks[0].copy()
        base["doc_type"] = "law"

        texts = [c["text"].strip() for c in chunks if c.get("text", "").strip()]
        full_text = "\n".join(texts)

        if len(chunks) == 1:
            base["text"] = full_text
            merged.append(base)
            stats["single"] += 1

        elif len(full_text) <= MAX_ARTICLE_CHARS:
            base["text"] = full_text
            merged.append(base)
            stats["merged"] += 1

        else:
            # 항 경계로 재분리
            law_name = base.get("law_name", base.get("source", ""))
            sub_texts = _split_by_hang(full_text, law_name, MAX_ARTICLE_CHARS)
            for i, sub_text in enumerate(sub_texts):
                sub = base.copy()
                sub["text"] = sub_text
                if i > 0:
                    sub["unique_id"] = f"{uid}_p{i + 1}"
                merged.append(sub)
            stats["split_large"] += 1

    return merged, stats


# ── Case doc source fixer ─────────────────────────────────────────────────

def fix_case_docs(case_docs: List[Dict]) -> List[Dict]:
    """판례 문서의 source 필드를 채우고 doc_type = 'case' 추가."""
    fixed: List[Dict] = []
    for doc in case_docs:
        d = doc.copy()
        case_name = (d.get("case_name") or "").strip()
        court = (d.get("court") or "").strip()
        case_number = (d.get("case_number") or "").strip()

        if case_name:
            d["source"] = f"{court} {case_name}".strip() if court else case_name
        elif case_number:
            d["source"] = f"{court} {case_number}".strip() if court else case_number
        else:
            d["source"] = court or "판례"

        d["doc_type"] = "case"
        fixed.append(d)
    return fixed


# ── Main ──────────────────────────────────────────────────────────────────

def rechunk(input_path: str, output_path: str) -> None:
    inp = Path(input_path)
    out = Path(output_path)

    print(f"[rechunk] 로드 중: {inp} …")
    with open(inp, "rb") as f:
        docs: List[Dict[str, Any]] = pickle.load(f)
    print(f"[rechunk] 원본 총 {len(docs):,}개 청크")

    # ── 분리 ─────────────────────────────────────────────────────────────
    law_docs  = [d for d in docs if d.get("source", "").strip()]
    case_docs = [d for d in docs if not d.get("source", "").strip()]
    print(f"[rechunk] 법령 청크: {len(law_docs):,}개  /  판례 청크: {len(case_docs):,}개")

    # ── 법령 조문 병합 ────────────────────────────────────────────────────
    print("[rechunk] 법령 조문 병합 중 …")
    merged_law, stats = merge_law_docs(law_docs)
    print(f"[rechunk] 법령 결과: {len(merged_law):,}개 조문")
    print(f"          단일 청크 그대로: {stats['single']:,}")
    print(f"          병합됨:           {stats['merged']:,}")
    print(f"          너무 커서 재분리: {stats['split_large']:,}")

    # ── 판례 source 수정 ──────────────────────────────────────────────────
    print("[rechunk] 판례 source 수정 중 …")
    fixed_cases = fix_case_docs(case_docs)
    print(f"[rechunk] 판례 결과: {len(fixed_cases):,}개")

    # ── 합치기: 법령 먼저 ─────────────────────────────────────────────────
    all_docs = merged_law + fixed_cases
    print(f"\n[rechunk] 최종 문서: {len(all_docs):,}개")
    print(f"          법령: {len(merged_law):,}  /  판례: {len(fixed_cases):,}")

    # ── 텍스트 길이 통계 ──────────────────────────────────────────────────
    law_lens  = [len(d["text"]) for d in merged_law]
    case_lens = [len(d["text"]) for d in fixed_cases]
    if law_lens:
        print(f"\n[rechunk] 법령 텍스트 길이 → 평균 {sum(law_lens)//len(law_lens)}자"
              f"  최대 {max(law_lens)}자  최소 {min(law_lens)}자")
    if case_lens:
        print(f"[rechunk] 판례 텍스트 길이 → 평균 {sum(case_lens)//len(case_lens)}자"
              f"  최대 {max(case_lens)}자  최소 {min(case_lens)}자")

    # ── 저장 ─────────────────────────────────────────────────────────────
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(all_docs, f)
    print(f"\n[rechunk] 저장 완료 → {out}")

    # ── 샘플 출력 ─────────────────────────────────────────────────────────
    print("\n[rechunk] 법령 샘플 (3개):")
    for d in merged_law[:3]:
        print(f"  [{d.get('unique_id','')}]  {len(d['text'])}자")
        print(f"  {repr(d['text'][:120])}")

    print("\n[rechunk] 판례 샘플 (2개):")
    for d in fixed_cases[:2]:
        print(f"  source: {d.get('source','')} | {len(d['text'])}자")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/db/docs.pkl",   help="원본 docs.pkl 경로")
    parser.add_argument("--output", default="data/db/docs_v2.pkl", help="출력 경로")
    args = parser.parse_args()
    rechunk(args.input, args.output)
