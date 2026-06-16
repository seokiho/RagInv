"""
법령 구조화 DB 생성: data/law_db.json

docs.pkl의 법령 문서를 조문 단위로 집약.
rechunk 시 _p2, _p3으로 분할된 것도 재통합해 완전한 원문 보장.

구조:
  {
    "형법_329": {
      "law_name": "형법",
      "article_num": "329",
      "enforcement_date": "20230808",
      "full_text": "절도\\n제329조(절도) 타인의 재물을 절취한 자는..."
    },
    ...
  }

Usage:
    python scripts/build_law_db.py
    python scripts/build_law_db.py --input data/db/docs.pkl --output data/law_db.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path


def _base_uid(uid: str) -> str:
    """_p2, _p3 … suffix 제거 → 원본 조문 ID."""
    return re.sub(r"_p\d+$", "", uid)


def build(docs_path: str, output_path: str) -> None:
    print(f"[build_law_db] 로드 중: {docs_path} …")
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)

    law_docs = [d for d in docs if d.get("doc_type") == "law"]
    print(f"[build_law_db] 법령 문서: {len(law_docs):,}개")

    # base_uid 기준으로 그루핑 (삽입 순서 유지)
    uid_order: list[str] = []
    groups: dict[str, list] = defaultdict(list)

    for d in law_docs:
        raw_uid = d.get("unique_id", "")
        uid = _base_uid(raw_uid)
        if uid not in groups:
            uid_order.append(uid)
        groups[uid].append(d)

    law_db: dict[str, dict] = {}
    for uid in uid_order:
        chunks = groups[uid]
        first = chunks[0]

        # 파트가 여럿이면 전체 텍스트 재통합
        full_text = "\n".join(
            c["text"].strip() for c in chunks if c.get("text", "").strip()
        )
        if not full_text:
            continue

        law_db[uid] = {
            "law_name":         first.get("law_name")  or first.get("source", ""),
            "law_num":          first.get("law_num",   ""),
            "article_num":      first.get("article_num") or first.get("section", ""),
            "enforcement_date": first.get("enforcement_date", ""),
            "full_text":        full_text,
        }

    # 저장
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(law_db, f, ensure_ascii=False, indent=2)

    print(f"[build_law_db] {len(law_db):,}개 조문 저장 → {out}")

    # 통계
    lengths = [len(e["full_text"]) for e in law_db.values()]
    print(f"[build_law_db] 텍스트 길이 — 평균 {sum(lengths)//len(lengths)}자  "
          f"최대 {max(lengths)}자  최소 {min(lengths)}자")

    # 샘플
    print("\n[build_law_db] 샘플:")
    for uid in list(law_db.keys())[100:103]:
        e = law_db[uid]
        print(f"  [{uid}] {e['law_name']} 제{e['article_num']}조 | "
              f"{len(e['full_text'])}자")
        print(f"  {repr(e['full_text'][:120])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/db/docs.pkl")
    parser.add_argument("--output", default="data/law_db.json")
    args = parser.parse_args()
    build(args.input, args.output)
