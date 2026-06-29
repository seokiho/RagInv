#!/usr/bin/env python3
"""
AIHUB Validation 라벨 JSON → 평가셋 (eval_set.jsonl) 생성

추출 대상: Validation 라벨링 데이터 (VL_*)
  - 형사: 판결문_QA, 결정례_QA, 해석례_QA
  - 민사: 판결문_QA, 심결례_QA, 유권해석_QA

출력 형식 (줄당 JSON):
  {
    "query":       "질문 텍스트",
    "gold_name":   "사기등",           # 죄명/사건명 (죄명 분류 gold)
    "gold_num":    "2023노809",        # 사건번호 (환각 체크 gold)
    "gold_court":  "서울서부지방법원",
    "gold_date":   "2024.05.02",
    "source_id":   "337906",           # precedId/doc_id (leave-one-out 키)
    "jurisdiction":"criminal",
    "doc_type":    "case",
    "split":       "val"
  }

Usage:
    .venv/bin/python3 scripts/build_eval_set.py
    .venv/bin/python3 scripts/build_eval_set.py --out data/eval_set.jsonl
    .venv/bin/python3 scripts/build_eval_set.py --stat   # 통계만 출력
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

AIHUB = PROJECT_ROOT / "AIHUB"

# 법령 인용 패턴
# - compact: 형법, 전기용품안전관리법, 풍속영업의규제에관한법률  (공백 없음)
# - spaced:  마약류 관리에 관한 법률, 특정경제범죄 가중처벌 등에 관한 법률
_LAW_RE = re.compile(
    r"(?:"
    r"(?<![가-힣])[가-힣]{2,20}(?:법률|법|령|규칙)"                           # compact (공백 없음)
    r"|(?<![가-힣])(?:[가-힣]{1,10}\s){0,3}[가-힣]{1,10}에\s+관한\s+법률"    # X(Y)(Z)에 관한 법률 (공백 필수)
    r")"
    r"\s*제\s*\d+\s*조(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?"
)


def _extract_law_citations(text: str) -> list[str]:
    """정답 텍스트에서 법령 조항 인용 추출."""
    return [m.strip() for m in _LAW_RE.findall(text)]


# ── 형사 파서 ─────────────────────────────────────────────────────────────────

def _parse_criminal_case_qa(aihub: Path):
    """형사 판결문_QA → eval 레코드."""
    for split_dir, vl_prefix in [("Training", "TL"), ("Validation", "VL")]:
        # 평가셋은 Validation만 — Training은 건너뜀
        if vl_prefix == "TL":
            continue
        folder = aihub / "형사데이터" / split_dir / "02.라벨링데이터" / f"{vl_prefix}_판결문_QA"
        if not folder.exists():
            continue
        for jf in sorted(folder.glob("*.json")):
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            info  = d.get("info", {})
            label = d.get("label", {})
            query = label.get("input", "").strip()
            if not query:
                continue
            yield {
                "query":        query,
                "gold_name":    info.get("caseName", ""),
                "gold_num":     info.get("caseNum", ""),
                "gold_court":   info.get("courtName", ""),
                "gold_date":    info.get("sentenceDate", ""),
                "gold_laws":    _extract_law_citations(label.get("output", "")),
                "source_id":    str(info.get("precedId", "")),
                "jurisdiction": "criminal",
                "doc_type":     "case",
                "split":        "val",
            }


def _parse_criminal_decision_qa(aihub: Path):
    """형사 결정례_QA → eval 레코드."""
    folder = aihub / "형사데이터" / "Validation" / "02.라벨링데이터" / "VL_결정례_QA"
    if not folder.exists():
        return
    for jf in sorted(folder.glob("*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        info  = d.get("info", {})
        label = d.get("label", {})
        query = label.get("input", "").strip()
        if not query:
            continue
        yield {
            "query":        query,
            "gold_name":    info.get("caseName", ""),
            "gold_num":     info.get("caseNum", ""),
            "gold_court":   info.get("courtCode", ""),
            "gold_date":    info.get("finalDate", ""),
            "gold_laws":    _extract_law_citations(label.get("output", "")),
            "source_id":    str(info.get("determintId", "")),
            "jurisdiction": "criminal",
            "doc_type":     "decision",
            "split":        "val",
        }


def _parse_criminal_interpretation_qa(aihub: Path):
    """형사 해석례_QA → eval 레코드."""
    folder = aihub / "형사데이터" / "Validation" / "02.라벨링데이터" / "VL_해석례_QA"
    if not folder.exists():
        return
    for jf in sorted(folder.glob("*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        info  = d.get("info", {})
        label = d.get("label", {})
        query = label.get("input", "").strip()
        if not query:
            continue
        yield {
            "query":        query,
            "gold_name":    info.get("agenda", "")[:80],
            "gold_num":     info.get("agendaNum", ""),
            "gold_court":   info.get("interpreMinName", ""),
            "gold_date":    info.get("interpreDate", ""),
            "gold_laws":    _extract_law_citations(label.get("output", "")),
            "source_id":    str(info.get("interpreId", "")),
            "jurisdiction": "criminal",
            "doc_type":     "interpretation",
            "split":        "val",
        }


# ── 민사 파서 ─────────────────────────────────────────────────────────────────

def _parse_civil_case_qa(aihub: Path):
    """민사 판결문_QA → eval 레코드."""
    folder = (aihub / "민사데이터" / "Validation" / "02.라벨링데이터"
              / "VL_01. 민사법_001. 판결문_0001. 질의응답")
    if not folder.exists():
        return
    for jf in sorted(folder.glob("*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        info     = d.get("info", {})
        taskinfo = d.get("taskinfo", {})
        query    = taskinfo.get("input", "").strip()
        if not query:
            continue
        yield {
            "query":        query,
            "gold_name":    info.get("casenames", ""),
            "gold_num":     info.get("doc_id", ""),
            "gold_court":   info.get("normalized_court", ""),
            "gold_date":    str(info.get("announce_date", ""))[:10],
            "gold_laws":    _extract_law_citations(taskinfo.get("output", "")),
            "source_id":    info.get("doc_id", jf.stem),
            "jurisdiction": "civil",
            "doc_type":     "case",
            "split":        "val",
        }


def _parse_civil_decision_qa(aihub: Path):
    """민사 심결례_QA → eval 레코드."""
    folder = (aihub / "민사데이터" / "Validation" / "02.라벨링데이터"
              / "VL_01. 민사법_003. 심결례_0001. 질의응답")
    if not folder.exists():
        return
    for jf in sorted(folder.glob("*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        info     = d.get("info", {})
        taskinfo = d.get("taskinfo", {})
        query    = taskinfo.get("input", "").strip()
        if not query:
            continue
        yield {
            "query":        query,
            "gold_name":    info.get("document_type", ""),
            "gold_num":     info.get("doc_id", ""),
            "gold_court":   "",
            "gold_date":    info.get("decision_date", ""),
            "gold_laws":    _extract_law_citations(taskinfo.get("output", "")),
            "source_id":    info.get("doc_id", jf.stem),
            "jurisdiction": "civil",
            "doc_type":     "decision",
            "split":        "val",
        }


def _parse_civil_interpretation_qa(aihub: Path):
    """민사 유권해석_QA → eval 레코드."""
    folder = (aihub / "민사데이터" / "Validation" / "02.라벨링데이터"
              / "VL_01. 민사법_004. 유권해석_0001. 질의응답")
    if not folder.exists():
        return
    for jf in sorted(folder.glob("*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        info     = d.get("info", {})
        taskinfo = d.get("taskinfo", {})
        query    = taskinfo.get("input", "").strip()
        if not query:
            continue
        yield {
            "query":        query,
            "gold_name":    info.get("title", ""),
            "gold_num":     info.get("doc_id", ""),
            "gold_court":   info.get("response_institute", ""),
            "gold_date":    info.get("response_date", ""),
            "gold_laws":    _extract_law_citations(taskinfo.get("output", "")),
            "source_id":    info.get("doc_id", jf.stem),
            "jurisdiction": "civil",
            "doc_type":     "interpretation",
            "split":        "val",
        }


# ── 메인 ─────────────────────────────────────────────────────────────────────

def build(aihub: Path, out_path: Path, stat_only: bool = False) -> None:
    parsers = [
        _parse_criminal_case_qa(aihub),
        _parse_criminal_decision_qa(aihub),
        _parse_criminal_interpretation_qa(aihub),
        _parse_civil_case_qa(aihub),
        _parse_civil_decision_qa(aihub),
        _parse_civil_interpretation_qa(aihub),
    ]

    records = []
    for parser in parsers:
        for rec in parser:
            records.append(rec)

    # 통계
    jur   = Counter(r["jurisdiction"] for r in records)
    types = Counter(f"{r['jurisdiction']}/{r['doc_type']}" for r in records)
    print(f"[eval_set] 총 {len(records):,}개 쿼리")
    print(f"  형사: {jur['criminal']:,}  /  민사: {jur['civil']:,}")
    for k, v in sorted(types.items()):
        print(f"  {k}: {v:,}")

    # 샘플 출력
    if records:
        print(f"\n[샘플]")
        for r in records[:2]:
            print(f"  query:      {r['query'][:60]}...")
            print(f"  gold_name:  {r['gold_name']}")
            print(f"  gold_num:   {r['gold_num']}")
            print(f"  source_id:  {r['source_id']}")
            print()

    if stat_only:
        return

    # 저장
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[eval_set] 저장 완료 → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AIHUB Validation → eval_set.jsonl")
    ap.add_argument("--aihub", default=str(AIHUB),                          help="AIHUB 루트")
    ap.add_argument("--out",   default=str(PROJECT_ROOT / "data/eval_set.jsonl"), help="출력 경로")
    ap.add_argument("--stat",  action="store_true",                          help="통계만 출력")
    args = ap.parse_args()

    build(Path(args.aihub), Path(args.out), stat_only=args.stat)
