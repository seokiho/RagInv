#!/usr/bin/env python3
"""
형사 쿼리에 민사(경제범죄 관련) 판례를 교차 검색해서 추가하는 실험용 추론 스크립트.

민사 허용 셋: 형사 경제범죄 쿼리 1,563개와 의미적으로 가까운 민사 판례 4,254개
             (data/civil_clean_ids.json, 임베딩 유사도 + 빈도 기반 선별)

기존 run_inference.py 대비 차이:
  - FAISS k=fetch_k 검색 후 형사 top-K + 민사(허용 셋 내) top-civil-k 분리
  - 결과에 n_criminal_docs, n_civil_docs 필드 추가
  - 출력 파일: preds_{model}_cross.jsonl

Usage:
    .venv/bin/python3 scripts/run_inference_cross.py --model exaone3.5:7.8b
    .venv/bin/python3 scripts/run_inference_cross.py --model exaone3.5:7.8b --limit 50
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.embedder import KoreanEmbedder
from core.vectorstore import FAISSStore
from core.generator import OllamaGenerator

_LAW_RE = re.compile(
    r"(?:"
    r"(?<![가-힣])[가-힣]{2,20}(?:법률|법|령|규칙)"
    r"|(?<![가-힣])(?:[가-힣]{1,10}\s){0,3}[가-힣]{1,10}에\s+관한\s+법률"
    r")"
    r"\s*제\s*\d+\s*조(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?"
)

SYSTEM_PROMPT = """당신은 형사수사 전문 AI 어시스턴트입니다.
제공된 형사판례, 민사판례, 법령만을 근거로 답변하세요.

【출력 형식 — 반드시 이 순서대로, 헤더 포함하여 작성】

[죄명]: 사기 (핵심 죄명 하나. 판단 불가시 "해당없음")

[답변]: 2~4문장으로 핵심 답변.

[근거법령]: 법명 제X조 — 내용 요약. 없으면 "해당없음".

[참조판례]: 사건번호 (법원) — 요지 한 줄. 없으면 생략.

위 4개 헤더를 반드시 출력하세요. 다른 형식은 사용하지 마세요."""


def _build_prompt(query: str, criminal_docs: list[dict], civil_docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(criminal_docs, 1):
        case_num = doc.get("case_num") or doc.get("source_id", "")
        court    = doc.get("court", "")
        parts.append(f"[형사판례 {i}] {case_num} ({court})\n{doc.get('text','')[:700]}")

    for i, doc in enumerate(civil_docs, 1):
        case_num = doc.get("case_num") or doc.get("source_id", "")
        court    = doc.get("court", "")
        parts.append(f"[민사판례 {i}] {case_num} ({court})\n{doc.get('text','')[:700]}")

    context = "\n\n---\n\n".join(parts) if parts else "(관련 자료 없음)"
    return f"{context}\n\n[질문]\n{query}"


def _extract_section(text: str, section: str) -> str:
    m = re.search(
        rf"\[{re.escape(section)}\]\s*:?\s*(.*?)(?=\[[가-힣A-Za-z]+\]|\Z)",
        text, re.S
    )
    return m.group(1).strip() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    default="exaone3.5:7.8b")
    ap.add_argument("--db",       default="data/db_aihub")
    ap.add_argument("--eval",     default="data/eval_criminal.jsonl")
    ap.add_argument("--civil-ids", default="data/civil_clean_ids.json")
    ap.add_argument("--top-k",    default=5,  type=int, help="형사 검색 top-K")
    ap.add_argument("--civil-k",  default=2,  type=int, help="민사 교차 검색 top-K")
    ap.add_argument("--fetch-k",  default=40, type=int, help="FAISS 초기 후보 수")
    ap.add_argument("--limit",    default=None, type=int)
    ap.add_argument("--timeout",  default=120.0, type=float)
    args = ap.parse_args()

    model_safe = args.model.replace(":", "_").replace("/", "_")
    db_tag     = Path(args.db).name
    out_path   = PROJECT_ROOT / "results" / f"preds_{model_safe}_cross_{db_tag}.jsonl"

    with open(PROJECT_ROOT / args.civil_ids) as f:
        civil_allowed: set[str] = set(json.load(f))
    print(f"[cross] 민사 허용 source_id: {len(civil_allowed)}개")

    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["source_id"])
                except Exception:
                    pass
        print(f"체크포인트: {len(done_ids):,}개 완료")

    with open(PROJECT_ROOT / args.eval, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        records = records[: args.limit]

    todo = [r for r in records if r["source_id"] not in done_ids]
    print(f"남은 쿼리: {len(todo):,}개 / 전체 {len(records):,}개")
    if not todo:
        print("모두 완료됨.")
        return

    print("[cross] FAISS 로드 중...")
    store    = FAISSStore.load(str(PROJECT_ROOT / args.db))
    embedder = KoreanEmbedder.get("BAAI/bge-m3")
    gen      = OllamaGenerator(model=args.model, timeout=args.timeout)

    if not gen.is_alive():
        print("ERROR: ollama serve가 실행 중이지 않습니다.")
        sys.exit(1)

    print(f"[cross] 모델: {args.model} | 형사 top-{args.top_k} + 민사 top-{args.civil_k} | 출력: {out_path.name}")

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as out_f:
        for rec in tqdm(todo, desc=f"{args.model}+civil"):
            source_id = rec["source_id"]
            query     = rec["query"]

            try:
                q_vec = embedder.embed_one(query)
                raw   = store.search(q_vec, k=args.fetch_k)

                criminal_docs, civil_docs = [], []
                for doc, _score, _idx in raw:
                    if doc.get("jurisdiction") == "criminal" and len(criminal_docs) < args.top_k:
                        criminal_docs.append(doc)
                    elif (doc.get("jurisdiction") == "civil"
                          and doc.get("source_id") in civil_allowed
                          and len(civil_docs) < args.civil_k):
                        civil_docs.append(doc)

                prompt   = _build_prompt(query, criminal_docs, civil_docs)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ]
                pred_text = gen.chat(messages)

                pred_crime = _extract_section(pred_text, "죄명")
                pred_laws  = _LAW_RE.findall(pred_text)

                result = {
                    "source_id":       source_id,
                    "jurisdiction":    rec.get("jurisdiction"),
                    "doc_type":        rec.get("doc_type"),
                    "pred_crime":      pred_crime,
                    "pred_laws":       pred_laws,
                    "pred_text":       pred_text,
                    "n_criminal_docs": len(criminal_docs),
                    "n_civil_docs":    len(civil_docs),
                }

            except Exception as e:
                tqdm.write(f"  ERROR {source_id}: {e}")
                result = {
                    "source_id":    source_id,
                    "jurisdiction": rec.get("jurisdiction"),
                    "doc_type":     rec.get("doc_type"),
                    "pred_crime":   "",
                    "pred_laws":    [],
                    "pred_text":    "",
                    "n_criminal_docs": 0,
                    "n_civil_docs":    0,
                    "error":        str(e),
                }

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"\n[cross] 완료 → {out_path}")
    print(f"다음 단계 (기존 결과와 비교):")
    print(f"  .venv/bin/python3 scripts/eval_metrics.py hallucination --pred {out_path}")
    print(f"  .venv/bin/python3 scripts/eval_metrics.py generation --pred {out_path} --gold data/eval_criminal.jsonl")


if __name__ == "__main__":
    main()
