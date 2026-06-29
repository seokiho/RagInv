#!/usr/bin/env python3
"""
RAG + 5-shot 추론 — 검색 결과 + AIHUB 예시 5개를 함께 제공.

비교 실험:
  run_inference.py       : RAG + 상세 형식 지시
  run_inference_zeroshot : RAG + 최소 지시
  run_inference_fewshot  : RAG + 5-shot 예시 (이 스크립트)

Usage:
    .venv/bin/python3 scripts/run_inference_fewshot.py --model exaone3.5:7.8b
    .venv/bin/python3 scripts/run_inference_fewshot.py --model llama3.1:8b --db data/db_sectional
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.embedder import KoreanEmbedder
from core.generator import OllamaGenerator
from core.vectorstore import FAISSStore

_LAW_RE = re.compile(
    r"(?:"
    r"(?<![가-힣])[가-힣]{2,20}(?:법률|법|령|규칙)"
    r"|(?<![가-힣])(?:[가-힣]{1,10}\s){0,3}[가-힣]{1,10}에\s+관한\s+법률"
    r")"
    r"\s*제\s*\d+\s*조(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?"
)

SYSTEM_PROMPT = """당신은 형사수사 전문 AI 어시스턴트입니다.
아래 참고 자료와 예시를 참고하여 동일한 형식으로 답변하세요.

【출력 형식】
[죄명]: 핵심 죄명 하나
[답변]: 2~4문장 핵심 답변
[근거법령]: 법명 제X조 — 요약. 없으면 "해당없음"
[참조판례]: 사건번호 (법원) — 요지. 없으면 생략"""


def _load_fewshot_examples(aihub: Path) -> list[dict]:
    slots = {
        "사기_단순":    lambda n: "사기" in n and "특정경제" not in n and "특정범죄" not in n,
        "횡령":         lambda n: "횡령" in n and "특정경제" not in n,
        "배임":         lambda n: "배임" in n and "특정경제" not in n and "자본시장" not in n,
        "자본시장":     lambda n: "자본시장" in n,
        "특정경제범죄": lambda n: "특정경제범죄" in n and "사기" in n,
    }
    collected: dict[str, dict | None] = {k: None for k in slots}

    files = sorted(glob.glob(
        str(aihub / "형사데이터" / "Training" / "02.라벨링데이터" / "TL_판결문_QA" / "*.json")
    ))

    for fpath in files:
        d     = json.loads(Path(fpath).read_text(encoding="utf-8"))
        info  = d.get("info", {})
        label = d.get("label", {})
        name  = info.get("caseName", "")
        q     = label.get("input", "").strip()
        a     = label.get("output", "").strip()
        cnum  = info.get("caseNum", "")
        court = info.get("courtName", "")

        if len(a) < 80 or len(q) < 20:
            continue
        if not _LAW_RE.search(a):
            continue

        for key, fn in slots.items():
            if collected[key] is not None:
                continue
            if fn(name):
                collected[key] = {
                    "query":  q,
                    "answer": a,
                    "case":   f"{cnum} ({court})",
                    "name":   name,
                }
        if all(v is not None for v in collected.values()):
            break

    examples = [v for v in collected.values() if v is not None]
    print(f"[fewshot] {len(examples)}개 예시 로드:")
    for ex in examples:
        print(f"  [{ex['name'][:30]}] {ex['case']}")
    return examples


def _build_prompt(query: str, docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        if doc.get("doc_type") == "law":
            header = f"[법령 {i}] {doc.get('law_name', '')} {doc.get('section', '')}"
        else:
            case_num = doc.get("case_num") or doc.get("source_id", "")
            court    = doc.get("court", "")
            header   = f"[판례 {i}] {case_num} ({court})"
        parts.append(f"{header}\n{doc.get('text', '')[:800]}")

    context = "\n\n---\n\n".join(parts) if parts else "(관련 자료 없음)"
    return f"{context}\n\n[질문]\n{query}"


def _build_messages(examples: list[dict], prompt: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        messages.append({"role": "user",      "content": f"[질문]\n{ex['query']}"})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": prompt})
    return messages


def _extract_section(text: str, section: str) -> str:
    m = re.search(
        rf"\[{re.escape(section)}\]\s*:?\s*(.*?)(?=\[[가-힣A-Za-z]+\]|\Z)",
        text, re.S
    )
    return m.group(1).strip() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="exaone3.5:7.8b")
    ap.add_argument("--db",      default="data/db_aihub")
    ap.add_argument("--eval",    default="data/eval_criminal.jsonl")
    ap.add_argument("--aihub",   default="AIHUB")
    ap.add_argument("--top-k",   default=5, type=int)
    ap.add_argument("--limit",   default=None, type=int)
    ap.add_argument("--timeout", default=120.0, type=float)
    args = ap.parse_args()

    model_safe = args.model.replace(":", "_").replace("/", "_")
    db_tag     = Path(args.db).name
    out_path   = PROJECT_ROOT / "results" / f"preds_{model_safe}_fewshot_{db_tag}.jsonl"
    aihub      = PROJECT_ROOT / args.aihub

    examples = _load_fewshot_examples(aihub)
    if len(examples) < 3:
        print("WARNING: 예시가 3개 미만입니다. AIHUB 경로를 확인하세요.")

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
    print(f"[fewshot] 모델: {args.model} | DB: {args.db} | top-k: {args.top_k} | 출력: {out_path.name}")

    if not todo:
        print("모두 완료됨.")
        return

    store    = FAISSStore.load(str(PROJECT_ROOT / args.db))
    embedder = KoreanEmbedder.get("BAAI/bge-m3")
    gen      = OllamaGenerator(model=args.model, timeout=args.timeout)

    if not gen.is_alive():
        print("ERROR: ollama serve가 실행 중이지 않습니다.")
        sys.exit(1)

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as out_f:
        for rec in tqdm(todo, desc=f"{args.model} fewshot"):
            source_id = rec["source_id"]
            query     = rec["query"]
            try:
                q_vec    = embedder.embed_one(query)
                raw      = store.search(q_vec, k=args.top_k)
                docs     = [doc for doc, _score, _idx in raw]
                prompt   = _build_prompt(query, docs)
                messages = _build_messages(examples, prompt)
                pred_text  = gen.chat(messages)
                pred_crime = _extract_section(pred_text, "죄명")
                pred_laws  = _LAW_RE.findall(pred_text)

                result = {
                    "source_id":    source_id,
                    "jurisdiction": rec.get("jurisdiction"),
                    "doc_type":     rec.get("doc_type"),
                    "pred_crime":   pred_crime,
                    "pred_laws":    pred_laws,
                    "pred_text":    pred_text,
                    "n_docs":       len(docs),
                    "n_shots":      len(examples),
                }
            except Exception as e:
                result = {
                    "source_id":  source_id,
                    "pred_crime": "",
                    "pred_laws":  [],
                    "pred_text":  f"ERROR: {e}",
                    "n_docs":     0,
                    "n_shots":    len(examples),
                }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()


if __name__ == "__main__":
    main()