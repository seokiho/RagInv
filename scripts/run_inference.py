#!/usr/bin/env python3
"""
eval_set.jsonl → results/preds_{model}.jsonl

각 쿼리를 RAG(검색+생성)에 통과시켜 예측 결과 저장.
eval_metrics.py generation / hallucination 서브커맨드가 이 파일을 입력으로 사용.

Usage:
    .venv/bin/python3 scripts/run_inference.py --model exaone3.5:7.8b
    .venv/bin/python3 scripts/run_inference.py --model llama3.1:8b
    .venv/bin/python3 scripts/run_inference.py --model exaone-deep:7.8b
    .venv/bin/python3 scripts/run_inference.py --model exaone3.5:7.8b --limit 100  # 테스트
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

# 법령 인용 추출 패턴 (build_eval_set.py 와 동일)
_LAW_RE = re.compile(
    r"(?:"
    r"(?<![가-힣])[가-힣]{2,20}(?:법률|법|령|규칙)"
    r"|(?<![가-힣])(?:[가-힣]{1,10}\s){0,3}[가-힣]{1,10}에\s+관한\s+법률"
    r")"
    r"\s*제\s*\d+\s*조(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?"
)

SYSTEM_PROMPT = """당신은 형사수사 전문 AI 어시스턴트입니다.
제공된 법령과 판례만을 근거로 답변하세요.

【출력 형식 — 반드시 이 순서대로, 헤더 포함하여 작성】

[죄명]: 사기 (핵심 죄명 하나. 판단 불가시 "해당없음")

[답변]: 2~4문장으로 핵심 답변.

[근거법령]: 법명 제X조 — 내용 요약. 없으면 "해당없음".

[참조판례]: 사건번호 (법원) — 요지 한 줄. 없으면 생략.

위 4개 헤더를 반드시 출력하세요. 다른 형식은 사용하지 마세요."""


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


def _extract_section(text: str, section: str) -> str:
    # "[섹션]: 내용" 또는 "[섹션]\n내용" 두 형식 모두 지원
    m = re.search(
        rf"\[{re.escape(section)}\]\s*:?\s*(.*?)(?=\[[가-힣A-Za-z]+\]|\Z)",
        text, re.S
    )
    return m.group(1).strip() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="exaone3.5:7.8b", help="Ollama 모델명")
    ap.add_argument("--db",      default="data/db_aihub",  help="FAISS DB 경로")
    ap.add_argument("--eval",    default="data/eval_set.jsonl")
    ap.add_argument("--top-k",   default=5, type=int,      help="검색 top-K")
    ap.add_argument("--limit",   default=None, type=int,   help="쿼리 수 제한 (테스트용)")
    ap.add_argument("--timeout", default=120.0, type=float)
    args = ap.parse_args()

    model_safe = args.model.replace(":", "_").replace("/", "_")
    db_tag     = Path(args.db).name
    out_path   = PROJECT_ROOT / "results" / f"preds_{model_safe}_{db_tag}.jsonl"

    # ── 체크포인트 로드 ────────────────────────────────────────────────────
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["source_id"])
                except Exception:
                    pass
        print(f"체크포인트: {len(done_ids):,}개 완료")

    # ── eval set 로드 ──────────────────────────────────────────────────────
    with open(PROJECT_ROOT / args.eval, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        records = records[: args.limit]

    todo = [r for r in records if r["source_id"] not in done_ids]
    print(f"남은 쿼리: {len(todo):,}개 / 전체 {len(records):,}개")

    if not todo:
        print("모두 완료됨.")
        return

    # ── 컴포넌트 초기화 ────────────────────────────────────────────────────
    print(f"[run_inference] FAISS 로드 중...")
    store    = FAISSStore.load(str(PROJECT_ROOT / args.db))
    embedder = KoreanEmbedder.get("BAAI/bge-m3")
    gen      = OllamaGenerator(model=args.model, timeout=args.timeout)

    if not gen.is_alive():
        print("ERROR: ollama serve가 실행 중이지 않습니다. 터미널에서 'ollama serve' 실행 후 재시도.")
        sys.exit(1)

    print(f"[run_inference] 모델: {args.model} | top-k: {args.top_k} | 출력: {out_path.name}")

    # ── 추론 루프 ──────────────────────────────────────────────────────────
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as out_f:
        for rec in tqdm(todo, desc=args.model):
            source_id = rec["source_id"]
            query     = rec["query"]

            try:
                q_vec    = embedder.embed_one(query)
                raw      = store.search(q_vec, k=args.top_k)
                docs     = [doc for doc, _score, _idx in raw]
                prompt   = _build_prompt(query, docs)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ]
                pred_text = gen.chat(messages)

                pred_crime = _extract_section(pred_text, "죄명")
                pred_laws  = _LAW_RE.findall(pred_text)

                result = {
                    "source_id":    source_id,
                    "jurisdiction": rec.get("jurisdiction"),
                    "doc_type":     rec.get("doc_type"),
                    "pred_crime":   pred_crime,
                    "pred_laws":    pred_laws,
                    "pred_text":    pred_text,
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
                    "error":        str(e),
                }

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"\n[run_inference] 완료 → {out_path}")
    print(f"다음 단계:")
    print(f"  .venv/bin/python3 scripts/eval_metrics.py generation --pred {out_path} --gold data/eval_set.jsonl")
    print(f"  .venv/bin/python3 scripts/eval_metrics.py hallucination --pred {out_path}")


if __name__ == "__main__":
    main()
