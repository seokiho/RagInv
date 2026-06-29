#!/usr/bin/env python3
"""
LoRA 파인튜닝 모델로 RAG 추론 — M2 맥미니 (mlx-lm) 전용.

사전 조건 (M2 맥미니에서):
    pip install mlx-lm
    mlx_lm.lora --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
                --train --data data/lora_train \
                --iters 1000 --batch-size 4 --lora-layers 16 \
                --adapter-path adapters/llama_eco_crime

Usage:
    python3 scripts/run_inference_lora.py
    python3 scripts/run_inference_lora.py --limit 300
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

_LAW_RE = re.compile(
    r"(?:"
    r"(?<![가-힣])[가-힣]{2,20}(?:법률|법|령|규칙)"
    r"|(?<![가-힣])(?:[가-힣]{1,10}\s){0,3}[가-힣]{1,10}에\s+관한\s+법률"
    r")"
    r"\s*제\s*\d+\s*조(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?"
)

SYSTEM = """당신은 형사수사 전문 AI 어시스턴트입니다.
제공된 판결문과 법령만을 근거로 답변하세요.

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
    m = re.search(
        rf"\[{re.escape(section)}\]\s*:?\s*(.*?)(?=\[[가-힣A-Za-z]+\]|\Z)",
        text, re.S
    )
    return m.group(1).strip() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    default="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    ap.add_argument("--adapter",  default="adapters/llama_eco_crime")
    ap.add_argument("--db",       default="data/db_aihub")
    ap.add_argument("--eval",     default="data/eval_criminal.jsonl")
    ap.add_argument("--top-k",    default=5, type=int)
    ap.add_argument("--limit",    default=None, type=int)
    ap.add_argument("--max-tokens", default=512, type=int)
    args = ap.parse_args()

    # mlx-lm import (M2 맥미니에서만 사용 가능)
    try:
        from mlx_lm import load, generate
    except ImportError:
        print("ERROR: mlx-lm이 설치되어 있지 않습니다.")
        print("  pip install mlx-lm")
        sys.exit(1)

    from core.embedder import KoreanEmbedder
    from core.vectorstore import FAISSStore

    adapter_path = PROJECT_ROOT / args.adapter
    out_path     = PROJECT_ROOT / "results" / "preds_llama_lora.jsonl"

    if not adapter_path.exists():
        print(f"ERROR: 어댑터 없음: {adapter_path}")
        print("먼저 M2에서 LoRA 훈련을 완료하세요.")
        sys.exit(1)

    # 체크포인트
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
    print(f"남은 쿼리: {len(todo):,}개")

    print(f"[lora] 모델 로드 중: {args.model}")
    model, tokenizer = load(args.model, adapter_path=str(adapter_path))

    print(f"[lora] FAISS 로드 중...")
    store    = FAISSStore.load(str(PROJECT_ROOT / args.db))
    embedder = KoreanEmbedder.get("BAAI/bge-m3")

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as out_f:
        for rec in tqdm(todo, desc="llama-lora"):
            source_id = rec["source_id"]
            query     = rec["query"]
            try:
                q_vec   = embedder.embed_one(query)
                raw     = store.search(q_vec, k=args.top_k)
                docs    = [doc for doc, _s, _i in raw]
                content = _build_prompt(query, docs)

                # mlx-lm은 apply_chat_template 사용
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": content},
                ]
                prompt_str = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                pred_text  = generate(model, tokenizer, prompt=prompt_str,
                                      max_tokens=args.max_tokens, verbose=False)
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
                }
            except Exception as e:
                result = {
                    "source_id":  source_id,
                    "pred_crime": "",
                    "pred_laws":  [],
                    "pred_text":  f"ERROR: {e}",
                    "n_docs":     0,
                }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"\n완료. 결과: {out_path}")
    print("평가:")
    print("  python3 scripts/eval_metrics.py generation --pred results/preds_llama_lora.jsonl --gold data/eval_criminal.jsonl --out results/generation_llama_lora.json")
    print("  python3 scripts/eval_metrics.py hallucination --pred results/preds_llama_lora.jsonl --out results/hallucination_llama_lora.json")


if __name__ == "__main__":
    main()
