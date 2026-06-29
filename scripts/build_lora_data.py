#!/usr/bin/env python3
"""
AIHUB Training QA → LoRA 훈련 데이터 (mlx-lm 형식)

처리 흐름:
  형사 TL_판결문_QA JSON  +  TS_판결문 CSV (원문)
  → 경제범죄 필터링
  → [시스템] + [판결문 컨텍스트 + 질문] + [구조화 답변]
  → data/lora_train/train.jsonl, valid.jsonl

Usage:
    .venv/bin/python3 scripts/build_lora_data.py
    .venv/bin/python3 scripts/build_lora_data.py --stat   # 통계만
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

AIHUB = PROJECT_ROOT / "AIHUB"
OUT_DIR = PROJECT_ROOT / "data" / "lora_train"

ECO_CRIMES = ["사기", "횡령", "배임", "절도", "공갈", "뇌물", "자본시장", "금융", "알선", "특정경제범죄"]

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


def _load_source_text(aihub: Path, prec_id: str) -> str:
    """TS_판결문 CSV에서 판례 원문 텍스트 조합."""
    csv_path = aihub / "형사데이터" / "Training" / "01.원천데이터" / "TS_판결문" / f"HS_P_{prec_id}.csv"
    if not csv_path.exists():
        return ""
    try:
        rows = list(csv.reader(csv_path.open(encoding="utf-8")))
        # header: 판례일련번호, 구분, 문장번호, 내용
        texts = [r[3] for r in rows[1:] if len(r) >= 4 and r[3].strip()]
        return " ".join(texts)
    except Exception:
        return ""


def _format_answer(info: dict, label: dict) -> str:
    """AIHUB 라벨 → 구조화된 답변 텍스트."""
    case_name = info.get("caseName", "해당없음")
    answer    = label.get("output", "").strip()
    case_num  = info.get("caseNum", "")
    court     = info.get("courtName", "")

    # 법령 추출
    laws = _LAW_RE.findall(answer)
    if laws:
        law_str = "\n".join(f"{l}" for l in laws[:3])
    else:
        law_str = "해당없음"

    ref = f"{case_num} ({court})" if case_num and court else ""

    parts = [
        f"[죄명]: {case_name}",
        f"\n[답변]: {answer}",
        f"\n[근거법령]: {law_str}",
    ]
    if ref:
        parts.append(f"\n[참조판례]: {ref}")
    return "".join(parts)


def iter_examples(aihub: Path, split: str = "Training") -> Iterator[dict]:
    """Training/Validation QA → (system, user, assistant) 튜플."""
    prefix = "TL" if split == "Training" else "VL"
    folder = aihub / "형사데이터" / split / "02.라벨링데이터" / f"{prefix}_판결문_QA"
    if not folder.exists():
        return

    for jf in sorted(folder.glob("*.json")):
        try:
            d     = json.loads(jf.read_text(encoding="utf-8"))
            info  = d.get("info", {})
            label = d.get("label", {})

            case_name = info.get("caseName", "")
            prec_id   = str(info.get("precedId", ""))
            query     = label.get("input", "").strip()
            output    = label.get("output", "").strip()

            # 경제범죄 필터
            if not any(k in case_name for k in ECO_CRIMES):
                continue
            if not query or len(output) < 50:
                continue

            # 원문 텍스트 (최대 1200자)
            src_text = _load_source_text(aihub, prec_id)
            ctx = src_text[:1200] if src_text else "(판결문 원문 없음)"

            user_msg  = f"[판결문]\n{ctx}\n\n[질문]\n{query}"
            asst_msg  = _format_answer(info, label)

            yield {
                "messages": [
                    {"role": "system",    "content": SYSTEM},
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": asst_msg},
                ]
            }
        except Exception:
            continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", action="store_true", help="통계만 출력")
    ap.add_argument("--valid-ratio", default=0.1, type=float, help="검증셋 비율")
    args = ap.parse_args()

    print("Training QA 로드 중...")
    train_examples = list(iter_examples(AIHUB, "Training"))
    print(f"  경제범죄 Training QA: {len(train_examples):,}개")

    if args.stat:
        # 죄명 분포
        names = []
        for ex in train_examples:
            asst = ex["messages"][2]["content"]
            m = re.search(r"\[죄명\]:\s*(.+)", asst)
            if m:
                names.append(m.group(1).strip().split("(")[0].strip())
        c = Counter(names)
        print("\n죄명 분포 (상위 15):")
        for n, cnt in c.most_common(15):
            print(f"  {cnt:4d}건  {n}")
        return

    if not train_examples:
        print("ERROR: 데이터가 없습니다. AIHUB 경로를 확인하세요.")
        sys.exit(1)

    # train / valid 분할 (섞지 않고 앞 90% train, 뒤 10% valid)
    split_idx   = int(len(train_examples) * (1 - args.valid_ratio))
    train_split = train_examples[:split_idx]
    valid_split = train_examples[split_idx:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, data in [("train.jsonl", train_split), ("valid.jsonl", valid_split)]:
        out = OUT_DIR / fname
        with open(out, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"저장: {out}  ({len(data):,}건)")

    print(f"\n완료. data/lora_train/ 에 {len(train_examples):,}건 저장됨.")
    print("M2 맥미니로 RagInv 폴더 복사 후 아래 명령어 실행:")
    print()
    print("  pip install mlx-lm")
    print("  mlx_lm.lora \\")
    print("    --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \\")
    print("    --train \\")
    print("    --data data/lora_train \\")
    print("    --iters 1000 \\")
    print("    --batch-size 4 \\")
    print("    --lora-layers 16 \\")
    print("    --adapter-path adapters/llama_eco_crime")


if __name__ == "__main__":
    main()
