#!/usr/bin/env python3
"""
results/ 폴더의 generation_*.json + hallucination_*.json 을 읽어
논문용 비교표를 출력한다.

Usage:
    .venv/bin/python3 scripts/summary_results.py
    .venv/bin/python3 scripts/summary_results.py --format md
    .venv/bin/python3 scripts/summary_results.py --format csv > results/summary.csv
    .venv/bin/python3 scripts/summary_results.py --db db_sectional
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results"

_GEN_RE  = re.compile(r"^generation_(.+?)_(db_\w+)\.json$")
_HALL_RE = re.compile(r"^hallucination_(.+?)_(db_\w+)\.json$")

MODEL_LABELS = {
    "qwen2.5_7b":     "Qwen2.5-7B",
    "llama3.1_8b":    "LLaMA3.1-8B",
    "exaone3.5_7.8b": "EXAONE3.5-7.8B",
    "qwen3_32b":      "Qwen3-32B",
    "exaone4.5_33b":  "EXAONE4.5-33B",
}

COND_LABELS = {
    "standard": "RAG (standard)",
    "zeroshot": "zero-shot",
    "fewshot":  "few-shot",
    "cross":    "cross-DB",
    "lora":     "LoRA",
}

COND_ORDER  = ["standard", "zeroshot", "fewshot", "cross", "lora"]
MODEL_ORDER = ["qwen2.5_7b", "llama3.1_8b", "exaone3.5_7.8b", "qwen3_32b", "exaone4.5_33b"]


def parse_filename(fname: str) -> tuple[str, str, str] | None:
    if fname in ("generation_eval.json", "hallucination_eval.json"):
        return None
    m = _GEN_RE.match(fname) or _HALL_RE.match(fname)
    if not m:
        return None
    body, db = m.group(1), m.group(2)
    for cond in ["zeroshot", "fewshot", "cross", "lora"]:
        if body.endswith(f"_{cond}"):
            return body[: -(len(cond) + 1)], cond, db
    return body, "standard", db


def _get_law_f1(g: dict) -> float | None:
    """generation JSON에서 법령 F1(article 기준, 주 지표) 추출. 구/신 포맷 모두 처리."""
    # 신 포맷: law_f1_article dict
    art = g.get("law_f1_article")
    if isinstance(art, dict) and art.get("f1") is not None:
        return art["f1"]
    # 구 포맷: law_citation_f1 float
    old = g.get("law_citation_f1")
    if old is not None:
        return old
    return None


def _get_law_f1_exact(g: dict) -> float | None:
    ext = g.get("law_f1_exact")
    if isinstance(ext, dict):
        return ext.get("f1")
    return None


def load_results() -> dict:
    """{ (model, cond, db): {law_f1, law_f1_exact, halluc_rate, lang_err, n} }"""
    data: dict[tuple, dict] = {}

    for f in sorted(RESULTS_DIR.glob("generation_*.json")):
        parsed = parse_filename(f.name)
        if not parsed:
            continue
        model, cond, db = parsed
        with open(f) as fp:
            g = json.load(fp)
        key = (model, cond, db)
        data.setdefault(key, {})
        data[key].update({
            "law_f1":       _get_law_f1(g),
            "law_f1_exact": _get_law_f1_exact(g),
            "lang_err":     g.get("language_error_rate"),
            "n":            g.get("n_evaluated"),
        })

    for f in sorted(RESULTS_DIR.glob("hallucination_*.json")):
        parsed = parse_filename(f.name)
        if not parsed:
            continue
        model, cond, db = parsed
        with open(f) as fp:
            h = json.load(fp)
        key = (model, cond, db)
        data.setdefault(key, {})
        # 신 포맷: hallucination_rate_per_doc / 구 포맷: hallucination_rate
        rate = h.get("hallucination_rate_per_doc") or h.get("hallucination_rate")
        data[key]["halluc_rate"] = rate

    return data


def fmt(v, pct=False):
    if v is None:
        return "—"
    return f"{v*100:.1f}%" if pct else f"{v:.4f}"


def print_table(data: dict, fmt_type: str, db_filter: str | None = None):
    dbs = sorted({k[2] for k in data})
    if db_filter:
        dbs = [d for d in dbs if d == db_filter]

    for db in dbs:
        rows = []
        for model in MODEL_ORDER:
            for cond in COND_ORDER:
                key = (model, cond, db)
                if key not in data:
                    continue
                rows.append((model, cond, data[key]))

        if not rows:
            continue

        if fmt_type == "md":
            print(f"\n### {db}\n")
            print("| 모델 | 조건 | 법령F1(조단위) | 법령F1(엄격) | 환각률 | 언어오류율 | N |")
            print("|------|------|:------------:|:----------:|:------:|:--------:|---|")
            for model, cond, d in rows:
                note = " ⚠" if (d.get("n") or 0) < 1563 else ""
                print(f"| {MODEL_LABELS.get(model, model)} "
                      f"| {COND_LABELS.get(cond, cond)} "
                      f"| {fmt(d.get('law_f1'))} "
                      f"| {fmt(d.get('law_f1_exact'))} "
                      f"| {fmt(d.get('halluc_rate'), pct=True)} "
                      f"| {fmt(d.get('lang_err'), pct=True)} "
                      f"| {d.get('n','—')}{note} |")

        elif fmt_type == "csv":
            print("db,model,condition,law_f1_article,law_f1_exact,hallucination_rate,language_error_rate,n")
            for model, cond, d in rows:
                print(f"{db},{MODEL_LABELS.get(model,model)},{COND_LABELS.get(cond,cond)},"
                      f"{d.get('law_f1','')},{d.get('law_f1_exact','')},{d.get('halluc_rate','')},{d.get('lang_err','')},{d.get('n','')}")

        else:  # plain text
            print(f"\n{'='*78}")
            print(f"  [ {db} ]")
            print(f"{'='*78}")
            print(f"{'모델':<20} {'조건':<12} {'법령F1(조단위)':>13} {'법령F1(엄격)':>12} {'환각률':>8} {'언어오류':>8} {'N':>6}")
            print("-" * 78)
            for model, cond, d in rows:
                ml   = MODEL_LABELS.get(model, model)
                cl   = COND_LABELS.get(cond, cond)
                note = " ⚠" if (d.get("n") or 0) < 1563 else ""
                print(f"{ml:<20} {cl:<12} "
                      f"{fmt(d.get('law_f1')):>13} "
                      f"{fmt(d.get('law_f1_exact')):>12} "
                      f"{fmt(d.get('halluc_rate'), pct=True):>8} "
                      f"{fmt(d.get('lang_err'), pct=True):>8} "
                      f"{str(d.get('n','—'))+note:>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["text", "md", "csv"], default="text")
    ap.add_argument("--db",     default=None, help="db_aihub / db_sectional 필터")
    args = ap.parse_args()

    data = load_results()
    if not data:
        print("결과 파일 없음. results/ 폴더를 확인하세요.")
        return

    print_table(data, args.format, db_filter=args.db)

    if args.format == "text":
        valid = {k: v for k, v in data.items() if v.get("law_f1") is not None}
        if not valid:
            return
        print(f"\n{'='*78}")
        print(f"  요약: {len(data)}개 결과 로드됨")

        best = max(valid.items(), key=lambda x: x[1]["law_f1"])
        bk, bd = best
        print(f"  법령F1 최고: {MODEL_LABELS.get(bk[0],bk[0])} / {COND_LABELS.get(bk[1],bk[1])} / {bk[2]} → {fmt(bd['law_f1'])}")

        hall_valid = {k: v for k, v in data.items() if v.get("halluc_rate") is not None}
        if hall_valid:
            lowest = min(hall_valid.items(), key=lambda x: x[1]["halluc_rate"])
            hk, hd = lowest
            print(f"  환각률 최저: {MODEL_LABELS.get(hk[0],hk[0])} / {COND_LABELS.get(hk[1],hk[1])} / {hk[2]} → {fmt(hd['halluc_rate'], pct=True)}")

        lang_valid = {k: v for k, v in data.items() if v.get("lang_err") is not None}
        if lang_valid:
            worst_lang = max(lang_valid.items(), key=lambda x: x[1]["lang_err"])
            wk, wd = worst_lang
            print(f"  언어오류 최고: {MODEL_LABELS.get(wk[0],wk[0])} / {COND_LABELS.get(wk[1],wk[1])} / {wk[2]} → {fmt(wd['lang_err'], pct=True)}")


if __name__ == "__main__":
    main()
