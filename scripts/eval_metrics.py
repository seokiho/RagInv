#!/usr/bin/env python3
"""
eval_metrics.py — RAG 검색·생성 평가 자동 채점

Usage:
    # 검색 평가 (임베딩 완료 후)
    .venv/bin/python3 scripts/eval_metrics.py retrieval \\
        --eval data/eval_set.jsonl \\
        --db data/db_aihub \\
        --top-k 5 10 20 \\
        --out results/retrieval_eval.json

    # 생성 평가 (LLM 추론 결과)
    .venv/bin/python3 scripts/eval_metrics.py generation \\
        --pred results/preds.jsonl \\
        --gold data/eval_set.jsonl \\
        --out results/generation_eval.json

    # 환각률 체크 (사건번호 실재 여부)
    .venv/bin/python3 scripts/eval_metrics.py hallucination \\
        --pred results/preds.jsonl \\
        --db data/db_aihub \\
        --out results/hallucination_eval.json

지표 정의:
  Retrieval  : Recall@K (gold source_id가 top-K 안에 있는지)
               MRR@K (mean reciprocal rank)
  Generation : 죄명 분류 정확도 / F1 (caseName 기준)
               법조 인용 정확도 (참조조문 포함 여부)
  Hallucination: 생성된 사건번호가 DB에 실재하는지 여부
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[eval] 저장 → {path}")


# 모델 출력에서 실제 죄명만 추출: "[죄명]: 사기 (부가설명...)" → "사기"
# 괄호 앞 첫 번째 토큰만 취함
_CRIME_EXTRACT_PAT = re.compile(
    r"(?:\[죄명\]\s*[:：]\s*|죄명\s*[:：]\s*|\*+죄명\*+\s*[:：]\s*)?"
    r"([^\(\（\n\*\[]{1,30})"
)


def extract_crime_name(raw: str) -> str:
    """
    모델 출력 pred_crime / pred_text 첫 줄에서 죄명 핵심어만 추출.
    "[죄명]: 사기 (핵심 죄명 하나. 판단 불가시 ...)" → "사기"
    """
    # 첫 줄만 사용
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    # [죄명]: 이후 또는 전체에서 괄호/설명 전 부분만 추출
    line = re.sub(r"\[죄명\]\s*[:：]?\s*", "", line)
    line = re.sub(r"\*+", "", line)
    # 괄호 앞까지
    line = re.split(r"[\(\（]", line)[0]
    # 쉼표·슬래시 앞까지 (경합범 첫 죄명만)
    line = re.split(r"[,，/]", line)[0]
    return line.strip()


def normalize_crime_name(name: str) -> str:
    """죄명 정규화: 공백/구두점 제거."""
    name = name.strip()
    name = re.sub(r"[\s\-\(\)（）·,、]", "", name)
    name = re.sub(r"죄$", "", name)   # '죄' 접미사 통일
    name = re.sub(r"등$", "", name)   # 경합범 표기 제거
    return name


def crime_name_f1(gold: str, pred: str) -> tuple[float, float, float]:
    """
    죄명 분류 F1 — 2음절 이상 서브시퀀스(n-gram) 기반.
    글자 단위 단순 집합은 "사기"↔"절도" 구분 불가, n-gram으로 보완.
    Returns (precision, recall, f1).
    """
    def ngrams(s: str, n: int = 2) -> set[str]:
        return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else set(s)

    gn = normalize_crime_name(gold)
    pn = normalize_crime_name(pred)
    if not gn or not pn:
        exact = float(gn == pn)
        return exact, exact, exact
    # bigram + unigram 혼합
    g_tokens = ngrams(gn, 2) | set(gn)
    p_tokens = ngrams(pn, 2) | set(pn)
    common = g_tokens & p_tokens
    prec = len(common) / len(p_tokens)
    rec  = len(common) / len(g_tokens)
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


_CJK_PAT = re.compile(r"[一-鿿㐀-䶿]")
_CJK_SENTENCE_PAT = re.compile(r"[一-鿿㐀-䶿]{5,}")


def has_language_error(text: str) -> bool:
    """한국어 응답에 CJK 한자(중국어/일본어 한자)가 포함되었는지 감지."""
    return bool(_CJK_PAT.search(text))


def extract_case_nums(text: str) -> list[str]:
    """
    텍스트에서 사건번호 패턴 추출.
    예: 2023노809, 2021도12345, 2019나2034956
    """
    pattern = r"\d{4}[가-힣]{1,3}\d{1,8}"
    return re.findall(pattern, text)


_LAW_ARTICLE_RE = re.compile(r"제\d+항.*$")  # 항/호 이하 제거용

def normalize_law(law: str) -> str:
    """법령 조항 정규화: 공백 제거."""
    return re.sub(r"\s+", "", law)

def normalize_law_article(law: str) -> str:
    """법령 조 단위 정규화: 공백 제거 + 항/호 제거 → '형법제347조' 수준."""
    s = re.sub(r"\s+", "", law)
    s = _LAW_ARTICLE_RE.sub("", s)
    return s


def law_citation_f1(
    gold_laws: list[str],
    pred_laws: list[str],
    mode: str = "prefix",
) -> tuple[float, float, float]:
    """
    법령 인용 F1.

    mode="prefix"  : 기존 방식 — 정규화 후 prefix 포함 관계면 매칭
                     (gold '형법제347조제1항' ↔ pred '형법제347조' 허용)
    mode="article" : 조(條) 단위 정규화 후 exact match
                     항/호 차이는 같은 조로 봄 (제347조제1항 == 제347조제2항 → same)
    mode="exact"   : 전체 일치 (항/호까지 정확히 일치해야 매칭)
    """
    if not gold_laws:
        return 0.0, 0.0, 0.0

    if mode == "article":
        g_norms = [normalize_law_article(g) for g in gold_laws]
        p_norms = [normalize_law_article(p) for p in pred_laws]
        def g_match(g): return g in p_norms
        def p_match(p): return p in g_norms
    elif mode == "exact":
        g_norms = [normalize_law(g) for g in gold_laws]
        p_norms = [normalize_law(p) for p in pred_laws]
        def g_match(g): return g in p_norms
        def p_match(p): return p in g_norms
    else:  # prefix (기존)
        g_norms = [normalize_law(g) for g in gold_laws]
        p_norms = [normalize_law(p) for p in pred_laws]
        def g_match(g): return any(p.startswith(g) or g.startswith(p) for p in p_norms)
        def p_match(p): return any(p.startswith(g) or g.startswith(p) for g in g_norms)

    tp_gold   = sum(1 for g in g_norms if g_match(g))
    recall    = tp_gold / len(g_norms)
    if not pred_laws:
        return 0.0, recall, 0.0
    tp_pred   = sum(1 for p in p_norms if p_match(p))
    precision = tp_pred / len(p_norms)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ── 1. 검색 평가 (Recall@K, MRR@K) ──────────────────────────────────────────

def run_retrieval_eval(eval_path: Path, db_path: Path, top_ks: list[int], out_path: Path,
                       jurisdictions: list[str] | None = None) -> dict:
    """
    leave-one-out 검색 평가.
    각 쿼리의 source_id를 제외하고 검색 → gold source_id가 top-K 안에 있는지.
    """
    from core.embedder import KoreanEmbedder
    from core.vectorstore import FAISSStore

    eval_records = load_jsonl(eval_path)
    if jurisdictions:
        eval_records = [r for r in eval_records if r.get("jurisdiction") in jurisdictions]
    print(f"[retrieval] 평가 쿼리: {len(eval_records):,}개")

    store = FAISSStore.load(str(db_path))
    embedder = KoreanEmbedder.get("BAAI/bge-m3")

    max_k = max(top_ks)
    metrics: dict[str, Any] = {k: {"recall": 0, "mrr": 0.0, "total": 0} for k in top_ks}
    by_jur: dict[str, dict] = defaultdict(lambda: {k: {"recall": 0, "mrr": 0.0, "total": 0} for k in top_ks})
    by_type: dict[str, dict] = defaultdict(lambda: {k: {"recall": 0, "mrr": 0.0, "total": 0} for k in top_ks})

    for i, rec in enumerate(eval_records):
        if i % 500 == 0:
            print(f"  {i}/{len(eval_records)}...", flush=True)

        query      = rec["query"]
        gold_id    = rec["source_id"]
        jur        = rec.get("jurisdiction", "unknown")
        doc_type   = rec.get("doc_type", "unknown")

        q_vec = embedder.embed_one(query)
        raw = store.search(q_vec, k=max_k)

        for k in top_ks:
            top = raw[:k]
            top_ids = [doc.get("source_id", "") for doc, _score, _idx in top]
            hit = gold_id in top_ids
            rank = top_ids.index(gold_id) + 1 if hit else 0
            rr = 1.0 / rank if rank > 0 else 0.0

            metrics[k]["recall"] += int(hit)
            metrics[k]["mrr"]    += rr
            metrics[k]["total"]  += 1
            by_jur[jur][k]["recall"] += int(hit)
            by_jur[jur][k]["mrr"]    += rr
            by_jur[jur][k]["total"]  += 1
            by_type[doc_type][k]["recall"] += int(hit)
            by_type[doc_type][k]["mrr"]    += rr
            by_type[doc_type][k]["total"]  += 1

    # 집계
    result = {"overall": {}, "by_jurisdiction": {}, "by_doc_type": {}}
    for k in top_ks:
        n = metrics[k]["total"]
        result["overall"][f"Recall@{k}"] = round(metrics[k]["recall"] / n, 4) if n else 0
        result["overall"][f"MRR@{k}"]   = round(metrics[k]["mrr"] / n, 4) if n else 0
    result["overall"]["n_queries"] = metrics[top_ks[0]]["total"]

    for jur, jdata in by_jur.items():
        result["by_jurisdiction"][jur] = {}
        for k in top_ks:
            n = jdata[k]["total"]
            result["by_jurisdiction"][jur][f"Recall@{k}"] = round(jdata[k]["recall"] / n, 4) if n else 0
            result["by_jurisdiction"][jur][f"MRR@{k}"]   = round(jdata[k]["mrr"] / n, 4) if n else 0

    for dt, dtdata in by_type.items():
        result["by_doc_type"][dt] = {}
        for k in top_ks:
            n = dtdata[k]["total"]
            result["by_doc_type"][dt][f"Recall@{k}"] = round(dtdata[k]["recall"] / n, 4) if n else 0

    _print_retrieval(result, top_ks)
    save_json(result, out_path)
    return result


def _print_retrieval(result: dict, top_ks: list[int]) -> None:
    print("\n[Retrieval Results]")
    print(f"  쿼리 수: {result['overall'].get('n_queries', 0):,}")
    for k in top_ks:
        r = result["overall"].get(f"Recall@{k}", 0)
        m = result["overall"].get(f"MRR@{k}", 0)
        print(f"  Recall@{k}: {r:.4f}   MRR@{k}: {m:.4f}")
    print()
    for jur, jdata in result.get("by_jurisdiction", {}).items():
        vals = "  ".join(f"R@{k}={jdata.get(f'Recall@{k}', 0):.3f}" for k in top_ks)
        print(f"  [{jur}] {vals}")


# ── 2. 생성 평가 (죄명 분류 정확도, 법조 인용) ────────────────────────────────

def run_generation_eval(pred_path: Path, gold_path: Path, out_path: Path) -> dict:
    """
    pred.jsonl 포맷:
      {"source_id": "...", "pred_crime": "사기", "pred_laws": ["형법 제347조"], "pred_text": "..."}
    gold = eval_criminal.jsonl (gold_name=죄명, gold_laws=법령 인용 목록)

    법령 인용 F1 세 가지 모드로 측정:
      prefix  : 기존 방식 (양방향 prefix 허용) — 관대한 기준
      article : 조(條) 단위 정규화 후 exact (항/호 차이 무시) — 중간 기준
      exact   : 전체 문자열 exact (항/호까지 일치) — 엄격한 기준
    """
    preds = {r["source_id"]: r for r in load_jsonl(pred_path)}
    golds = load_jsonl(gold_path)

    crime_correct  = 0
    crime_f1_sum   = 0.0
    crime_n        = 0   # doc_type=="case"인 건수만

    # 법령 F1 — 세 가지 기준
    law_stats: dict[str, dict] = {
        "prefix":  {"f1": 0.0, "prec": 0.0, "rec": 0.0, "n": 0},
        "article": {"f1": 0.0, "prec": 0.0, "rec": 0.0, "n": 0},
        "exact":   {"f1": 0.0, "prec": 0.0, "rec": 0.0, "n": 0},
    }
    # doc_type별 법령 F1 (article 기준)
    law_by_dtype: dict[str, dict] = defaultdict(
        lambda: {"f1": 0.0, "prec": 0.0, "rec": 0.0, "n": 0}
    )
    # 법령 인용 커버리지: gold 유무 관계없이 pred에 법령이 있는 건수
    law_coverage_n = 0   # 법령을 1개라도 인용한 건수 (전체 n 분모)

    lang_errors          = 0
    lang_errors_sentence = 0
    skipped              = 0

    by_jur: dict[str, dict] = defaultdict(
        lambda: {"correct": 0, "f1": 0.0, "crime_n": 0, "total": 0, "lang_errors": 0}
    )

    _law_re = re.compile(
        r"(?:"
        r"(?<![가-힣])[가-힣]{2,20}(?:법률|법|령|규칙)"
        r"|(?<![가-힣])(?:[가-힣]{1,10}\s){0,3}[가-힣]{1,10}에\s+관한\s+법률"
        r")"
        r"\s*제\s*\d+\s*조(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?"
    )

    for g in golds:
        sid = g["source_id"]
        if sid not in preds:
            skipped += 1
            continue
        p        = preds[sid]
        jur      = g.get("jurisdiction", "unknown")
        doc_type = g.get("doc_type", "unknown")

        # ── 언어 오류 감지 ─────────────────────────────────────────────────
        pred_text = p.get("pred_text", "")
        lang_err = has_language_error(pred_text)
        if lang_err:
            lang_errors += 1
            by_jur[jur]["lang_errors"] += 1
            if _CJK_SENTENCE_PAT.search(pred_text):
                lang_errors_sentence += 1

        by_jur[jur]["total"] += 1

        # ── 죄명 정확도 — 판결문(case)만 ──────────────────────────────────
        if doc_type == "case":
            gold_crime = g.get("gold_name", "")
            pred_crime = extract_crime_name(p.get("pred_crime", "") or pred_text)
            exact = int(normalize_crime_name(gold_crime) == normalize_crime_name(pred_crime))
            _, _, f1 = crime_name_f1(gold_crime, pred_crime)
            crime_correct += exact
            crime_f1_sum  += f1
            crime_n       += 1
            by_jur[jur]["correct"] += exact
            by_jur[jur]["f1"]      += f1
            by_jur[jur]["crime_n"] += 1

        # ── 법령 인용 커버리지 (gold 유무 무관) ───────────────────────────
        pred_laws_from_text = _law_re.findall(pred_text)
        pred_laws = p.get("pred_laws") or pred_laws_from_text
        if pred_laws:
            law_coverage_n += 1

        # ── 법령 F1 — gold 있을 때만 ─────────────────────────────────────
        gold_laws = g.get("gold_laws", [])
        if gold_laws:
            for mode in ("prefix", "article", "exact"):
                prec, rec, f1 = law_citation_f1(gold_laws, pred_laws, mode=mode)
                law_stats[mode]["f1"]   += f1
                law_stats[mode]["prec"] += prec
                law_stats[mode]["rec"]  += rec
                law_stats[mode]["n"]    += 1
            # doc_type별 (article 기준)
            prec, rec, f1 = law_citation_f1(gold_laws, pred_laws, mode="article")
            law_by_dtype[doc_type]["f1"]   += f1
            law_by_dtype[doc_type]["prec"] += prec
            law_by_dtype[doc_type]["rec"]  += rec
            law_by_dtype[doc_type]["n"]    += 1

    n = len(golds) - skipped

    def avg(key, mode):
        nn = law_stats[mode]["n"]
        return round(law_stats[mode][key] / nn, 4) if nn else None

    result = {
        "n_evaluated":  n,
        "n_skipped":    skipped,
        # 죄명 분류: 판결문(case)만
        "crime_n":        crime_n,
        "crime_accuracy": round(crime_correct / crime_n, 4) if crime_n else 0,
        "crime_f1":       round(crime_f1_sum / crime_n, 4)  if crime_n else 0,
        # 법령 인용 커버리지
        "law_coverage_n":    law_coverage_n,
        "law_coverage_rate": round(law_coverage_n / n, 4) if n else 0,
        # 법령 F1 — 세 가지 기준
        "law_f1_gold_n": law_stats["prefix"]["n"],  # gold_laws 있는 건수
        "law_f1_prefix":  {"f1": avg("f1","prefix"),  "prec": avg("prec","prefix"),  "rec": avg("rec","prefix")},
        "law_f1_article": {"f1": avg("f1","article"), "prec": avg("prec","article"), "rec": avg("rec","article")},
        "law_f1_exact":   {"f1": avg("f1","exact"),   "prec": avg("prec","exact"),   "rec": avg("rec","exact")},
        # doc_type별 법령 F1 (article 기준)
        "law_f1_by_doctype": {
            dt: {
                "f1":   round(v["f1"]   / v["n"], 4) if v["n"] else None,
                "prec": round(v["prec"] / v["n"], 4) if v["n"] else None,
                "rec":  round(v["rec"]  / v["n"], 4) if v["n"] else None,
                "n":    v["n"],
            }
            for dt, v in law_by_dtype.items()
        },
        # 언어 오류
        "language_error_n":          lang_errors,
        "language_error_rate":       round(lang_errors / n, 4) if n else 0,
        "language_error_sentence_n": lang_errors_sentence,
        "by_jurisdiction": {},
    }

    for jur, jdata in by_jur.items():
        nt = jdata["total"]
        cn = jdata["crime_n"]
        result["by_jurisdiction"][jur] = {
            "accuracy":            round(jdata["correct"] / cn, 4) if cn else None,
            "f1":                  round(jdata["f1"] / cn, 4)      if cn else None,
            "crime_n":             cn,
            "language_error_n":    jdata["lang_errors"],
            "language_error_rate": round(jdata["lang_errors"] / nt, 4) if nt else 0,
            "n":                   nt,
        }

    # 출력
    lp = result["law_f1_prefix"]
    la = result["law_f1_article"]
    le = result["law_f1_exact"]
    law_n = result["law_f1_gold_n"]
    print("\n[Generation Results]")
    print(f"  평가: {n}개 (판결문 {crime_n}건 / 결정·해석례 {n-crime_n}건) / 스킵: {skipped}개")
    print(f"  죄명 정확도: {result['crime_accuracy']:.4f}  (판결문 {crime_n}건 기준)")
    print(f"  죄명 F1:     {result['crime_f1']:.4f}")
    print(f"  법령 인용 커버리지: {result['law_coverage_rate']:.4f}  ({law_coverage_n}/{n}건 법령 인용)")
    print(f"  법령 F1 평가 대상: {law_n}건 (gold_laws 있는 건)")
    if lp["f1"] is not None:
        print(f"  ┌ [prefix / 관대]  F1={lp['f1']:.4f}  P={lp['prec']:.4f}  R={lp['rec']:.4f}")
        print(f"  ├ [article/ 조단위] F1={la['f1']:.4f}  P={la['prec']:.4f}  R={la['rec']:.4f}")
        print(f"  └ [exact  / 엄격]  F1={le['f1']:.4f}  P={le['prec']:.4f}  R={le['rec']:.4f}")
        print(f"  법령 F1 by doc_type (article 기준):")
        for dt, dv in result["law_f1_by_doctype"].items():
            if dv["f1"] is not None:
                print(f"    [{dt}] F1={dv['f1']:.4f}  n={dv['n']}")
    print(f"  언어 오류율: {result['language_error_rate']:.4f}"
          f"  ({lang_errors}건 / 중국어 문장: {lang_errors_sentence}건)")
    for jur, jd in result["by_jurisdiction"].items():
        acc_str = f"{jd['accuracy']:.3f}" if jd["accuracy"] is not None else "N/A"
        f1_str  = f"{jd['f1']:.3f}"       if jd["f1"]       is not None else "N/A"
        print(f"  [{jur}] 죄명정확도={acc_str}  죄명F1={f1_str}"
              f"  언어오류={jd['language_error_n']}건  n={jd['n']}")

    save_json(result, out_path)
    return result


# ── 3. 환각률 평가 (사건번호 실재 여부) ──────────────────────────────────────

def run_hallucination_eval(pred_path: Path, db_path: Path, out_path: Path) -> dict:
    """
    pred_text에서 사건번호 패턴을 추출, DB의 gold_num 목록과 대조.
    실재하지 않는 번호를 생성했으면 환각으로 간주.
    """
    from core.vectorstore import FAISSStore

    preds = load_jsonl(pred_path)
    store = FAISSStore.load(str(db_path))

    # DB에 있는 모든 사건번호 수집
    # 민사 문서는 case_num 필드가 없고 source_id에 "법원-사건번호" 형태로 포함
    known_nums: set[str] = set()
    _case_pat = re.compile(r"\d{4}[가-힣]{1,3}\d{1,8}")
    for doc in store.documents:
        num = doc.get("case_num") or doc.get("gold_num", "")
        if num:
            known_nums.add(num.strip())
        sid = doc.get("source_id", "")
        for m in _case_pat.findall(sid):
            known_nums.add(m)

    # 사건번호 단위 카운트 (기존 — 참고용)
    total_nums_generated = 0
    hallucinated_nums    = 0

    # 문서 단위 카운트 (주 지표)
    docs_with_any_num        = 0
    docs_with_halluc         = 0
    docs_no_nums             = 0
    halluc_details           = []

    for p in preds:
        text = p.get("pred_text", "")
        found = extract_case_nums(text)

        if not found:
            docs_no_nums += 1
            continue

        docs_with_any_num += 1
        doc_halluc = False
        for num in found:
            total_nums_generated += 1
            if num not in known_nums:
                hallucinated_nums += 1
                doc_halluc = True
                halluc_details.append({"source_id": p.get("source_id"), "hallucinated_num": num})

        if doc_halluc:
            docs_with_halluc += 1

    n_docs = len(preds)
    # 문서 단위 환각률: 전체 문서 기준 / 사건번호를 생성한 문서 기준 두 가지 모두 제공
    doc_rate_all   = round(docs_with_halluc / n_docs, 4) if n_docs else 0.0
    doc_rate_cited = round(docs_with_halluc / docs_with_any_num, 4) if docs_with_any_num else 0.0
    num_rate       = round(hallucinated_nums / total_nums_generated, 4) if total_nums_generated else 0.0

    result = {
        # 주 지표: 문서 단위
        "n_docs":                          n_docs,
        "docs_citing_case_nums":           docs_with_any_num,
        "docs_no_case_nums":               docs_no_nums,
        "docs_with_hallucination":         docs_with_halluc,
        "hallucination_rate_per_doc":      doc_rate_all,    # 전체 문서 분모
        "hallucination_rate_per_cited_doc": doc_rate_cited, # 사건번호 인용한 문서 분모
        # 참고: 사건번호 단위 (기존 방식)
        "total_case_nums_generated":       total_nums_generated,
        "hallucinated_case_nums":          hallucinated_nums,
        "hallucination_rate_per_num":      num_rate,
        "examples":                        halluc_details[:20],
    }
    print(f"\n[Hallucination]")
    print(f"  문서 단위 환각률 (전체 기준):      {doc_rate_all:.2%}  ({docs_with_halluc}/{n_docs}건)")
    print(f"  문서 단위 환각률 (인용 문서 기준): {doc_rate_cited:.2%}  ({docs_with_halluc}/{docs_with_any_num}건)")
    print(f"  사건번호 단위 환각률 (참고):       {num_rate:.2%}  ({hallucinated_nums}/{total_nums_generated}개)")
    save_json(result, out_path)
    return result


# ── 4. 법령 조항 환각 평가 (law_db.json 기반) ────────────────────────────────

def run_law_hallucination_eval(pred_path: Path, law_db_path: Path, out_path: Path) -> dict:
    """
    pred_text에서 인용한 법령 조항을 추출, law_db.json과 대조.

    law_db.json은 39개 경제범죄 관련 법령 수록.
    해당 법령을 인용한 케이스에 한해 조항 실재 여부 검증.
    나머지 법령(마약류관리법, 공직선거법 등)은 DB 미수록 → 검증 불가.
    """
    with open(law_db_path, encoding="utf-8") as f:
        db_raw = json.load(f)

    # 법령명 → 실재 조문번호 집합
    law_names_in_db: set[str] = set(v["law_name"] for v in db_raw.values())
    valid_articles: dict[str, set[str]] = defaultdict(set)
    for v in db_raw.values():
        valid_articles[v["law_name"]].add(str(v["article_num"]))

    preds = load_jsonl(pred_path)

    _law_name_art_re = re.compile(
        r"((?:[가-힣]{1,20}(?:\s[가-힣]{1,10}){0,3})\s*(?:법률|법|령|규칙))"
        r"\s*제\s*(\d+)\s*조"
    )

    total_citations  = 0
    verifiable_n     = 0   # law_db에 있는 법령 인용 건수
    confirmed_real_n = 0   # 조항이 실재
    halluc_art_n     = 0   # 조항이 없음 (법령 이름은 실재, 조항 번호가 없음)
    halluc_details   = []

    for p in preds:
        pred_txt = p.get("pred_text", "")
        for m in _law_name_art_re.finditer(pred_txt):
            law_name = m.group(1).strip()
            art_num  = m.group(2)
            total_citations += 1

            if law_name in law_names_in_db:
                verifiable_n += 1
                if art_num in valid_articles[law_name]:
                    confirmed_real_n += 1
                else:
                    halluc_art_n += 1
                    halluc_details.append({
                        "source_id": p.get("source_id"),
                        "law_name":  law_name,
                        "article":   art_num,
                        "cited":     m.group(0),
                    })

    result = {
        "total_law_citations":       total_citations,
        "verifiable_n":              verifiable_n,
        "verifiable_rate":           round(verifiable_n / total_citations, 4) if total_citations else 0,
        "confirmed_real_n":          confirmed_real_n,
        "hallucinated_article_n":    halluc_art_n,
        "hallucinated_article_rate": round(halluc_art_n / verifiable_n, 4) if verifiable_n else 0,
        "law_db_laws":               len(law_names_in_db),
        "examples":                  halluc_details[:20],
    }

    print(f"\n[Law Hallucination (law_db 기반)]")
    print(f"  전체 법령 인용:          {total_citations}건")
    print(f"  law_db 검증 가능:        {verifiable_n}건 ({result['verifiable_rate']:.1%})  "
          f"(law_db: {len(law_names_in_db)}개 법령)")
    print(f"  검증 가능 중 실존 조항:  {confirmed_real_n}건")
    print(f"  검증 가능 중 없는 조항:  {halluc_art_n}건 ({result['hallucinated_article_rate']:.1%})")
    if halluc_details:
        print(f"  예시: {halluc_details[0]['law_name']} 제{halluc_details[0]['article']}조")

    save_json(result, out_path)
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="RAG 평가 자동 채점")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # retrieval
    r = sub.add_parser("retrieval", help="검색 평가 (Recall@K, MRR@K)")
    r.add_argument("--eval",  default="data/eval_set.jsonl", help="평가셋")
    r.add_argument("--db",    default="data/db_aihub",       help="FAISS DB 디렉토리")
    r.add_argument("--top-k", nargs="+", type=int, default=[5, 10, 20])
    r.add_argument("--jur",   nargs="*", help="jurisdiction 필터 (criminal/civil)")
    r.add_argument("--out",   default="results/retrieval_eval.json")

    # generation
    g = sub.add_parser("generation", help="생성 평가 (죄명·법조)")
    g.add_argument("--pred", required=True, help="예측 결과 jsonl")
    g.add_argument("--gold", default="data/eval_set.jsonl")
    g.add_argument("--out",  default="results/generation_eval.json")

    # hallucination (사건번호 기반)
    h = sub.add_parser("hallucination", help="환각률 평가 (사건번호 실재 여부)")
    h.add_argument("--pred", required=True, help="예측 결과 jsonl")
    h.add_argument("--db",   default="data/db_aihub")
    h.add_argument("--out",  default="results/hallucination_eval.json")

    # law-hallucination (law_db.json 기반 법령 조항 환각)
    lh = sub.add_parser("law-hallucination", help="법령 조항 환각 평가 (law_db.json 기반)")
    lh.add_argument("--pred",   required=True, help="예측 결과 jsonl")
    lh.add_argument("--law-db", default="data/law_db.json", help="law_db.json 경로")
    lh.add_argument("--out",    default="results/law_hallucination_eval.json")

    args = ap.parse_args()

    if args.cmd == "retrieval":
        run_retrieval_eval(
            eval_path=PROJECT_ROOT / args.eval,
            db_path=PROJECT_ROOT / args.db,
            top_ks=args.top_k,
            out_path=PROJECT_ROOT / args.out,
            jurisdictions=args.jur,
        )
    elif args.cmd == "generation":
        run_generation_eval(
            pred_path=PROJECT_ROOT / args.pred,
            gold_path=PROJECT_ROOT / args.gold,
            out_path=PROJECT_ROOT / args.out,
        )
    elif args.cmd == "hallucination":
        run_hallucination_eval(
            pred_path=PROJECT_ROOT / args.pred,
            db_path=PROJECT_ROOT / args.db,
            out_path=PROJECT_ROOT / args.out,
        )
    elif args.cmd == "law-hallucination":
        run_law_hallucination_eval(
            pred_path=PROJECT_ROOT / args.pred,
            law_db_path=PROJECT_ROOT / args.law_db,
            out_path=PROJECT_ROOT / args.out,
        )


if __name__ == "__main__":
    main()
