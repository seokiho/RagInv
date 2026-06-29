"""
데이터셋 통계 집계 스크립트 — 논문용

AIHUB 형사/민사 원천데이터에서 문서 수, 법령 수, 판결문 수 등을 집계합니다.
임베딩 전 원본 파일 기준으로 동작합니다.

Usage:
    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --aihub /path/to/AIHUB
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ── 경로 설정 ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
AIHUB = ROOT / "AIHUB"

CRIMINAL_SPLITS = {
    "Training":   "형사데이터/Training/01.원천데이터",
    "Validation": "형사데이터/Validation/01.원천데이터",
}
CIVIL_SPLITS = {
    "Training":   "민사데이터/Training/01.원천데이터",
    "Validation": "민사데이터/Validation/01.원천데이터",
}

# 카테고리 디렉터리 이름 → 표준 레이블
CRIMINAL_CAT_MAP = {
    "TS_판결문": "판결문", "VS_판결문": "판결문",
    "TS_법령":   "법령",   "VS_법령":   "법령",
    "TS_결정례": "결정례", "VS_결정례": "결정례",
    "TS_해석례": "해석례", "VS_해석례": "해석례",
}
CIVIL_CAT_MAP = {
    "TS_01. 민사법_001. 판결문": "판결문", "VS_01. 민사법_001. 판결문": "판결문",
    "TS_01. 민사법_002. 법령":   "법령",   "VS_01. 민사법_002. 법령":   "법령",
    "TS_01. 민사법_003. 심결례": "심결례", "VS_01. 민사법_003. 심결례": "심결례",
    "TS_01. 민사법_004. 유권해석": "유권해석", "VS_01. 민사법_004. 유권해석": "유권해석",
}

# 경제범죄 키워드 (형사 판결문 casename 필터용)
ECO_CRIME_KEYWORDS = ["사기", "횡령", "배임", "특정경제범죄", "특경"]


# ── 헬퍼 ──────────────────────────────────────────────────────────────────

def count_unique_in_csv(path: Path, id_col: str) -> int:
    try:
        df = pd.read_csv(path, usecols=[id_col])
        return df[id_col].nunique()
    except Exception:
        return 1  # 파일 1개 = 문서 1개로 fallage


def extract_criminal_case_meta(csv_path: Path) -> dict:
    """형사 판결문 CSV에서 메타 추출 (판례일련번호 기준 문서 1개)."""
    try:
        df = pd.read_csv(csv_path)
        # 구분 컬럼에서 참조조문 여부 등 확인
        sections = set(df["구분"].dropna().tolist()) if "구분" in df.columns else set()
        return {"sections": sections}
    except Exception:
        return {}


def extract_civil_case_meta(json_path: Path) -> dict:
    """민사 판결문 JSON에서 메타 추출."""
    try:
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
        return {
            "casetype":        d.get("casetype", ""),
            "casenames":       d.get("casenames", ""),
            "normalized_court": d.get("normalized_court", ""),
            "announce_date":   d.get("announce_date", "")[:4] if d.get("announce_date") else "",
        }
    except Exception:
        return {}


def is_eco_crime(casenames: str) -> bool:
    return any(kw in casenames for kw in ECO_CRIME_KEYWORDS)


# ── 형사 통계 ──────────────────────────────────────────────────────────────

def collect_criminal_stats(aihub: Path) -> dict:
    stats = {
        "총_파일수":        collections.Counter(),   # 카테고리별 (정규화된 레이블)
        "총_문서수":        collections.Counter(),   # 카테고리별 (unique ID 기준)
        "판결문_구분_분포": collections.Counter(),   # 판시사항/판결요지/판례내용/참조조문
        "법령_유니크_id":   set(),
    }

    for split, rel_path in CRIMINAL_SPLITS.items():
        base = aihub / rel_path
        if not base.exists():
            print(f"  [경고] 경로 없음: {base}")
            continue

        for cat_dir in sorted(base.iterdir()):
            label = CRIMINAL_CAT_MAP.get(cat_dir.name)
            if label is None:
                continue  # 알 수 없는 디렉터리 무시
            files = sorted(cat_dir.glob("*.csv"))
            stats["총_파일수"][label] += len(files)

            if label == "판결문":
                for f in tqdm(files, desc=f"형사/{split}/판결문", leave=False):
                    stats["총_문서수"]["판결문"] += 1  # CSV 1개 = 판결문 1건
                    try:
                        df = pd.read_csv(f, usecols=["구분"])
                        stats["판결문_구분_분포"].update(df["구분"].dropna().tolist())
                    except Exception:
                        pass

            elif label == "법령":
                for f in tqdm(files, desc=f"형사/{split}/법령", leave=False):
                    try:
                        df = pd.read_csv(f, usecols=["법령일련번호"])
                        for uid in df["법령일련번호"].dropna().unique():
                            stats["법령_유니크_id"].add(f"criminal_{uid}")
                    except Exception:
                        stats["법령_유니크_id"].add(str(f))
                stats["총_문서수"]["법령"] = len(stats["법령_유니크_id"])

            else:
                stats["총_문서수"][label] += len(files)

    return stats


# ── 민사 통계 ──────────────────────────────────────────────────────────────

def collect_civil_stats(aihub: Path) -> dict:
    stats = {
        "총_파일수":      collections.Counter(),
        "총_문서수":      collections.Counter(),
        "법원별":         collections.Counter(),
        "연도별_판결문":  collections.Counter(),
        "사건유형별":     collections.Counter(),
    }

    for split, rel_path in CIVIL_SPLITS.items():
        base = aihub / rel_path
        if not base.exists():
            print(f"  [경고] 경로 없음: {base}")
            continue

        for cat_dir in sorted(base.iterdir()):
            label = CIVIL_CAT_MAP.get(cat_dir.name)
            if label is None:
                continue
            json_files = list(cat_dir.glob("*.json"))
            csv_files  = list(cat_dir.glob("*.csv"))
            all_files  = json_files + csv_files

            stats["총_파일수"][label] += len(all_files)
            stats["총_문서수"][label] += len(all_files)

            if label == "판결문":
                # 샘플링으로 메타 수집 (전체 70k+ 파싱 대신 5000개 샘플)
                sample = json_files[:5000]
                for f in tqdm(sample, desc=f"민사/{split}/판결문(샘플)", leave=False):
                    meta = extract_civil_case_meta(f)
                    if meta.get("normalized_court"):
                        stats["법원별"][meta["normalized_court"]] += 1
                    if meta.get("announce_date"):
                        stats["연도별_판결문"][meta["announce_date"]] += 1
                    if meta.get("casetype"):
                        stats["사건유형별"][meta["casetype"]] += 1

    return stats


# ── 출력 ──────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_counter(c: dict | collections.Counter, top: int = 20, label: str = ""):
    if label:
        print(f"\n  [{label}]")
    for k, v in sorted(c.items(), key=lambda x: -x[1])[:top]:
        print(f"    {k:30s}: {v:>10,}")


def run(aihub: Path):
    print(f"\n[데이터셋 통계] AIHUB 경로: {aihub}\n")

    # ── 형사 ──
    print_section("형사 데이터 (Training + Validation 합산)")
    crim = collect_criminal_stats(aihub)

    print("\n  [파일 수 (카테고리별)]")
    for k, v in sorted(crim["총_파일수"].items()):
        print(f"    {k:10s}: {v:>8,} 파일")

    print("\n  [문서 수 (카테고리별, unique ID 기준)]")
    crim_total = 0
    for k, v in sorted(crim["총_문서수"].items()):
        print(f"    {k:10s}: {v:>8,} 건")
        crim_total += v
    print(f"    {'합계':10s}: {crim_total:>8,} 건")

    print("\n  [판결문 내 구분 유형 (구성 섹션)]")
    print_counter(crim["판결문_구분_분포"])

    # ── 민사 ──
    print_section("민사 데이터 (Training + Validation 합산)")
    civil = collect_civil_stats(aihub)

    print("\n  [파일 수 (카테고리별)]")
    for k, v in sorted(civil["총_파일수"].items()):
        print(f"    {k:10s}: {v:>8,} 파일")

    print("\n  [문서 수 (카테고리별)]")
    civil_total = 0
    for k, v in sorted(civil["총_문서수"].items()):
        print(f"    {k:10s}: {v:>8,} 건")
        civil_total += v
    print(f"    {'합계':10s}: {civil_total:>8,} 건")

    if civil["연도별_판결문"]:
        print("\n  [연도별 판결문 (샘플 기준, 상위 15년)]")
        print_counter(civil["연도별_판결문"], top=15)

    if civil["법원별"]:
        print("\n  [법원별 판결문 (샘플 기준, 상위 15)]")
        print_counter(civil["법원별"], top=15)

    # ── 전체 요약 ──
    print_section("전체 요약 (논문 Table용)")
    grand_total = crim_total + civil_total
    rows = [
        ("형사", "판결문", crim["총_문서수"].get("판결문", 0)),
        ("형사", "법령",   crim["총_문서수"].get("법령", 0)),
        ("형사", "결정례", crim["총_문서수"].get("결정례", 0)),
        ("형사", "해석례", crim["총_문서수"].get("해석례", 0)),
        ("민사", "판결문", civil["총_문서수"].get("판결문", 0)),
        ("민사", "법령",   civil["총_문서수"].get("법령", 0)),
        ("민사", "심결례", civil["총_문서수"].get("심결례", 0)),
        ("민사", "유권해석", civil["총_문서수"].get("유권해석", 0)),
    ]
    print(f"\n  {'도메인':6s} {'유형':8s} {'문서 수':>10s}")
    print(f"  {'-'*30}")
    for domain, dtype, cnt in rows:
        print(f"  {domain:6s} {dtype:8s} {cnt:>10,}")
    print(f"  {'-'*30}")
    print(f"  {'합계':15s} {grand_total:>10,}")

    # ── LoRA 학습데이터 ──
    print_section("경제범죄 LoRA 학습 데이터 (메모리 기준)")
    lora_rows = [
        ("사기",    815),
        ("횡령",    544),
        ("배임",    488),
        ("특경법",  460),
    ]
    lora_total = sum(v for _, v in lora_rows)
    for dtype, cnt in lora_rows:
        print(f"    {dtype:8s}: {cnt:>6,} 건")
    print(f"    {'합계':8s}: {lora_total:>6,} 건")


# ── 진입점 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIHUB 데이터셋 통계 집계 (논문용)")
    parser.add_argument("--aihub", type=Path, default=AIHUB, help="AIHUB 루트 경로")
    args = parser.parse_args()

    run(args.aihub)
