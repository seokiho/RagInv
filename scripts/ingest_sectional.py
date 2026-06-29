#!/usr/bin/env python3
"""
섹션 인식 청킹 — 판결문 구조(주문/범죄사실/법령의 적용/…)를 인식하여
섹션 단위로 청킹하고 section_type 메타데이터를 부착한다.

기존 ingest_aihub.py 와 비교 실험 (A vs B):
  A: data/db_aihub/     ← 슬라이딩 윈도우 1500자
  B: data/db_sectional/ ← 섹션 인식 청킹 (이 스크립트)

Usage:
    .venv/bin/python3 scripts/ingest_sectional.py --dry-run
    .venv/bin/python3 scripts/ingest_sectional.py
    .venv/bin/python3 scripts/ingest_sectional.py --jurisdiction criminal
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.embedder import KoreanEmbedder
from core.vectorstore import FAISSStore

MAX_CHARS = 1200   # 섹션 청크 최대 길이 (기존 1500보다 작게 — 섹션이 더 응집돼 있음)

# ── 섹션 인식 ─────────────────────────────────────────────────────────────────

# (정규식, section_type 레이블) — 판결문에서 자주 나오는 섹션 헤더
_SECTION_MAP: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'주\s*문'),                           "주문"),
    (re.compile(r'범\s*죄\s*사\s*실|공\s*소\s*사\s*실'), "범죄사실"),
    (re.compile(r'법\s*령\s*의?\s*적\s*용|적\s*용\s*법\s*조'), "법령의_적용"),
    (re.compile(r'증\s*거\s*의?\s*요\s*지|증\s*거\s*목\s*록'), "증거요지"),
    (re.compile(r'양\s*형'),                           "양형이유"),
    (re.compile(r'이\s*유'),                           "이유"),
    (re.compile(r'항\s*소\s*이\s*유'),                 "항소이유"),
    (re.compile(r'상\s*고\s*이\s*유'),                 "상고이유"),
]


def _detect_section(line: str) -> str | None:
    """짧은 줄에서 섹션 헤더를 감지. 헤더면 section_type 반환, 아니면 None."""
    stripped = re.sub(r'\s+', '', line.strip())
    # 헤더는 보통 짧다 (20자 이하). 긴 줄은 헤더가 아님.
    if not stripped or len(stripped) > 20:
        return None
    for pattern, name in _SECTION_MAP:
        if pattern.search(stripped):
            return name
    return None


def _split_into_sections(lines: List[str]) -> List[Tuple[str, List[str]]]:
    """줄 목록 → [(section_type, [줄, ...]), ...] 로 그룹화."""
    sections: List[Tuple[str, List[str]]] = []
    cur_type  = "기타"
    cur_lines: List[str] = []

    for line in lines:
        sec = _detect_section(line)
        if sec is not None:
            if cur_lines:
                sections.append((cur_type, cur_lines))
            cur_type  = sec
            cur_lines = []
        else:
            if line.strip():
                cur_lines.append(line.strip())

    if cur_lines:
        sections.append((cur_type, cur_lines))

    return sections if sections else [("기타", lines)]


def _lines_to_chunks(lines: List[str], max_chars: int = MAX_CHARS) -> List[str]:
    """줄 목록을 max_chars 이내로 묶어 청크 리스트 반환. 줄 중간에서 자르지 않음."""
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for line in lines:
        add_len = len(line) + 1  # +1 은 개행
        if current and current_len + add_len > max_chars:
            chunks.append("\n".join(current))
            current     = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += add_len

    if current:
        chunks.append("\n".join(current))

    return [c for c in chunks if c.strip()]


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def doc(text: str, **meta: Any) -> Dict[str, Any]:
    return {"text": text, **meta}


# ── 형사 라벨 메타 로드 (ingest_aihub.py 와 동일) ────────────────────────────

def load_criminal_meta(aihub: Path) -> Dict[str, Dict]:
    id_field_map = {
        "TL_판결문_QA":  "precedId",    "VL_판결문_QA":  "precedId",
        "TL_판결문_SUM": "precedId",    "VL_판결문_SUM": "precedId",
        "TL_법령_QA":    "lawId",       "VL_법령_QA":    "lawId",
        "TL_결정례_QA":  "determintId", "VL_결정례_QA":  "determintId",
        "TL_결정례_SUM": "determintId", "VL_결정례_SUM": "determintId",
        "TL_해석례_QA":  "interpreId",  "VL_해석례_QA":  "interpreId",
        "TL_해석례_SUM": "interpreId",  "VL_해석례_SUM": "interpreId",
    }
    meta: Dict[str, Dict] = {}
    for split_dir in ("Training", "Validation"):
        label_root = aihub / "형사데이터" / split_dir / "02.라벨링데이터"
        if not label_root.exists():
            continue
        for folder_name, id_field in id_field_map.items():
            folder = label_root / folder_name
            if not folder.exists():
                continue
            for jf in folder.glob("*.json"):
                try:
                    d   = json.loads(jf.read_text(encoding="utf-8"))
                    info = d.get("info", {})
                    sid  = str(info.get(id_field, "")).strip()
                    if sid and sid not in meta:
                        meta[sid] = info
                except Exception:
                    pass
    print(f"  → 형사 메타 {len(meta):,}건 로드")
    return meta


# ── 형사 CSV 폴더 헬퍼 ────────────────────────────────────────────────────────

def _criminal_csv_folders(aihub: Path, sub: str):
    for split_name, split_dir, prefix in [
        ("train", "Training",   "TS"),
        ("val",   "Validation", "VS"),
    ]:
        folder = aihub / "형사데이터" / split_dir / "01.원천데이터" / f"{prefix}_{sub}"
        if folder.exists():
            yield split_name, folder


# ── 형사 판결문: 섹션 인식 청킹 (핵심 변경) ──────────────────────────────────

def parse_criminal_cases(aihub: Path, meta: Dict) -> Iterator[Dict]:
    """형사 판결문 CSV → 섹션 단위 청크. 각 청크에 section_type 메타 부착."""
    for split, folder in _criminal_csv_folders(aihub, "판결문"):
        for csv_path in tqdm(sorted(folder.glob("*.csv")),
                             desc=f"형사판결문/{split}", leave=False):
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception:
                continue
            if "판례일련번호" not in df.columns:
                continue

            for prec_id, grp in df.groupby("판례일련번호"):
                lines = (
                    grp.sort_values("문장번호")["내용"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )
                if not lines:
                    continue

                m    = meta.get(str(prec_id), {})
                base = dict(
                    doc_type="case", jurisdiction="criminal", split=split,
                    source_id=str(prec_id),
                    case_name=m.get("caseName", ""),
                    case_num=m.get("caseNum", ""),
                    court=m.get("courtName", ""),
                    date=m.get("sentenceDate", ""),
                )

                sections = _split_into_sections(lines)
                for sec_type, sec_lines in sections:
                    for chunk_text in _lines_to_chunks(sec_lines):
                        yield doc(chunk_text, section_type=sec_type, **base)


# ── 형사 법령/결정례/해석례: 기존 로직 유지 ──────────────────────────────────

def parse_criminal_laws(aihub: Path, meta: Dict) -> Iterator[Dict]:
    for split, folder in _criminal_csv_folders(aihub, "법령"):
        for csv_path in tqdm(sorted(folder.glob("*.csv")),
                             desc=f"형사법령/{split}", leave=False):
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception:
                continue
            if "법령일련번호" not in df.columns:
                continue
            law_id   = str(df["법령일련번호"].iloc[0])
            m        = meta.get(law_id, {})
            law_name = m.get("title", csv_path.stem)

            cur_lines: List[str] = []
            cur_section = ""
            for _, row in df.iterrows():
                section = str(row.get("구분", "")).strip()
                content = str(row.get("내용", "")).strip()
                if not content:
                    continue
                if section == "조문" and cur_lines:
                    for chunk in _lines_to_chunks(cur_lines):
                        yield doc(chunk, doc_type="law", jurisdiction="criminal",
                                  split=split, source_id=law_id, law_name=law_name,
                                  section_type=cur_section,
                                  date=m.get("effectDate", ""))
                    cur_lines, cur_section = [], content
                else:
                    if section == "조문":
                        cur_section = content
                    cur_lines.append(content)
            if cur_lines:
                for chunk in _lines_to_chunks(cur_lines):
                    yield doc(chunk, doc_type="law", jurisdiction="criminal",
                              split=split, source_id=law_id, law_name=law_name,
                              section_type=cur_section,
                              date=m.get("effectDate", ""))


def parse_criminal_decisions(aihub: Path, meta: Dict) -> Iterator[Dict]:
    for split, folder in _criminal_csv_folders(aihub, "결정례"):
        for csv_path in tqdm(sorted(folder.glob("*.csv")),
                             desc=f"형사결정례/{split}", leave=False):
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception:
                continue
            if "결정례일련번호" not in df.columns:
                continue
            for dec_id, grp in df.groupby("결정례일련번호"):
                lines = (
                    grp.sort_values("문장번호")["내용"]
                    .dropna().astype(str).tolist()
                )
                m    = meta.get(str(dec_id), {})
                base = dict(
                    doc_type="decision", jurisdiction="criminal", split=split,
                    source_id=str(dec_id),
                    case_name=m.get("caseName", ""),
                    case_num=m.get("caseNum", ""),
                    court=m.get("courtCode", ""),
                    date=m.get("finalDate", ""),
                    section_type="전문",
                )
                for chunk in _lines_to_chunks(lines):
                    yield doc(chunk, **base)


def parse_criminal_interpretations(aihub: Path, meta: Dict) -> Iterator[Dict]:
    for split, folder in _criminal_csv_folders(aihub, "해석례"):
        for csv_path in tqdm(sorted(folder.glob("*.csv")),
                             desc=f"형사해석례/{split}", leave=False):
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception:
                continue
            if "해석례일련번호" not in df.columns:
                continue
            for interp_id, grp in df.groupby("해석례일련번호"):
                m    = meta.get(str(interp_id), {})
                base = dict(
                    doc_type="interpretation", jurisdiction="criminal", split=split,
                    source_id=str(interp_id),
                    case_name=str(m.get("agenda", ""))[:80],
                    date=m.get("interpreDate", ""),
                )
                for section_name, sec_grp in grp.groupby("구분", sort=False):
                    lines = (
                        sec_grp.sort_values("문장번호")["내용"]
                        .dropna().astype(str).tolist()
                    )
                    for chunk in _lines_to_chunks(lines):
                        yield doc(chunk, section_type=str(section_name), **base)


# ── 민사 JSON 파서들 (기존 로직 + section_type 추가) ─────────────────────────

def _civil_json_folder(aihub: Path, sub: str, split_dir: str, prefix: str) -> Path:
    return aihub / "민사데이터" / split_dir / "01.원천데이터" / f"{prefix}_01. 민사법_{sub}"


def parse_civil_cases(aihub: Path) -> Iterator[Dict]:
    for split, split_dir, prefix in [("train", "Training", "TS"), ("val", "Validation", "VS")]:
        folder = _civil_json_folder(aihub, "001. 판결문", split_dir, prefix)
        if not folder.exists():
            continue
        for jf in tqdm(sorted(folder.glob("*.json")),
                       desc=f"민사판결문/{split}", leave=False):
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            sentences = [str(s).strip() for s in d.get("sentences", []) if s]
            if not sentences:
                continue
            base = dict(
                doc_type="case", jurisdiction="civil", split=split,
                source_id=d.get("doc_id", jf.stem),
                case_name=d.get("casenames", ""),
                court=d.get("normalized_court", ""),
                date=str(d.get("announce_date", ""))[:10],
            )
            sections = _split_into_sections(sentences)
            for sec_type, sec_lines in sections:
                for chunk in _lines_to_chunks(sec_lines):
                    yield doc(chunk, section_type=sec_type, **base)


def parse_civil_laws(aihub: Path) -> Iterator[Dict]:
    for split, split_dir, prefix in [("train", "Training", "TS"), ("val", "Validation", "VS")]:
        folder = _civil_json_folder(aihub, "002. 법령", split_dir, prefix)
        if not folder.exists():
            continue
        for jf in tqdm(sorted(folder.glob("*.json")),
                       desc=f"민사법령/{split}", leave=False):
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            sentences = [str(s).strip() for s in d.get("sentences", []) if s]
            base = dict(
                doc_type="law", jurisdiction="civil", split=split,
                source_id=d.get("statute_name", jf.stem),
                law_name=d.get("statute_name", ""),
                date=str(d.get("effective_date", ""))[:10],
                section_type="조문",
            )
            for chunk in _lines_to_chunks(sentences):
                yield doc(chunk, **base)


def parse_civil_decisions(aihub: Path) -> Iterator[Dict]:
    for split, split_dir, prefix in [("train", "Training", "TS"), ("val", "Validation", "VS")]:
        folder = _civil_json_folder(aihub, "003. 심결례", split_dir, prefix)
        if not folder.exists():
            continue
        for jf in tqdm(sorted(folder.glob("*.json")),
                       desc=f"민사심결례/{split}", leave=False):
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            sentences = [str(s).strip() for s in d.get("sentences", []) if s]
            base = dict(
                doc_type="decision", jurisdiction="civil", split=split,
                source_id=d.get("doc_id", jf.stem),
                case_name=d.get("document_type", ""),
                date=d.get("decision_date", ""),
                section_type="전문",
            )
            for chunk in _lines_to_chunks(sentences):
                yield doc(chunk, **base)


def parse_civil_interpretations(aihub: Path) -> Iterator[Dict]:
    for split, split_dir, prefix in [("train", "Training", "TS"), ("val", "Validation", "VS")]:
        folder = _civil_json_folder(aihub, "004. 유권해석", split_dir, prefix)
        if not folder.exists():
            continue
        for jf in tqdm(sorted(folder.glob("*.json")),
                       desc=f"민사유권해석/{split}", leave=False):
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            sentences = [str(s).strip() for s in d.get("sentences", []) if s]
            base = dict(
                doc_type="interpretation", jurisdiction="civil", split=split,
                source_id=d.get("doc_id", jf.stem),
                case_name=d.get("title", ""),
                court=d.get("response_institute", ""),
                date=d.get("response_date", ""),
                section_type="전문",
            )
            for chunk in _lines_to_chunks(sentences):
                yield doc(chunk, **base)


# ── 섹션 분포 통계 ────────────────────────────────────────────────────────────

def _print_stats(all_docs: List[Dict]) -> None:
    jur    = Counter(d.get("jurisdiction", "") for d in all_docs)
    sects  = Counter(d.get("section_type", "없음") for d in all_docs
                     if d.get("jurisdiction") == "criminal"
                     and d.get("doc_type") == "case")
    print(f"\n총 {len(all_docs):,}개 청크")
    print(f"  형사: {jur['criminal']:,}  /  민사: {jur['civil']:,}")
    print(f"\n형사 판결문 섹션 분포:")
    for k, v in sects.most_common():
        print(f"  {v:7,}개  [{k}]")


# ── 문서 수집 ─────────────────────────────────────────────────────────────────

def _collect_docs(aihub: Path) -> List[Dict]:
    print("[ingest_sectional] 형사 라벨 메타 로드 중...")
    criminal_meta = load_criminal_meta(aihub)

    parsers = [
        ("형사판결문",   parse_criminal_cases(aihub, criminal_meta)),
        ("형사법령",     parse_criminal_laws(aihub, criminal_meta)),
        ("형사결정례",   parse_criminal_decisions(aihub, criminal_meta)),
        ("형사해석례",   parse_criminal_interpretations(aihub, criminal_meta)),
        ("민사판결문",   parse_civil_cases(aihub)),
        ("민사법령",     parse_civil_laws(aihub)),
        ("민사심결례",   parse_civil_decisions(aihub)),
        ("민사유권해석", parse_civil_interpretations(aihub)),
    ]

    all_docs: List[Dict] = []
    for name, parser in parsers:
        before = len(all_docs)
        for d in parser:
            all_docs.append(d)
        print(f"  {name}: +{len(all_docs) - before:,}개 청크")
    return all_docs


# ── 인덱스 빌드 ───────────────────────────────────────────────────────────────
#
# 메모리 문제 해결을 위해 두 단계로 분리:
#   Phase 2a: 임베딩 → 벡터를 디스크(.npy)에 저장  (모델만 메모리에)
#   Phase 2b: 저장된 벡터 → FAISS 빌드             (모델 해제 후)
#
# 두 단계 분리 효과:
#   기존: 모델(570MB) + FAISS 누적(최대 3.6GB) → 메모리 경합 → 속도 폭망
#   개선: 각 단계는 한 가지만 메모리 사용 → 안정적 속도

def build_index(
    aihub:            Path,
    output:           Path,
    embed_model:      str  = "BAAI/bge-m3",
    embed_dim:        int  = 1024,
    batch_size:       int  = 64,
    checkpoint_every: int  = 50,
    dry_run:          bool = False,
    jurisdiction:     str | None = None,
    sleep_ms:         int  = 0,
) -> None:
    import gc, torch

    output.mkdir(parents=True, exist_ok=True)
    docs_raw_path = output / "docs_sectional.pkl"

    if jurisdiction:
        jur_output = output.parent / f"db_sectional_{jurisdiction}"
        jur_output.mkdir(parents=True, exist_ok=True)
    else:
        jur_output = output

    vec_dir        = jur_output / "vecs"          # 임시 벡터 저장 폴더
    embed_cp_path  = jur_output / "embed_cp.json" # Phase 2a 체크포인트
    faiss_cp_path  = jur_output / "faiss_cp.json" # Phase 2b 체크포인트

    # ── Phase 1: 문서 수집 ───────────────────────────────────────────────────
    if docs_raw_path.exists():
        print(f"[ingest_sectional] docs_sectional.pkl 발견 — 수집 건너뜀")
        with open(docs_raw_path, "rb") as f:
            all_docs = pickle.load(f)
        print(f"  → {len(all_docs):,}개 청크 로드")
        if jurisdiction:
            all_docs = [d for d in all_docs if d.get("jurisdiction") == jurisdiction]
            print(f"  → jurisdiction={jurisdiction} 필터 후: {len(all_docs):,}개")
    else:
        all_docs = _collect_docs(aihub)
        with open(docs_raw_path, "wb") as f:
            pickle.dump(all_docs, f)
        print(f"  → docs_sectional.pkl 저장 완료")

    _print_stats(all_docs)

    if dry_run:
        print("\n[dry-run] 임베딩·저장 생략")
        return

    # ── Phase 2a: 임베딩 → 디스크 저장 ─────────────────────────────────────
    vec_dir.mkdir(exist_ok=True)

    # 체크포인트: embed_cp.json 우선, 없으면 기존 .npy 파일에서 자동 계산
    embed_start = 0
    if embed_cp_path.exists():
        embed_start = json.loads(embed_cp_path.read_text()).get("processed", 0)
        print(f"\n[임베딩 체크포인트] {embed_start:,}/{len(all_docs):,}부터 재개")
    elif vec_dir.exists():
        existing_npy = sorted(vec_dir.glob("*.npy"))
        if existing_npy:
            last_npy   = existing_npy[-1]
            last_start = int(last_npy.stem)
            last_count = np.load(last_npy).shape[0]
            embed_start = last_start + last_count
            print(f"\n[.npy 자동감지] {embed_start:,}/{len(all_docs):,}부터 재개")
        else:
            print(f"\n[Phase 2a] 임베딩 시작 (배치: {batch_size})")
    else:
        print(f"\n[Phase 2a] 임베딩 시작 (배치: {batch_size})")

    if embed_start < len(all_docs):
        embedder  = KoreanEmbedder.get(embed_model)
        remaining = all_docs[embed_start:]
        n_batches = (len(remaining) + batch_size - 1) // batch_size

        for b_num, i in enumerate(
            tqdm(range(0, len(remaining), batch_size), total=n_batches, desc="Embedding")
        ):
            batch     = remaining[i: i + batch_size]
            vecs      = embedder.embed([d["text"] for d in batch])
            abs_start = embed_start + i
            np.save(vec_dir / f"{abs_start:08d}.npy", vecs)

            if (b_num + 1) % checkpoint_every == 0:
                processed = abs_start + len(batch)
                embed_cp_path.write_text(
                    json.dumps({"processed": processed, "total": len(all_docs)})
                )
                tqdm.write(f"  [임베딩 체크포인트] {processed:,}/{len(all_docs):,}")

        embed_cp_path.write_text(json.dumps({"processed": len(all_docs), "total": len(all_docs)}))

        # 모델 해제 — FAISS 빌드 전에 GPU 메모리 확보
        del embedder
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("  모델 해제 완료 → FAISS 빌드 시작")

    # ── Phase 2b: 디스크 벡터 → FAISS 빌드 ──────────────────────────────────
    vec_files = sorted(vec_dir.glob("*.npy"))
    if not vec_files:
        print("ERROR: 벡터 파일이 없습니다. Phase 2a부터 다시 실행하세요.")
        return

    faiss_start_file = 0
    if faiss_cp_path.exists():
        faiss_start_file = json.loads(faiss_cp_path.read_text()).get("file_idx", 0)
        print(f"[FAISS 체크포인트] {faiss_start_file}/{len(vec_files)} 파일부터 재개")
        store = FAISSStore.load(str(jur_output))
    else:
        print(f"[Phase 2b] FAISS 빌드 ({len(vec_files)}개 파일)")
        store = FAISSStore(dim=embed_dim)

    for f_idx, vec_path in enumerate(
        tqdm(vec_files[faiss_start_file:], desc="FAISS 빌드",
             initial=faiss_start_file, total=len(vec_files))
    ):
        abs_start = int(vec_path.stem)
        vecs      = np.load(vec_path)
        batch_end = min(abs_start + len(vecs), len(all_docs))
        store.add(vecs, all_docs[abs_start:batch_end])

        cur_f_idx = faiss_start_file + f_idx + 1
        if cur_f_idx % checkpoint_every == 0:
            store.save(str(jur_output))
            faiss_cp_path.write_text(json.dumps({"file_idx": cur_f_idx}))
            tqdm.write(f"  [FAISS 체크포인트] {cur_f_idx}/{len(vec_files)} 파일")

    store.save(str(jur_output))

    # 임시 파일 정리
    for f in vec_files:
        f.unlink()
    vec_dir.rmdir()
    embed_cp_path.unlink(missing_ok=True)
    faiss_cp_path.unlink(missing_ok=True)

    print(f"\n[ingest_sectional] 완료 → {jur_output}")
    print(f"  총 {len(store):,}개 벡터 저장")
    print()
    print("비교 실험 실행:")
    print("  .venv/bin/python3 scripts/run_inference.py --db data/db_sectional --model exaone3.5:7.8b")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="섹션 인식 청킹 FAISS 인덱스 빌드")
    ap.add_argument("--aihub",            default=str(PROJECT_ROOT / "AIHUB"))
    ap.add_argument("--output",           default=str(PROJECT_ROOT / "data/db_sectional"))
    ap.add_argument("--embed-model",      default="BAAI/bge-m3")
    ap.add_argument("--embed-dim",        default=1024, type=int)
    ap.add_argument("--batch-size",       default=16,   type=int)
    ap.add_argument("--checkpoint-every", default=1,    type=int)
    ap.add_argument("--dry-run",          action="store_true")
    ap.add_argument("--jurisdiction",     choices=["criminal", "civil"], default=None)
    ap.add_argument("--sleep-ms",         default=50, type=int)
    args = ap.parse_args()

    build_index(
        aihub            = Path(args.aihub),
        output           = Path(args.output),
        embed_model      = args.embed_model,
        embed_dim        = args.embed_dim,
        batch_size       = args.batch_size,
        checkpoint_every = args.checkpoint_every,
        dry_run          = args.dry_run,
        jurisdiction     = args.jurisdiction,
        sleep_ms         = args.sleep_ms,
    )
