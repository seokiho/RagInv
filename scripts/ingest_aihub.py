#!/usr/bin/env python3
"""
AIHUB 형사·민사 법률 데이터 → data/db_aihub/ FAISS 인덱스 빌드

형사: 원천 CSV (판결문/법령/결정례/해석례) + 라벨 JSON으로 메타 보강
민사: 원천 JSON (판결문/법령/심결례/유권해석) 직접 파싱

기존 data/db/ 는 건드리지 않음.

Usage:
    .venv/bin/python3 scripts/ingest_aihub.py --dry-run     # 청크 수 확인만
    .venv/bin/python3 scripts/ingest_aihub.py               # 전체 빌드
    .venv/bin/python3 scripts/ingest_aihub.py --aihub AIHUB --output data/db_aihub
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import settings
from core.embedder import KoreanEmbedder
from core.vectorstore import FAISSStore

MAX_CHARS = 1500   # 청크 최대 길이 (~500 토큰)
OVERLAP   = 100    # 청크 간 오버랩


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def split_text(text: str) -> List[str]:
    """긴 텍스트를 MAX_CHARS 단위로 분할."""
    if len(text) <= MAX_CHARS:
        return [text] if text.strip() else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        if end < len(text):
            for sep in ("\n", ".", " "):
                pos = text.rfind(sep, start + OVERLAP, end)
                if pos > start:
                    end = pos + 1
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - OVERLAP if end < len(text) else end
    return chunks


def doc(text: str, **meta: Any) -> Dict[str, Any]:
    return {"text": text, **meta}


# ── 형사 라벨 메타 로드 ───────────────────────────────────────────────────────

def load_criminal_meta(aihub: Path) -> Dict[str, Dict]:
    """
    형사 라벨 JSON 전체 → {source_id: info_dict}
    판결문(precedId), 법령(lawId), 결정례(determintId), 해석례(interpreId)
    """
    id_field_map = {
        "TL_판결문_QA":  "precedId",   "VL_판결문_QA":  "precedId",
        "TL_판결문_SUM": "precedId",   "VL_판결문_SUM": "precedId",
        "TL_법령_QA":    "lawId",      "VL_법령_QA":    "lawId",
        "TL_결정례_QA":  "determintId","VL_결정례_QA":  "determintId",
        "TL_결정례_SUM": "determintId","VL_결정례_SUM": "determintId",
        "TL_해석례_QA":  "interpreId", "VL_해석례_QA":  "interpreId",
        "TL_해석례_SUM": "interpreId", "VL_해석례_SUM": "interpreId",
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
                    d = json.loads(jf.read_text(encoding="utf-8"))
                    info = d.get("info", {})
                    sid = str(info.get(id_field, "")).strip()
                    if sid and sid not in meta:
                        meta[sid] = info
                except Exception:
                    pass
    print(f"  → 형사 메타 {len(meta):,}건 로드")
    return meta


# ── 형사 CSV 파서들 ───────────────────────────────────────────────────────────

def _criminal_csv_folders(aihub: Path, sub: str):
    """Training/Validation 양쪽 폴더와 split 레이블을 반환."""
    for split_name, split_dir, prefix in [
        ("train", "Training",   "TS"),
        ("val",   "Validation", "VS"),
    ]:
        folder = aihub / "형사데이터" / split_dir / "01.원천데이터" / f"{prefix}_{sub}"
        if folder.exists():
            yield split_name, folder


def parse_criminal_cases(aihub: Path, meta: Dict) -> Iterator[Dict]:
    """형사 판결문 CSV → 판례 단위 청크."""
    for split, folder in _criminal_csv_folders(aihub, "판결문"):
        csv_files = sorted(folder.glob("*.csv"))
        for csv_path in tqdm(csv_files, desc=f"형사판결문/{split}", leave=False):
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception:
                continue
            if "판례일련번호" not in df.columns:
                continue
            for prec_id, grp in df.groupby("판례일련번호"):
                text = "\n".join(
                    grp.sort_values("문장번호")["내용"].dropna().astype(str)
                ).strip()
                if not text:
                    continue
                m = meta.get(str(prec_id), {})
                base = dict(
                    doc_type="case", jurisdiction="criminal", split=split,
                    source_id=str(prec_id),
                    case_name=m.get("caseName", ""),
                    case_num=m.get("caseNum", ""),
                    court=m.get("courtName", ""),
                    date=m.get("sentenceDate", ""),
                    section="판례내용",
                )
                for chunk in split_text(text):
                    yield doc(chunk, **base)


def parse_criminal_laws(aihub: Path, meta: Dict) -> Iterator[Dict]:
    """형사 법령 CSV → 조문 단위 청크."""
    for split, folder in _criminal_csv_folders(aihub, "법령"):
        for csv_path in tqdm(sorted(folder.glob("*.csv")), desc=f"형사법령/{split}", leave=False):
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception:
                continue
            if "법령일련번호" not in df.columns:
                continue
            law_id   = str(df["법령일련번호"].iloc[0])
            m        = meta.get(law_id, {})
            law_name = m.get("title", csv_path.stem)

            # 조문(구분='조문') 경계마다 청크 분리
            cur_lines: List[str] = []
            cur_section = ""
            for _, row in df.iterrows():
                section = str(row.get("구분", "")).strip()
                content = str(row.get("내용", "")).strip()
                if not content:
                    continue
                if section == "조문" and cur_lines:
                    text = "\n".join(cur_lines).strip()
                    for chunk in split_text(text):
                        yield doc(chunk, doc_type="law", jurisdiction="criminal",
                                  split=split, source_id=law_id, law_name=law_name,
                                  section=cur_section, date=m.get("effectDate", ""))
                    cur_lines, cur_section = [], content
                else:
                    if section == "조문":
                        cur_section = content
                    cur_lines.append(content)
            if cur_lines:
                text = "\n".join(cur_lines).strip()
                for chunk in split_text(text):
                    yield doc(chunk, doc_type="law", jurisdiction="criminal",
                              split=split, source_id=law_id, law_name=law_name,
                              section=cur_section, date=m.get("effectDate", ""))


def parse_criminal_decisions(aihub: Path, meta: Dict) -> Iterator[Dict]:
    """형사 결정례 CSV → 전문 단위 청크."""
    for split, folder in _criminal_csv_folders(aihub, "결정례"):
        for csv_path in tqdm(sorted(folder.glob("*.csv")), desc=f"형사결정례/{split}", leave=False):
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception:
                continue
            if "결정례일련번호" not in df.columns:
                continue
            for dec_id, grp in df.groupby("결정례일련번호"):
                text = "\n".join(
                    grp.sort_values("문장번호")["내용"].dropna().astype(str)
                ).strip()
                if not text:
                    continue
                m = meta.get(str(dec_id), {})
                base = dict(
                    doc_type="decision", jurisdiction="criminal", split=split,
                    source_id=str(dec_id),
                    case_name=m.get("caseName", ""),
                    case_num=m.get("caseNum", ""),
                    court=m.get("courtCode", ""),
                    date=m.get("finalDate", ""),
                    section="전문",
                )
                for chunk in split_text(text):
                    yield doc(chunk, **base)


def parse_criminal_interpretations(aihub: Path, meta: Dict) -> Iterator[Dict]:
    """형사 해석례 CSV → 질의요지/회답/이유 섹션별 청크."""
    for split, folder in _criminal_csv_folders(aihub, "해석례"):
        for csv_path in tqdm(sorted(folder.glob("*.csv")), desc=f"형사해석례/{split}", leave=False):
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception:
                continue
            if "해석례일련번호" not in df.columns:
                continue
            for interp_id, grp in df.groupby("해석례일련번호"):
                m = meta.get(str(interp_id), {})
                base = dict(
                    doc_type="interpretation", jurisdiction="criminal", split=split,
                    source_id=str(interp_id),
                    case_name=str(m.get("agenda", ""))[:80],
                    date=m.get("interpreDate", ""),
                )
                for section_name, sec_grp in grp.groupby("구분", sort=False):
                    text = "\n".join(
                        sec_grp.sort_values("문장번호")["내용"].dropna().astype(str)
                    ).strip()
                    if not text:
                        continue
                    for chunk in split_text(text):
                        yield doc(chunk, section=str(section_name), **base)


# ── 민사 JSON 파서들 ──────────────────────────────────────────────────────────

def _civil_json_folder(aihub: Path, sub: str, split_dir: str, prefix: str) -> Path:
    return aihub / "민사데이터" / split_dir / "01.원천데이터" / f"{prefix}_01. 민사법_{sub}"


def parse_civil_cases(aihub: Path) -> Iterator[Dict]:
    """민사 판결문 JSON → sentences 합치기."""
    for split, split_dir, prefix in [("train","Training","TS"), ("val","Validation","VS")]:
        folder = _civil_json_folder(aihub, "001. 판결문", split_dir, prefix)
        if not folder.exists():
            continue
        for jf in tqdm(sorted(folder.glob("*.json")), desc=f"민사판결문/{split}", leave=False):
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            text = " ".join(str(s).strip() for s in d.get("sentences", []) if s).strip()
            if not text:
                continue
            base = dict(
                doc_type="case", jurisdiction="civil", split=split,
                source_id=d.get("doc_id", jf.stem),
                case_name=d.get("casenames", ""),
                court=d.get("normalized_court", ""),
                date=str(d.get("announce_date", ""))[:10],
                section="",
            )
            for chunk in split_text(text):
                yield doc(chunk, **base)


def parse_civil_laws(aihub: Path) -> Iterator[Dict]:
    """민사 법령 JSON → sentences 합치기."""
    for split, split_dir, prefix in [("train","Training","TS"), ("val","Validation","VS")]:
        folder = _civil_json_folder(aihub, "002. 법령", split_dir, prefix)
        if not folder.exists():
            continue
        for jf in tqdm(sorted(folder.glob("*.json")), desc=f"민사법령/{split}", leave=False):
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            text = "\n".join(str(s).strip() for s in d.get("sentences", []) if s).strip()
            if not text:
                continue
            base = dict(
                doc_type="law", jurisdiction="civil", split=split,
                source_id=d.get("statute_name", jf.stem),
                law_name=d.get("statute_name", ""),
                date=str(d.get("effective_date", ""))[:10],
            )
            for chunk in split_text(text):
                yield doc(chunk, **base)


def parse_civil_decisions(aihub: Path) -> Iterator[Dict]:
    """민사 심결례 JSON → sentences 합치기."""
    for split, split_dir, prefix in [("train","Training","TS"), ("val","Validation","VS")]:
        folder = _civil_json_folder(aihub, "003. 심결례", split_dir, prefix)
        if not folder.exists():
            continue
        for jf in tqdm(sorted(folder.glob("*.json")), desc=f"민사심결례/{split}", leave=False):
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            text = " ".join(str(s).strip() for s in d.get("sentences", []) if s).strip()
            if not text:
                continue
            base = dict(
                doc_type="decision", jurisdiction="civil", split=split,
                source_id=d.get("doc_id", jf.stem),
                case_name=d.get("document_type", ""),
                date=d.get("decision_date", ""),
            )
            for chunk in split_text(text):
                yield doc(chunk, **base)


def parse_civil_interpretations(aihub: Path) -> Iterator[Dict]:
    """민사 유권해석 JSON → sentences 합치기."""
    for split, split_dir, prefix in [("train","Training","TS"), ("val","Validation","VS")]:
        folder = _civil_json_folder(aihub, "004. 유권해석", split_dir, prefix)
        if not folder.exists():
            continue
        for jf in tqdm(sorted(folder.glob("*.json")), desc=f"민사유권해석/{split}", leave=False):
            try:
                d = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            text = " ".join(str(s).strip() for s in d.get("sentences", []) if s).strip()
            if not text:
                continue
            base = dict(
                doc_type="interpretation", jurisdiction="civil", split=split,
                source_id=d.get("doc_id", jf.stem),
                case_name=d.get("title", ""),
                court=d.get("response_institute", ""),
                date=d.get("response_date", ""),
            )
            for chunk in split_text(text):
                yield doc(chunk, **base)


# ── 메인 ─────────────────────────────────────────────────────────────────────

def _collect_docs(aihub: Path) -> List[Dict]:
    """Phase 1: 모든 파서 실행 → 문서 리스트 반환."""
    print("[ingest_aihub] 형사 라벨 메타 로드 중...")
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


def _print_stats(all_docs: List[Dict]) -> None:
    jur    = Counter(d["jurisdiction"] for d in all_docs)
    types  = Counter(f"{d['jurisdiction']}/{d['doc_type']}" for d in all_docs)
    splits = Counter(d["split"] for d in all_docs)
    print(f"\n총 {len(all_docs):,}개 청크")
    print(f"  형사: {jur['criminal']:,}  /  민사: {jur['civil']:,}")
    print(f"  train: {splits['train']:,}  /  val: {splits['val']:,}")
    for k, v in sorted(types.items()):
        print(f"  {k}: {v:,}")


def build_index(
    aihub: Path,
    output: Path,
    embed_model: str = "BAAI/bge-m3",
    embed_dim: int = 1024,
    batch_size: int = 16,
    checkpoint_every: int = 50,
    dry_run: bool = False,
    jurisdiction: str | None = None,  # None=전체, "criminal", "civil"
    sleep_ms: int = 0,                 # 배치 사이 대기 (ms) — 시스템 과부하 방지
) -> None:
    import time

    output.mkdir(parents=True, exist_ok=True)
    docs_raw_path    = output / "docs_raw.pkl"
    checkpoint_path  = output / "checkpoint.json"

    # jurisdiction 필터가 있으면 별도 출력 경로 사용
    if jurisdiction:
        jur_output = output.parent / f"db_aihub_{jurisdiction}"
        jur_output.mkdir(parents=True, exist_ok=True)
        jur_checkpoint = jur_output / "checkpoint.json"
    else:
        jur_output     = output
        jur_checkpoint = checkpoint_path

    # ── Phase 1: 문서 수집 (이미 했으면 로드) ────────────────────────────
    if docs_raw_path.exists():
        print(f"[ingest_aihub] docs_raw.pkl 발견 — 수집 건너뜀")
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
        print(f"  → docs_raw.pkl 저장 완료")

    _print_stats(all_docs)

    if dry_run:
        print("\n[ingest_aihub] dry-run 완료 — 임베딩·저장 생략")
        return

    # ── Phase 2: 임베딩 + FAISS (배치 단위 즉시 추가, 체크포인트) ────────
    start_idx = 0
    if jur_checkpoint.exists():
        cp = json.loads(jur_checkpoint.read_text())
        start_idx = cp.get("processed", 0)
        print(f"\n[ingest_aihub] 체크포인트 발견 → {start_idx:,}/{len(all_docs):,}부터 재개")
        store = FAISSStore.load(str(jur_output))
    else:
        store = FAISSStore(dim=embed_dim)

    print(f"\n[ingest_aihub] 임베딩 시작 (모델: {embed_model}, 배치: {batch_size})")
    embedder  = KoreanEmbedder.get(embed_model)
    remaining = all_docs[start_idx:]
    n_batches = (len(remaining) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(tqdm(range(0, len(remaining), batch_size),
                                       total=n_batches, desc="Embedding")):
        batch = remaining[i: i + batch_size]
        vecs  = embedder.embed([d["text"] for d in batch])
        store.add(vecs, batch)

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

        # 체크포인트 저장
        processed = start_idx + i + len(batch)
        if (batch_num + 1) % checkpoint_every == 0:
            store.save(str(jur_output))
            jur_checkpoint.write_text(
                json.dumps({"processed": processed, "total": len(all_docs)})
            )
            tqdm.write(f"  [체크포인트] {processed:,}/{len(all_docs):,} 저장")

    # ── 최종 저장 및 임시 파일 정리 ──────────────────────────────────────
    store.save(str(jur_output))
    jur_checkpoint.unlink(missing_ok=True)
    if not jurisdiction:  # 전체 빌드일 때만 docs_raw.pkl 삭제
        docs_raw_path.unlink(missing_ok=True)
    print(f"\n[ingest_aihub] 완료 → {jur_output}")
    print(f"  총 {len(store):,}개 벡터 저장")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AIHUB → FAISS 인덱스 빌드")
    ap.add_argument("--aihub",       default=str(PROJECT_ROOT / "AIHUB"),         help="AIHUB 루트 폴더")
    ap.add_argument("--output",      default=str(PROJECT_ROOT / "data/db_aihub"), help="출력 디렉토리")
    ap.add_argument("--embed-model", default="BAAI/bge-m3",    help="임베딩 모델 (기본: BGE-M3)")
    ap.add_argument("--embed-dim",   default=1024, type=int,   help="임베딩 차원 (기본: 1024)")
    ap.add_argument("--batch-size",  default=16,   type=int,   help="임베딩 배치 크기 (기본: 16)")
    ap.add_argument("--checkpoint-every", default=50, type=int, help="N배치마다 중간 저장 (기본: 50)")
    ap.add_argument("--dry-run",     action="store_true",      help="청크 수만 확인, 저장 안 함")
    ap.add_argument("--jurisdiction", choices=["criminal", "civil"], default=None,
                    help="특정 jurisdiction만 임베딩 (기본: 전체)")
    ap.add_argument("--sleep-ms",    default=50, type=int,
                    help="배치 사이 대기 ms, 시스템 부하 조절 (기본: 50)")
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
