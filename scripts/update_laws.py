#!/usr/bin/env python3
"""
법제처 Open API → data/law_db.json 갱신

인증키 발급: https://open.law.go.kr → 회원가입 → OC 키 발급
루트 .env 에 OC=applekiho 처럼 추가 (Prj_RAG/.env 와 동일한 키)

Usage:
    python scripts/update_laws.py                         # 전체 법령 갱신
    python scripts/update_laws.py --laws 형법 특정경제범죄가중처벌등에관한법률
    python scripts/update_laws.py --changed-after 20240101  # 특정일 이후 개정된 것만
    python scripts/update_laws.py --dry-run               # 실제 저장 없이 확인만

API 구조 (법제처 DRF):
    GET/POST https://www.law.go.kr/DRF/lawSearch.do   → 법령 목록 검색
    GET/POST https://www.law.go.kr/DRF/lawService.do  → 법령 조문 원문
    공통 파라미터: OC(인증키), target, type(XML|JSON), ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# 스크립트 위치(scripts/)의 상위 = 프로젝트 루트 .env 를 명시적으로 지정
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL   = "https://www.law.go.kr/DRF"
SEARCH_EP  = f"{BASE_URL}/lawSearch.do"
SERVICE_EP = f"{BASE_URL}/lawService.do"

# 경제범죄 수사지원에 필요한 법령 목록
DEFAULT_LAWS = [
    # ── 형사 기본 ──────────────────────────────────────────────────────
    "형법",                                         # 사기·횡령·배임·공갈·강제집행면탈 등
    "형사소송법",                                   # 수사절차·증거·영장

    # ── 경제범죄 가중처벌 특별법 ───────────────────────────────────────
    "특정경제범죄가중처벌등에관한법률",              # 5억 이상 사기·횡령·배임 가중
    "특정범죄가중처벌등에관한법률시행령",              # 조세·관세·뇌물 등 가중
    "조세범처벌법",                                  # 탈세·세금계산서 위조

    # ── 금융·투자사기 ──────────────────────────────────────────────────
    "자본시장과금융투자업에관한법률",                # 증권사기·불공정거래
    "유사수신행위의규제에관한법률",                  # 유사수신 투자사기
    "대부업등의등록및금융이용자보호에관한법률",      # 불법대부·고금리
    "전자금융거래법",                                # 보이스피싱·전자사기
    "전기통신금융사기 피해 방지 및 피해금 환급에 관한 특별법",  # 보이스피싱 특별법
    "외국환거래법",                                  # 환치기·불법외환

    # ── 부동산사기 ─────────────────────────────────────────────────────
    "부동산실권리자명의등기에관한법률",              # 명의신탁·차명 부동산
    "공인중개사법",                                  # 중개업자 사기
    "부동산거래신고등에관한법률",                    # 다운계약서·업계약서

    # ── 다단계·방문판매 사기 ───────────────────────────────────────────
    "방문판매등에관한법률",                          # 다단계·방문판매 사기
    "할부거래에관한법률",                            # 선불식 할부 사기

    # ── 보험사기 ───────────────────────────────────────────────────────
    "보험업법",                                      # 보험사기 관련
    "보험사기방지특별법",                            # 보험사기 가중처벌

    # ── 범죄수익 환수 ──────────────────────────────────────────────────
    "범죄수익은닉의규제및처벌등에관한법률",          # 자금세탁·몰수추징
    "공중등협박목적및대량살상무기확산을위한자금조달행위의금지에관한법률",

    # ── 공정거래·소비자 ────────────────────────────────────────────────
    "독점규제및공정거래에관한법률",                  # 담합·부당거래
    "전자상거래등에서의소비자보호에관한법률",        # 온라인 사기
    "표시광고의공정화에관한법률",                    # 허위광고·기망

    # ── 특정금융정보 ───────────────────────────────────────────────────
    "특정금융거래정보의보고및이용등에관한법률",      # 자금세탁 신고의무
]


# ── API 클라이언트 ────────────────────────────────────────────────────────────

class LawAPIClient:
    """
    법제처 DRF API 클라이언트.
    GET/POST 모두 지원 (법제처는 GET이 기본, 일부 환경에서 POST 필요).
    """

    def __init__(self, oc_key: str, method: str = "GET", delay: float = 0.5):
        if not oc_key:
            raise ValueError("LAW_OC_KEY가 .env에 없습니다. open.law.go.kr에서 발급하세요.")
        self.oc_key = oc_key
        self.method = method.upper()
        self.delay  = delay  # 초당 요청 제한 대비

    def _request(self, endpoint: str, params: Dict[str, str]) -> requests.Response:
        params = {"OC": self.oc_key, **params}
        time.sleep(self.delay)

        if self.method == "POST":
            resp = requests.post(endpoint, data=params, timeout=30)
        else:
            resp = requests.get(endpoint, params=params, timeout=30)

        resp.raise_for_status()
        return resp

    # ── 법령 검색 ──────────────────────────────────────────────────────────

    def search_law(self, query: str, page: int = 1, display: int = 20) -> List[Dict]:
        """
        법령명 검색 → [{법령명, 법령ID(MST), 시행일, ...}, ...]
        """
        resp = self._request(SEARCH_EP, {
            "target":  "law",
            "type":    "XML",
            "query":   query,
            "page":    str(page),
            "display": str(display),
        })
        return _parse_search_xml(resp.text)

    def search_changed(self, after_date: str, page: int = 1, display: int = 100) -> List[Dict]:
        """
        특정일 이후 개정된 법령 목록. after_date 형식: YYYYMMDD
        """
        resp = self._request(SEARCH_EP, {
            "target":       "law",
            "type":         "XML",
            "efYd":         after_date,   # 시행일 기준
            "page":         str(page),
            "display":      str(display),
        })
        return _parse_search_xml(resp.text)

    # ── 법령 조문 원문 ─────────────────────────────────────────────────────

    def fetch_law(self, mst: str) -> Optional[Dict[str, Any]]:
        """
        법령 일련번호(MST)로 전체 조문 원문 조회.
        반환: {law_name, mst, enforcement_date, articles: [{num, title, text}, ...]}
        """
        resp = self._request(SERVICE_EP, {
            "target": "law",
            "MST":    mst,     # ID= 아닌 MST= 사용
            "type":   "XML",
        })
        return _parse_law_xml(resp.text)


# ── XML 파싱 ─────────────────────────────────────────────────────────────────

def _parse_search_xml(xml_text: str) -> List[Dict]:
    """lawSearch 응답 XML → 법령 목록."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"  [경고] XML 파싱 실패: {e}")
        return []

    results = []
    for item in root.findall(".//law"):
        results.append({
            "law_name":         _text(item, "법령명한글"),
            "mst":              _text(item, "법령일련번호"),
            "enforcement_date": _text(item, "시행일자"),
            "promulgation_date": _text(item, "공포일자"),
            "law_type":         _text(item, "법령구분명"),
        })
    return results


def _parse_law_xml(xml_text: str) -> Optional[Dict[str, Any]]:
    """lawService 응답 XML → 법령 조문 구조.
    실제 태그 구조:
      <법령> <기본정보> <법령명_한글>, <시행일자> ...
               <조문> <조문단위> <조문번호>, <조문제목>, <조문내용>, <항>...
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"  [경고] XML 파싱 실패: {e}")
        return None

    # 오류 메시지 체크
    if root.tag == "Law" or (root.text and "없습니다" in (root.text or "")):
        print(f"  [경고] API 오류 응답: {(root.text or '').strip()}")
        return None

    # 기본정보 — 한글 태그명
    info     = root.find("기본정보")
    src      = info if info is not None else root
    law_name = _text(src, "법령명_한글")
    eff_date = _text(src, "시행일자")
    mst      = _text(src, "법령ID")

    articles = []
    seen_nums: set = set()

    for article in root.findall(".//조문단위"):
        num        = _text(article, "조문번호").strip()
        is_article = _text(article, "조문여부").strip()

        # "전문"(편장 제목) 제외, "조문"만 수집
        if is_article != "조문":
            continue
        if not num or num in seen_nums:
            continue
        seen_nums.add(num)

        title = _text(article, "조문제목") or ""
        body  = _collect_text(article, "조문내용")

        # 항 내용 수집
        hang_parts = []
        for hang in article.findall("항"):
            h_num  = _text(hang, "항번호")
            h_body = _text(hang, "항내용")
            if h_body:
                hang_parts.append(h_body)
        hang_text = "\n".join(hang_parts)

        full = "\n".join(filter(None, [body, hang_text])).strip()
        if not full:
            continue

        articles.append({"num": num, "title": title, "text": full})

    if not articles:
        print(f"  [경고] 조문 파싱 결과 없음 (law_name={law_name})")
        return None

    return {
        "law_name":         law_name,
        "mst":              mst,
        "enforcement_date": eff_date,
        "articles":         articles,
    }


def _text(element: ET.Element, path: str) -> str:
    node = element.find(path)
    return (node.text or "").strip() if node is not None else ""


def _collect_text(element: ET.Element, tag: str) -> str:
    """여러 자식 텍스트를 줄바꿈으로 합침."""
    parts = []
    for child in element.findall(tag):
        t = "".join(child.itertext()).strip()
        if t:
            parts.append(t)
    return "\n".join(parts)


# ── 백업 ─────────────────────────────────────────────────────────────────────

def _backup_db(db_file: Path) -> None:
    """기존 law_db.json을 data/backup/ 폴더에 타임스탬프로 백업."""
    if not db_file.exists():
        return
    backup_dir = db_file.parent / "backup"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"{db_file.stem}_{stamp}{db_file.suffix}"
    shutil.copy2(db_file, dest)
    print(f"[backup] {db_file.name} → {dest.relative_to(db_file.parent.parent)}")


# ── law_db.json 갱신 ─────────────────────────────────────────────────────────

def _to_db_entries(law_data: Dict[str, Any]) -> Dict[str, Dict]:
    """
    API 응답 → law_db.json 형식으로 변환.
    key: {법령명}_{조번호}  (build_law_db.py와 동일한 형식)
    """
    entries = {}
    law_name = law_data["law_name"]
    mst      = law_data["mst"]
    eff_date = law_data["enforcement_date"]

    for art in law_data["articles"]:
        uid = f"{law_name}_{art['num']}"
        entries[uid] = {
            "law_name":         law_name,
            "law_num":          mst,
            "article_num":      art["num"],
            "enforcement_date": eff_date,
            "full_text":        art["text"],
        }
    return entries


def update_law_db(
    client:        LawAPIClient,
    law_names:     List[str],
    db_path:       str,
    changed_after: Optional[str] = None,
    dry_run:       bool = False,
) -> None:
    db_file = Path(db_path)

    # 기존 DB 로드
    existing: Dict[str, Dict] = {}
    if db_file.exists():
        with open(db_file, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"[update] 기존 law_db: {len(existing):,}개 조문")
    else:
        print(f"[update] {db_path} 없음 — 새로 생성")

    updated_count = 0
    skipped_count = 0

    for law_name in law_names:
        print(f"\n[update] 검색 중: {law_name}")
        results = client.search_law(law_name, display=5)

        if not results:
            print(f"  → 검색 결과 없음")
            continue

        # 법령명이 정확히 일치하는 것 우선
        match = next(
            (r for r in results if r["law_name"] == law_name),
            results[0]
        )
        mst      = match["mst"]
        eff_date = match["enforcement_date"]

        # 시행일 비교 — 이미 최신이면 스킵
        if changed_after and eff_date and eff_date <= changed_after:
            print(f"  → {law_name} 시행일 {eff_date} ≤ {changed_after}, 스킵")
            skipped_count += 1
            continue

        # 기존 DB의 해당 법령 시행일 확인
        existing_dates = {
            v["enforcement_date"]
            for k, v in existing.items()
            if v.get("law_name") == law_name and v.get("enforcement_date")
        }
        if existing_dates and eff_date and eff_date in existing_dates:
            print(f"  → {law_name} 이미 최신 ({eff_date}), 스킵")
            skipped_count += 1
            continue

        print(f"  → MST={mst}, 시행일={eff_date} 조문 다운로드 중...")
        law_data = client.fetch_law(mst)

        if not law_data:
            print(f"  → 조문 파싱 실패")
            continue

        new_entries = _to_db_entries(law_data)
        print(f"  → {len(new_entries)}개 조문 파싱 완료")

        if not dry_run:
            # 기존 DB에서 해당 법령 항목 제거 후 새 항목으로 교체
            existing = {k: v for k, v in existing.items()
                        if v.get("law_name") != law_name}
            existing.update(new_entries)

        updated_count += 1

    if dry_run:
        print(f"\n[update] dry-run 완료 — 갱신 대상 {updated_count}개, 스킵 {skipped_count}개 (저장 안 함)")
        return

    # 저장 (기존 파일 백업 후 덮어쓰기)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    _backup_db(db_file)
    with open(db_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"\n[update] 완료 — 갱신 {updated_count}개, 스킵 {skipped_count}개")
    print(f"[update] 총 {len(existing):,}개 조문 → {db_path}")


# ── 개정 법령 감지 ────────────────────────────────────────────────────────────

def check_changed(client: LawAPIClient, after_date: str) -> None:
    """특정일 이후 개정된 법령 목록만 출력 (갱신 없이 확인용)."""
    print(f"[check] {after_date} 이후 개정 법령 조회 중...")
    results = client.search_changed(after_date, display=100)
    if not results:
        print("  → 없음")
        return
    for r in results:
        print(f"  {r['law_name']} | 시행일: {r['enforcement_date']} | MST: {r['mst']}")
    print(f"\n총 {len(results)}건")


# ── 진입점 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="법제처 API → law_db.json 갱신")
    parser.add_argument("--laws",          nargs="+", default=DEFAULT_LAWS,
                        help="갱신할 법령명 목록 (기본: 경제범죄 관련 8개)")
    parser.add_argument("--db",            default="data/law_db.json")
    parser.add_argument("--changed-after", default=None,
                        help="이 날짜 이후 개정된 법령만 갱신 (YYYYMMDD)")
    parser.add_argument("--check-only",    action="store_true",
                        help="개정 목록만 출력, 실제 갱신 안 함")
    parser.add_argument("--dry-run",       action="store_true",
                        help="다운로드는 하되 저장 안 함")
    parser.add_argument("--method",        default="GET", choices=["GET", "POST"],
                        help="HTTP 메서드 (기본: GET)")
    parser.add_argument("--delay",         type=float, default=0.5,
                        help="요청 간 대기 초 (기본: 0.5)")
    args = parser.parse_args()

    oc_key = os.getenv("OC", "")
    if not oc_key:
        print("[오류] .env 파일에 OC=발급받은키 를 추가하세요.")
        print("       발급: https://open.law.go.kr → 로그인 → 마이페이지 → 인증키 관리")
        raise SystemExit(1)

    client = LawAPIClient(oc_key, method=args.method, delay=args.delay)

    if args.check_only:
        after = args.changed_after or "20240101"
        check_changed(client, after)
    else:
        update_law_db(
            client        = client,
            law_names     = args.laws,
            db_path       = args.db,
            changed_after = args.changed_after,
            dry_run       = args.dry_run,
        )
