"""
법령 구조화 DB — data/law_db.json 래퍼

FAISS는 "어떤 법령인가" 식별 용도만 사용.
실제 조문 원문은 이 DB에서 조회 → 환각 방지.

법령 개정 시: law_db.json만 교체하면 됨 (FAISS 재빌드 불필요).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


class LawDB:
    """
    (unique_id) → 완전한 조문 원문 조회.

    Usage:
        db = LawDB("data/law_db.json")
        doc = db.enrich(faiss_doc)   # text 필드를 완전 원문으로 교체
    """

    def __init__(self, path: str = "data/law_db.json"):
        self._path = Path(path)
        self._db: Dict[str, Dict[str, Any]] = {}
        # (law_name, article_num) → uid 역인덱스 (조항번호로 빠른 조회)
        self._article_idx: Dict[tuple, str] = {}

        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                self._db = json.load(f)
            self._build_article_idx()
            print(f"[LawDB] {len(self._db):,}개 조문 로드 ({self._path})")
        else:
            print(f"[LawDB] {path} 없음 — FAISS 원문 그대로 사용 (build_law_db.py 실행 권장)")

    def _build_article_idx(self) -> None:
        """(law_name, article_num) → uid 인덱스 빌드."""
        for uid, entry in self._db.items():
            name = entry.get("law_name", "").strip()
            num  = str(entry.get("article_num", "")).strip()
            if name and num:
                key = (name, num)
                # 더 짧은 uid(덜 쪼개진 것) 우선
                if key not in self._article_idx or len(uid) < len(self._article_idx[key]):
                    self._article_idx[key] = uid

    # ── 조회 ──────────────────────────────────────────────────────────────

    def get(self, unique_id: str) -> Optional[Dict[str, Any]]:
        """
        unique_id로 조문 정보 반환.
        _p2, _p3 분할 suffix는 자동으로 제거해 base 조문 조회.
        """
        entry = self._db.get(unique_id)
        if entry:
            return entry
        # suffix 제거 시도
        base = re.sub(r"_p\d+$", "", unique_id)
        return self._db.get(base)

    def enrich(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        FAISS doc의 'text' 필드를 law_db의 완전 원문으로 교체.
        DB에 없으면 원본 doc 그대로 반환.
        """
        entry = self.get(doc.get("unique_id", ""))
        if not entry:
            return doc

        enriched = doc.copy()
        enriched["text"] = entry["full_text"]
        # enforcement_date 등 최신 메타도 반영
        if entry.get("enforcement_date"):
            enriched["enforcement_date"] = entry["enforcement_date"]
        return enriched

    # ── 직접 조회 (참조조문 파싱 결과로 호출) ─────────────────────────────

    def get_by_article(
        self, law_name: str, article_num: str
    ) -> Optional[Dict[str, Any]]:
        """
        (law_name, article_num)으로 조문 조회.
        역인덱스 우선 사용, 없으면 uid 직접 패턴 시도.
        """
        # 역인덱스로 바로 조회 (형법_360_①_② 같은 케이스도 처리)
        uid = self._article_idx.get((law_name, article_num))
        if uid and uid in self._db:
            return {**self._db[uid], "unique_id": uid, "doc_type": "law"}

        # 공백 없는 법명으로 재시도
        uid2 = self._article_idx.get((law_name.replace(" ", ""), article_num))
        if uid2 and uid2 in self._db:
            return {**self._db[uid2], "unique_id": uid2, "doc_type": "law"}

        # 직접 uid 패턴 fallback
        for candidate in [
            f"{law_name}_{article_num}",
            f"{law_name.replace(' ', '')}_{article_num}",
        ]:
            entry = self._db.get(candidate)
            if entry:
                return {**entry, "unique_id": candidate, "doc_type": "law"}

        return None

    def __len__(self) -> int:
        return len(self._db)
