"""
RAG Pipeline — 판례 참조조문 기반 법령 조회 구조.

검색 흐름:
  1. QueryAnalyzer → 쿼리 재작성 + 서브쿼리
  2. FAISS 검색 → case_docs 먼저 선별
  3. case_docs의 reference_law 파싱 → 실제 적용 법령 직접 조회 (exact lookup)
  4. 부족한 법령 자리를 embedding 검색으로 보완
  5. 프롬프트: [관련 법령] 먼저, [참고 판례] 후
  6. LLM 생성

핵심: 판례가 실제로 적용한 법령만 가져오므로 엉뚱한 법령 인용 방지.

research_mode=True  → 메트릭 포함 결과 (paper용)
research_mode=False → 웹 응답용 깔끔한 결과
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Settings, settings as _default_settings
from .embedder import get_embedder
from .vectorstore import FAISSStore
from .retriever import Retriever
from .generator import OllamaGenerator
from .query_analyzer import QueryAnalyzer
from .law_db import LawDB


# ── System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 형사수사 전문 AI 어시스턴트입니다.

[절대 규칙]
- 아래 [관련 법령]과 [참고 판례]에 제시된 원문 텍스트만을 근거로 답변하세요.
- 제시된 문서에 없는 법령 조항, 판례, 사실을 절대 생성하거나 언급하지 마세요.
- 법령 조항의 내용을 설명할 때 반드시 제공된 원문을 직접 인용하세요. 임의 해석·창작 금지.
- 제공된 자료로 답변이 불가능하면 "제공된 자료에서 확인할 수 없습니다"라고만 쓰세요.
- URL, 외부 링크를 절대 생성하지 마세요.

[답변 형식]
아래 4개 섹션으로 답변하세요. 각 섹션 헤더 앞뒤에 빈 줄을 넣으세요.

[답변]
(제공된 법령과 판례에 근거한 핵심 답변. 2~4문장.)

[근거법령]
- 법명 제X조(조항명): 제공된 원문에서 직접 인용
(제공된 법령 중 질문과 관련된 것만 기재. 없으면 "해당 없음")

[참조판례]
- 사건번호 (법원명, 날짜): 제공된 판례 원문 기반 요지 한 줄
(제공된 판례가 없으면 이 섹션 생략)

[참고사항]
(제공된 자료 기반 실무 유의사항. 자료에 없는 내용은 작성하지 마세요.)"""

_NOT_FOUND_RESPONSE    = "해당 질문과 관련된 정보가 제공된 매뉴얼에 없습니다."
_OUT_OF_DOMAIN_RESPONSE = "이 시스템은 형사수사 관련 질문에만 답변합니다. 체포·구속·압수수색·수사절차 등에 관한 질문을 입력해 주세요."

# 판례 참조조문에서 법령명 + 조항번호 추출
# 예: "형법 제329조", "형사소송법 제200조의3", "아동ㆍ청소년의 성보호에 관한 법률 제56조"
_REF_LAW_RE = re.compile(
    r"([가-힣·ㆍ\s]+(?:법률|규정|규칙|지침|준칙|조례|법|령))\s*제(\d+)조",
    re.UNICODE,
)


# ── Prompt builder ────────────────────────────────────────────────────────

def _build_prompt(
    question: str,
    law_docs: List[Dict],
    case_docs: List[Dict],
    analysis: Optional[Dict] = None,
) -> str:
    """
    법령 조문 + 판례를 구분하여 프롬프트 구성.
    법령 먼저, 판례 후 → LLM이 법적 근거를 우선 참조하도록 유도.
    """
    sections: List[str] = []

    # ── 관련 법령 ─────────────────────────────────────────────────────────
    if law_docs:
        law_parts = []
        for i, doc in enumerate(law_docs, 1):
            law_name   = doc.get("law_name") or doc.get("source", "")
            article    = doc.get("article_num") or doc.get("section", "")
            text       = doc.get("text", "")
            enf_date   = doc.get("enforcement_date", "")
            header = f"[법령 {i}] {law_name}"
            if article:
                header += f" 제{article}조"
            if enf_date:
                header += f" (시행 {enf_date[:4]}.{enf_date[4:6]}.{enf_date[6:]})"
            law_parts.append(f"{header}\n{text}")
        sections.append("[관련 법령]\n\n" + "\n\n---\n\n".join(law_parts))

    # ── 참고 판례 ─────────────────────────────────────────────────────────
    if case_docs:
        case_parts = []
        for i, doc in enumerate(case_docs, 1):
            source      = doc.get("source", "")
            court       = doc.get("court", "")
            case_number = doc.get("case_number", "")
            jdate       = doc.get("judgment_date", "")
            ref_law     = doc.get("reference_law", "")
            text        = doc.get("text", "")
            # 사건번호를 헤더 앞에 명시
            if case_number:
                header = f"[판례 {i}] 사건번호 {case_number}"
                if court:
                    header += f" ({court}"
                    if jdate:
                        header += f", {jdate}"
                    header += ")"
            else:
                header = f"[판례 {i}] {source}"
                if jdate:
                    header += f" ({jdate})"
            meta = ""
            if ref_law:
                first_ref = ref_law.strip().split("\n")[0]
                meta = f"\n  참조조문: {first_ref}"
            case_parts.append(f"{header}{meta}\n{text}")
        sections.append("[참고 판례]\n\n" + "\n\n---\n\n".join(case_parts))

    context_str = "\n\n".join(sections) if sections else "(관련 자료 없음)"

    # ── 쿼리 분석 힌트 ────────────────────────────────────────────────────
    hint = ""
    if analysis and not analysis.get("skipped"):
        q_type    = analysis.get("query_type", "")
        key_terms = analysis.get("key_terms", [])
        if q_type or key_terms:
            hint = f"\n\n[검색 힌트]\n질문 유형: {q_type}"
            if key_terms:
                hint += f"\n핵심 용어: {', '.join(key_terms)}"

    return f"""{context_str}

[질문]
{question}

위에 제시된 법령과 판례 원문만을 사용하여 지정된 형식([답변] / [근거법령] / [참조판례] / [참고사항])으로 답변하세요.
제시되지 않은 조항이나 판례는 절대 언급하지 마세요."""


# ── Main pipeline ─────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Usage (web):
        pipeline = RAGPipeline()
        result = await pipeline.aquery("피의자 체포 요건은?")

    Usage (research):
        pipeline = RAGPipeline()
        result = pipeline.query("피의자 체포 요건은?", research_mode=True)
    """

    def __init__(self, cfg: Optional[Settings] = None):
        self.cfg = cfg or _default_settings

        self.embedder  = get_embedder(self.cfg)
        self.store     = FAISSStore.load(self.cfg.db_path)
        self.retriever = Retriever(
            self.store,
            top_k=self.cfg.top_k,
            fetch_k=self.cfg.fetch_k,
            mmr_lambda=self.cfg.mmr_lambda,
        )
        self.generator = OllamaGenerator(
            host=self.cfg.ollama_host,
            model=self.cfg.ollama_model,
        )
        self.analyzer  = QueryAnalyzer(self.generator, enabled=True)
        self.law_db    = LawDB(self.cfg.law_db_path)
        self._law_lookup = self._build_law_lookup()

    # ── 판례 참조조문 기반 법령 직접 조회 ────────────────────────────────────

    def _build_law_lookup(self) -> Dict[Tuple[str, str], List[Dict]]:
        """
        (law_name, article_num) → [law doc, ...] 인덱스 빌드.
        앱 시작 시 1회 실행. 판례의 reference_law로 정확한 조문을 찾기 위해 사용.
        """
        from collections import defaultdict
        lookup: Dict = defaultdict(list)
        for doc in self.store.documents:
            if doc.get("doc_type") == "law":
                name = doc.get("law_name", "").strip()
                num  = str(doc.get("article_num", "")).strip()
                if name and num:
                    lookup[(name, num)].append(doc)
        print(f"[RAGPipeline] law_lookup 빌드 완료: {len(lookup):,}개 조문 인덱스")
        return dict(lookup)

    def _laws_from_case_refs(self, case_docs: List[Dict]) -> List[Dict]:
        """
        판례의 reference_law 파싱 → LawDB에서 완전 원문 조회.
        1순위: LawDB (완전 원문 보장)
        2순위: FAISS _law_lookup fallback
        """
        found: List[Dict] = []
        seen_uids: set = set()

        for doc in case_docs:
            ref_text = doc.get("reference_law", "")
            if not ref_text:
                continue
            for m in _REF_LAW_RE.finditer(ref_text):
                law_name    = m.group(1).strip()
                article_num = m.group(2).strip()

                # LawDB에서 완전 원문 조회
                entry = self.law_db.get_by_article(law_name, article_num)
                if entry:
                    uid = entry.get("unique_id", f"{law_name}_{article_num}")
                    if uid not in seen_uids:
                        seen_uids.add(uid)
                        found.append(entry)
                    continue

                # Fallback: FAISS lookup
                for law_doc in self._law_lookup.get((law_name, article_num), []):
                    uid = law_doc.get("unique_id", law_doc.get("text", "")[:40])
                    if uid not in seen_uids:
                        seen_uids.add(uid)
                        found.append(self.law_db.enrich(law_doc))

        return found

    # ── 검색: 판례 먼저 → 참조조문 법령 + 임베딩 보완 ────────────────────────

    def _retrieve_split(
        self,
        q_vec: np.ndarray,
        primary_query: str,
        sub_queries: List[str],
    ) -> Tuple[List[Dict], List[Dict], float]:
        """
        1. FAISS 검색 → case_docs 선별
        2. case_docs의 reference_law에서 실제 적용 법령 직접 조회 (정확)
        3. 부족한 법령 자리를 임베딩 검색 결과로 보완
        Returns: (law_docs, case_docs, top_score)
        """
        raw: List[Tuple[Dict, float]] = []

        for doc, score, _ in self.store.search(q_vec, self.cfg.law_fetch_k):
            raw.append((doc, score))

        for sq in sub_queries[:2]:
            if sq.strip() and sq.strip() != primary_query.strip():
                try:
                    sq_vec = self.embedder.embed_one(sq)
                    for doc, score, _ in self.store.search(sq_vec, self.cfg.fetch_k):
                        raw.append((doc, score))
                except Exception as e:
                    print(f"[RAGPipeline] 서브쿼리 오류 ({sq}): {e}")

        if not raw:
            return [], [], 0.0

        # 중복 제거
        seen: set = set()
        unique: List[Tuple[Dict, float]] = []
        for doc, score in raw:
            key = doc.get("unique_id") or doc.get("text", "")[:80]
            if key not in seen:
                seen.add(key)
                unique.append((doc, score))

        unique.sort(key=lambda x: -x[1])
        top_score = unique[0][1]

        # doc_type 분리
        law_scored  = [(d, s) for d, s in unique if d.get("doc_type") == "law"]
        case_scored = [(d, s) for d, s in unique if d.get("doc_type") == "case"]

        # 구형 데이터 호환
        if not law_scored and not case_scored:
            for d, s in unique:
                if d.get("source", "").strip():
                    law_scored.append((d, s))
                else:
                    case_scored.append((d, s))

        # ── 판례 선별 ─────────────────────────────────────────────────────
        case_docs = [d for d, _ in case_scored[:self.cfg.top_case_k]]

        # ── 법령: 참조조문 직접 조회 (1순위) ─────────────────────────────
        ref_law_docs = self._laws_from_case_refs(case_docs)

        # ── 법령: 임베딩 검색 결과로 나머지 자리 보완 (2순위) + LawDB 원문 교체 ──
        ref_uids = {d.get("unique_id", d.get("text", "")[:40]) for d in ref_law_docs}
        embed_law_docs = [
            self.law_db.enrich(d)           # FAISS 청크 → LawDB 완전 원문으로 교체
            for d, _ in law_scored
            if d.get("unique_id", d.get("text", "")[:40]) not in ref_uids
        ]

        law_docs = (ref_law_docs + embed_law_docs)[: self.cfg.top_law_k]

        return law_docs, case_docs, top_score

    # ── Sync (research batch / CLI) ───────────────────────────────────────

    def query(self, question: str, research_mode: bool = False) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # 0. Self-Query Analysis
        analysis    = self.analyzer.analyze(question)
        search_q    = analysis["rewritten"]
        sub_queries = analysis.get("sub_queries", [])
        t_analyze   = time.perf_counter()

        # 0-1. 도메인 외 즉시 차단
        if not analysis.get("in_domain", True):
            t3 = time.perf_counter()
            result: Dict[str, Any] = {
                "question": question,
                "answer":   _OUT_OF_DOMAIN_RESPONSE,
                "sources":  [],
                "analysis": analysis,
            }
            if research_mode:
                _attach_research_meta(result, [], [], t0, t_analyze, t_analyze, t_analyze, t3, self.cfg, 0.0)
            return result

        # 1. Embed
        q_vec = self.embedder.embed_one(search_q)
        t1    = time.perf_counter()

        # 2. 법령 우선 검색
        law_docs, case_docs, top_score = self._retrieve_split(q_vec, search_q, sub_queries)
        t2 = time.perf_counter()

        # 2-1. 벡터 점수 임계값 체크
        if top_score < self.cfg.min_score:
            t3 = time.perf_counter()
            result = {
                "question": question,
                "answer":   _NOT_FOUND_RESPONSE,
                "sources":  [],
                "analysis": analysis,
            }
            if research_mode:
                _attach_research_meta(result, law_docs, case_docs, t0, t_analyze, t1, t2, t3, self.cfg, top_score)
            return result

        # 2-2. Relevance Gate
        all_docs = law_docs + case_docs
        gate = self.analyzer.gate(question, all_docs)
        if not gate["answerable"]:
            t3 = time.perf_counter()
            result = {
                "question": question,
                "answer":   _NOT_FOUND_RESPONSE,
                "sources":  [],
                "analysis": analysis,
                "gate":     gate,
            }
            if research_mode:
                _attach_research_meta(result, law_docs, case_docs, t0, t_analyze, t1, t2, t3, self.cfg, top_score)
            return result

        # 3. Generate
        prompt   = _build_prompt(question, law_docs, case_docs, analysis)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        answer = self.generator.chat(messages)
        t3 = time.perf_counter()

        # 4. Build result
        law_sources  = list({d.get("source", d.get("law_name", "")) for d in law_docs})
        case_sources = list({d.get("source", "") for d in case_docs})
        result = {
            "question": question,
            "answer":   answer,
            "sources":  law_sources + case_sources,
            "analysis": {
                "query_type":  analysis.get("query_type"),
                "rewritten":   search_q,
                "sub_queries": sub_queries,
            },
        }

        if research_mode:
            result["law_chunks"]  = _chunk_info(law_docs)
            result["case_chunks"] = _chunk_info(case_docs)
            result["gate"]        = gate
            result["top_score"]   = round(top_score, 4)
            result["latency_ms"]  = {
                "analyze":  round((t_analyze - t0)  * 1000),
                "embed":    round((t1 - t_analyze)  * 1000),
                "retrieve": round((t2 - t1)         * 1000),
                "generate": round((t3 - t2)         * 1000),
                "total":    round((t3 - t0)         * 1000),
            }
            result["params"] = {
                "model":       self.cfg.ollama_model,
                "top_law_k":   self.cfg.top_law_k,
                "top_case_k":  self.cfg.top_case_k,
                "law_fetch_k": self.cfg.law_fetch_k,
                "embed":       self.cfg.embed_model,
            }

        return result

    # ── Async (FastAPI) ───────────────────────────────────────────────────

    async def aquery(self, question: str) -> Dict[str, Any]:
        import asyncio
        loop = asyncio.get_event_loop()

        analysis    = await loop.run_in_executor(None, self.analyzer.analyze, question)
        search_q    = analysis["rewritten"]
        sub_queries = analysis.get("sub_queries", [])

        if not analysis.get("in_domain", True):
            return {"question": question, "answer": _OUT_OF_DOMAIN_RESPONSE, "sources": []}

        q_vec = await loop.run_in_executor(None, self.embedder.embed_one, search_q)

        law_docs, case_docs, top_score = await loop.run_in_executor(
            None, self._retrieve_split, q_vec, search_q, sub_queries
        )

        if top_score < self.cfg.min_score:
            return {"question": question, "answer": _NOT_FOUND_RESPONSE, "sources": []}

        all_docs = law_docs + case_docs
        gate = await loop.run_in_executor(None, self.analyzer.gate, question, all_docs)
        if not gate["answerable"]:
            return {"question": question, "answer": _NOT_FOUND_RESPONSE, "sources": []}

        prompt   = _build_prompt(question, law_docs, case_docs, analysis)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        answer = await self.generator.achat(messages)

        return {
            "question": question,
            "answer":   answer,
            "sources":  list({d.get("source", d.get("law_name", "")) for d in all_docs}),
        }

    async def astream_query(self, question: str):
        import asyncio
        loop = asyncio.get_event_loop()

        analysis    = await loop.run_in_executor(None, self.analyzer.analyze, question)
        search_q    = analysis["rewritten"]
        sub_queries = analysis.get("sub_queries", [])

        if not analysis.get("in_domain", True):
            yield _OUT_OF_DOMAIN_RESPONSE
            return

        q_vec = await loop.run_in_executor(None, self.embedder.embed_one, search_q)

        law_docs, case_docs, top_score = await loop.run_in_executor(
            None, self._retrieve_split, q_vec, search_q, sub_queries
        )

        if top_score < self.cfg.min_score:
            yield _NOT_FOUND_RESPONSE
            return

        all_docs = law_docs + case_docs
        gate = await loop.run_in_executor(None, self.analyzer.gate, question, all_docs)
        if not gate["answerable"]:
            yield _NOT_FOUND_RESPONSE
            return

        prompt   = _build_prompt(question, law_docs, case_docs, analysis)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        async for token in self.generator.astream(messages):
            yield token


# ── Research helpers ──────────────────────────────────────────────────────

def _chunk_info(docs: List[Dict]) -> List[Dict]:
    return [
        {
            "text":      d.get("text", "")[:300],
            "source":    d.get("source", d.get("law_name", "")),
            "section":   d.get("section", d.get("article_num", "")),
            "doc_type":  d.get("doc_type", ""),
            "unique_id": d.get("unique_id", ""),
        }
        for d in docs
    ]


def _attach_research_meta(
    result: Dict,
    law_docs: List[Dict],
    case_docs: List[Dict],
    t0: float, t_analyze: float, t1: float, t2: float, t3: float,
    cfg,
    top_score: float,
) -> None:
    result["law_chunks"]  = _chunk_info(law_docs)
    result["case_chunks"] = _chunk_info(case_docs)
    result["top_score"]   = round(top_score, 4)
    result["latency_ms"]  = {
        "analyze":  round((t_analyze - t0) * 1000),
        "embed":    round((t1 - t_analyze) * 1000),
        "retrieve": round((t2 - t1) * 1000),
        "generate": 0,
        "total":    round((t3 - t0) * 1000),
    }
    result["params"] = {
        "model":       cfg.ollama_model,
        "top_law_k":   cfg.top_law_k,
        "top_case_k":  cfg.top_case_k,
        "law_fetch_k": cfg.law_fetch_k,
        "embed":       cfg.embed_model,
    }
