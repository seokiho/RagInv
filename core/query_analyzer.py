"""
Query Intelligence Layer — 두 단계로 동문서답을 차단합니다.

Stage 1 · Self-Query  (검색 전)
    - 질문 의도·유형 파악
    - 모호하면 검색에 최적화된 쿼리로 재작성
    - 복합 질문이면 서브쿼리로 분해
    - 관련 법령 예측 (어떤 법에서 답을 찾아야 하는지)

Stage 2 · Relevance Gate  (검색 후, 생성 전)
    - "이 문서들로 질문에 실제로 답할 수 있는가?" LLM 판단
    - NO이면 생성 단계를 건너뛰고 "정보 없음" 반환
    → 핵심: 엉뚱한 문서가 검색돼도 엉뚱한 답변을 막음

Usage:
    from core.query_analyzer import QueryAnalyzer
    from core.generator import OllamaGenerator

    gen      = OllamaGenerator()
    analyzer = QueryAnalyzer(gen)

    # Stage 1
    result = analyzer.analyze(question)
    search_query = result["rewritten"]   # 재작성된 쿼리 (원본과 같을 수도 있음)
    sub_queries  = result["sub_queries"] # 복합 질문 분해 (없으면 [])
    law_hints    = result["law_hints"]   # 관련 법령 힌트 (없으면 [])

    # Stage 2
    ok = analyzer.gate(question, docs)   # {"answerable": bool, ...}
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

# ──────────────────────────────────────────────────────────────────────────────
# 도메인 지식: 형사수사 관련 주요 법령
# ──────────────────────────────────────────────────────────────────────────────
_LAW_DOMAIN_CONTEXT = """[형사수사 관련 주요 법령]
- 형사소송법: 수사, 체포, 구속, 압수수색, 증거, 공소 전반
- 경찰관직무집행법: 불심검문, 보호조치, 위험발생 방지, 경찰장비 사용
- 검사와 사법경찰관의 상호협력과 일반적 수사준칙에 관한 규정: 수사개시, 송치, 협력
- 형법: 범죄 구성요건, 처벌규정
- 통신비밀보호법: 통신수사(감청, 통신사실 확인자료)
- 특정범죄가중처벌 등에 관한 법률: 주요 강력범죄 가중처벌
- 범죄수익은닉의 규제 및 처벌 등에 관한 법률: 몰수·추징
- 사법경찰관리의 직무를 수행할 자와 그 직무범위에 관한 법률: 특별사법경찰관"""

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 ·  Self-Query  프롬프트
# ──────────────────────────────────────────────────────────────────────────────
_SELF_QUERY_SYSTEM = f"""당신은 형사수사 RAG 시스템의 쿼리 분석 전문가입니다.
사용자 질문을 분석하여 정확한 문서 검색이 가능하도록 JSON으로 출력하세요.
반드시 JSON만 출력하고 다른 텍스트는 포함하지 마세요.

{_LAW_DOMAIN_CONTEXT}"""

_SELF_QUERY_USER = """다음 질문을 분석하세요.

[질문]
{question}

아래 JSON 형식으로만 출력하세요:
{{
  "in_domain": true 또는 false,
  "query_type": "절차|요건|정의|법령조항|서식|처벌|권한|기간|예외|기타",
  "clarity": "명확|모호",
  "key_terms": ["핵심 법률 용어 1", "핵심 법률 용어 2"],
  "law_hints": ["관련 법령1", "관련 법령2"],
  "rewritten": "검색에 최적화된 재작성 쿼리",
  "sub_queries": ["서브쿼리1", "서브쿼리2"]
}}

각 필드 설명:
- in_domain: 형사수사·경찰수사·법령·범죄수사 관련 질문이면 true.
    일상 대화, 욕설, 무의미한 텍스트, 음식/날씨/주식 등 무관한 주제면 false.
    의심스러우면 true로 처리.
- query_type: in_domain=false이면 "기타"로 고정.
    절차(수사 단계·방법), 요건(성립 조건), 정의(용어 설명), 법령조항(특정 조문),
    서식(양식·서류), 처벌(형량·제재), 권한(수사기관 권한), 기간(시한·기한),
    예외(예외규정), 기타
- clarity: 대상·범위가 구체적이면 "명확", 불분명하면 "모호"
- key_terms: 검색 핵심 법률 용어 최대 4개. 구어체는 법률 용어로 변환.
    in_domain=false이면 빈 배열 [].
    예) "잡을 수 있나요?" → "체포 요건", "긴급체포"
- law_hints: 이 질문의 답이 있을 가능성이 높은 법령명 최대 3개.
    in_domain=false이면 빈 배열 [].
    예) 체포 질문 → ["형사소송법", "경찰관직무집행법"]
- rewritten: in_domain=false이면 원본 그대로.
    모호하면 법률 용어 + 상황을 구체화해서 재작성.
    명확하면 원본 그대로. 구어체는 법률용어로.
    예) "경찰이 현행범 잡을 때 뭐해야 해?" → "현행범인 체포 시 고지 의무 및 절차"
- sub_queries: in_domain=false이면 빈 배열 [].
    복합 질문(2개 이상의 독립된 쟁점 포함)이면 쟁점별로 분해 (최대 3개).
    단순 질문은 반드시 빈 배열 [].
    예) "긴급체포 요건과 석방 절차는?" →
        ["긴급체포 성립 요건 형사소송법", "긴급체포 후 석방 절차 및 기간"]"""


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 ·  Relevance Gate  프롬프트
# ──────────────────────────────────────────────────────────────────────────────
_GATE_SYSTEM = """당신은 형사수사 문서 관련성 판단 전문가입니다.
주어진 문서들이 질문에 답변할 수 있는지 판단하여 JSON으로만 출력하세요."""

_GATE_USER = """[질문]
{question}

[검색된 문서 요약]
{doc_snippets}

위 문서들로 질문에 충분히 답할 수 있습니까?

다음 JSON 형식으로만 출력하세요:
{{
  "answerable": true 또는 false,
  "reason": "판단 근거 한 문장",
  "confidence": "high|medium|low"
}}

판단 기준:
- true (high): 문서에 질문과 직접 관련된 구체적 내용(절차, 조건, 법령 조항 등)이 있음
- true (medium): 문서가 관련 분야이나 일부 내용만 포함 — 부분적 답변 가능
- false: 문서가 다른 주제이거나, 질문의 핵심 내용이 없어 실질적 답변 불가"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _extract_json(text: str) -> Dict[str, Any]:
    """LLM 출력에서 JSON 블록을 파싱합니다 (```json ... ``` 포함)."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"JSON을 찾을 수 없음: {text[:200]}")
    return json.loads(text[start:end])


def _doc_snippet(doc: Dict[str, Any], max_chars: int = 250) -> str:
    text    = doc.get("text") or doc.get("page_content", "")
    src     = doc.get("source", "")
    section = doc.get("section", "")
    label   = f"[{src}]" + (f" {section}" if section else "")
    return f"{label} {text[:max_chars]}"


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────
class QueryAnalyzer:
    """
    Parameters
    ----------
    generator   : OllamaGenerator instance
    enabled     : False로 끄면 analyze()는 passthrough, gate()는 항상 True 반환
                  → ablation 실험 시 비교용
    """

    def __init__(self, generator, enabled: bool = True):
        self.gen     = generator
        self.enabled = enabled

    # ── Stage 1 ──────────────────────────────────────────────────────────
    def analyze(self, question: str) -> Dict[str, Any]:
        """
        질문을 분석하고 검색용 쿼리를 반환합니다.

        Returns
        -------
        {
            "original":    원본 질문,
            "rewritten":   검색용 재작성 쿼리,
            "query_type":  str,
            "clarity":     "명확|모호",
            "key_terms":   List[str],
            "law_hints":   List[str],   ← 관련 법령 힌트 (신규)
            "sub_queries": List[str],
            "skipped":     bool
        }
        """
        default = {
            "original":    question,
            "rewritten":   question,
            "in_domain":   True,   # 분석 실패 시 안전하게 통과
            "query_type":  "기타",
            "clarity":     "명확",
            "key_terms":   [],
            "law_hints":   [],
            "sub_queries": [],
            "skipped":     True,
        }

        if not self.enabled:
            return default

        try:
            messages = [
                {"role": "system", "content": _SELF_QUERY_SYSTEM},
                {"role": "user",   "content": _SELF_QUERY_USER.format(question=question)},
            ]
            raw    = self.gen.chat(messages, temperature=0.0)
            parsed = _extract_json(raw)

            in_domain = bool(parsed.get("in_domain", True))
            return {
                "original":    question,
                "rewritten":   parsed.get("rewritten", question) or question,
                "in_domain":   in_domain,
                "query_type":  parsed.get("query_type", "기타"),
                "clarity":     parsed.get("clarity", "명확"),
                "key_terms":   parsed.get("key_terms", []),
                "law_hints":   parsed.get("law_hints", []),
                "sub_queries": [s for s in parsed.get("sub_queries", []) if s.strip()],
                "skipped":     False,
            }

        except Exception as e:
            print(f"[QueryAnalyzer.analyze] 오류 (원본 쿼리 사용): {e}")
            return default

    # ── Stage 2 ──────────────────────────────────────────────────────────
    def gate(
        self,
        question: str,
        docs: List[Dict[str, Any]],
        threshold: int = 1,
    ) -> Dict[str, Any]:
        """
        검색된 문서들로 질문에 답할 수 있는지 판단합니다.

        Returns
        -------
        {
            "answerable": bool,
            "reason":     str,
            "confidence": "high|medium|low",
            "skipped":    bool
        }
        """
        if len(docs) < threshold:
            return {
                "answerable": False,
                "reason":     "검색된 문서 없음",
                "confidence": "high",
                "skipped":    False,
            }

        if not self.enabled:
            return {
                "answerable": True,
                "reason":     "gate 비활성화",
                "confidence": "high",
                "skipped":    True,
            }

        try:
            snippets = "\n".join(
                f"{i+1}. {_doc_snippet(d)}" for i, d in enumerate(docs[:5])
            )
            messages = [
                {"role": "system", "content": _GATE_SYSTEM},
                {"role": "user",   "content": _GATE_USER.format(
                    question=question,
                    doc_snippets=snippets,
                )},
            ]
            raw    = self.gen.chat(messages, temperature=0.0)
            parsed = _extract_json(raw)

            return {
                "answerable": bool(parsed.get("answerable", True)),
                "reason":     parsed.get("reason", ""),
                "confidence": parsed.get("confidence", "medium"),
                "skipped":    False,
            }

        except Exception as e:
            print(f"[QueryAnalyzer.gate] 오류 (통과 처리): {e}")
            return {
                "answerable": True,
                "reason":     f"gate 오류: {e}",
                "confidence": "low",
                "skipped":    True,
            }
