"""
형사수사 매뉴얼 QA  —  FastAPI Web Server

Run:  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
  GET  /                      → HTML chat UI
  POST /api/query             → JSON: {question} → {answer, sources}
  GET  /api/stream            → SSE streaming answer (EventSource)
  GET  /api/case/{case_num}   → 판례 원문 조회
  GET  /api/health            → {status, ollama_ok, docs_count}
"""
from __future__ import annotations

import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from core.config import settings
from core.pipeline import RAGPipeline

# ── App setup ────────────────────────────────────────────────────────────
app = FastAPI(
    title="형사수사 매뉴얼 QA",
    description="RAG-based Q&A system for Korean criminal investigation manuals",
    version="2.0.0",
)

pipeline: RAGPipeline | None = None

@app.on_event("startup")
async def startup():
    global pipeline
    pipeline = RAGPipeline(settings)
    print("[App] RAG pipeline ready ✓")


# ── Request / Response models ────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


# ── API endpoints ────────────────────────────────────────────────────────
@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해 주세요.")
    try:
        result = await pipeline.aquery(req.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stream")
async def stream_query(q: str):
    if not q.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해 주세요.")

    async def event_generator():
        try:
            async for token in pipeline.astream_query(q):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/case/{case_number}")
async def get_case(case_number: str):
    """판례 원문 조회 — 사건번호로 DB에서 해당 판례 청크 전체를 반환."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    docs = [
        d for d in pipeline.store.documents
        if d.get("case_number") == case_number
    ]
    if not docs:
        raise HTTPException(status_code=404, detail=f"사건번호 {case_number}를 찾을 수 없습니다.")

    first = docs[0]
    full_text = "\n\n".join(d.get("text", "") for d in docs if d.get("text", "").strip())

    return {
        "case_number":   case_number,
        "case_name":     first.get("case_name", ""),
        "court":         first.get("court", ""),
        "judgment_date": first.get("judgment_date", ""),
        "reference_law": first.get("reference_law", ""),
        "text":          full_text,
    }


@app.get("/api/health")
async def health():
    ok = pipeline is not None and pipeline.generator.is_alive()
    return {
        "status":      "ok" if ok else "ollama_not_running",
        "ollama_ok":   ok,
        "ollama_model": settings.ollama_model,
        "docs_count":  len(pipeline.store) if pipeline else 0,
    }


# ── HTML UI ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>형사수사 매뉴얼 QA</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
      background: #f0f2f8;
      display: flex; flex-direction: column; height: 100vh;
      font-size: 14px; color: #222;
    }

    header {
      background: #1a237e; color: #fff;
      padding: 14px 24px;
      font-size: 1rem; font-weight: 700; letter-spacing: .4px;
      display: flex; align-items: center; gap: 10px;
    }
    header span { font-size: .78rem; font-weight: 400; opacity: .75; }

    #chat {
      flex: 1; overflow-y: auto;
      padding: 20px 24px;
      display: flex; flex-direction: column; gap: 14px;
    }

    /* ── 말풍선 ── */
    .msg-wrap { display: flex; flex-direction: column; }
    .msg-wrap.user  { align-items: flex-end; }
    .msg-wrap.bot   { align-items: flex-start; }

    .bubble {
      max-width: 80%; padding: 12px 16px;
      border-radius: 14px; line-height: 1.65;
      font-size: .9rem;
    }
    .user .bubble {
      background: #1a237e; color: #fff;
      border-bottom-right-radius: 4px;
    }
    .bot .bubble {
      background: #fff; color: #222;
      border-bottom-left-radius: 4px;
      box-shadow: 0 1px 5px rgba(0,0,0,.1);
      padding: 16px 20px;
    }

    /* ── 섹션 렌더링 ── */
    .sec-block { margin: 12px 0 0; }
    .sec-block:first-child { margin-top: 0; }

    .sec-header {
      display: block;
      font-weight: 700; font-size: .9rem;
      margin-bottom: 4px;
    }

    .sec-body { white-space: pre-wrap; line-height: 1.75; }

    /* ── 판례번호 링크 ── */
    .case-link {
      color: #1565c0; text-decoration: underline dotted;
      cursor: pointer; font-weight: 600; padding: 0 2px;
      border-radius: 3px;
    }
    .case-link:hover { background: #e3f2fd; color: #0d47a1; }

    /* ── 입력창 ── */
    #form {
      display: flex; gap: 8px; padding: 14px 24px;
      background: #fff; border-top: 1px solid #ddd;
    }
    #question {
      flex: 1; padding: 10px 14px;
      border: 1px solid #ccc; border-radius: 8px;
      font-size: .9rem; outline: none;
      font-family: inherit;
    }
    #question:focus { border-color: #1a237e; }
    button#btn {
      background: #1a237e; color: #fff; border: none;
      padding: 10px 22px; border-radius: 8px;
      cursor: pointer; font-size: .9rem; font-weight: 600;
    }
    button#btn:disabled { opacity: .45; cursor: not-allowed; }

    .typing-dot::after {
      content: '▋'; animation: blink .7s infinite;
    }
    @keyframes blink { 50% { opacity: 0; } }

    /* ── 판례 모달 ── */
    #case-modal {
      display: none; position: fixed;
      inset: 0; z-index: 999;
      background: rgba(0,0,0,.45);
      justify-content: center; align-items: flex-start;
      padding: 60px 20px 20px;
    }
    #case-modal.open { display: flex; }

    .modal-box {
      background: #fff; border-radius: 12px;
      width: 100%; max-width: 620px;
      max-height: 80vh; overflow-y: auto;
      padding: 24px 28px;
      box-shadow: 0 8px 32px rgba(0,0,0,.25);
    }
    .modal-title {
      font-size: 1rem; font-weight: 700;
      color: #1a237e; margin-bottom: 4px;
    }
    .modal-meta {
      font-size: .78rem; color: #777;
      margin-bottom: 14px; padding-bottom: 12px;
      border-bottom: 1px solid #eee;
    }
    .modal-ref {
      font-size: .78rem; color: #555;
      background: #f9f9f9; padding: 8px 10px;
      border-radius: 6px; margin-bottom: 14px;
    }
    .modal-text {
      font-size: .85rem; line-height: 1.75;
      white-space: pre-wrap; color: #333;
    }
    .modal-close {
      float: right; background: none; border: none;
      font-size: 1.3rem; cursor: pointer; color: #888;
      margin-left: 12px;
    }
    .modal-close:hover { color: #333; }
    .modal-loading { color: #aaa; font-style: italic; }
  </style>
</head>
<body>

<header>
  🔍 형사수사 매뉴얼 QA
  <span>EXAONE 3.5 · 법령 우선 검색</span>
</header>

<div id="chat"></div>

<form id="form">
  <input id="question" placeholder="예: 긴급체포 요건은? / 절도죄 구성요건과 판례" autocomplete="off"/>
  <button id="btn" type="submit">전송</button>
</form>

<!-- 판례 상세 모달 -->
<div id="case-modal">
  <div class="modal-box">
    <button class="modal-close" onclick="closeModal()">✕</button>
    <div class="modal-title" id="modal-title">판례 상세</div>
    <div class="modal-meta" id="modal-meta"></div>
    <div class="modal-ref" id="modal-ref" style="display:none"></div>
    <div class="modal-text" id="modal-text">로딩 중...</div>
  </div>
</div>

<script>
const chat    = document.getElementById('chat');
const form    = document.getElementById('form');
const qInput  = document.getElementById('question');
const btn     = document.getElementById('btn');
const modal   = document.getElementById('case-modal');

// ── 섹션 정의 ─────────────────────────────────────────────────────────
const SECTIONS = ['[답변]', '[근거법령]', '[참조판례]', '[참고사항]'];

// 한국 법원 사건번호 패턴: 연도 + 부호 + 번호
const CASE_RE = /\b(\d{2,4}(?:도|노|고합|고단|고약|고정|가합|가단|구합|구단|카합|카단|나|다))\d+\b/g;

function linkifyCaseNums(text) {
  return text.replace(CASE_RE, (match) =>
    `<span class="case-link" onclick="showCase('${match}')">${match}</span>`
  );
}

function renderAnswer(rawText) {
  // 섹션별로 쪼개기
  const result = document.createElement('div');
  let remaining = rawText;

  // 섹션 시작 위치 탐색
  const positions = [];
  for (const sec of SECTIONS) {
    let idx = remaining.indexOf(sec);
    while (idx !== -1) {
      positions.push({ idx, sec });
      idx = remaining.indexOf(sec, idx + 1);
    }
  }
  positions.sort((a, b) => a.idx - b.idx);

  if (positions.length === 0) {
    // 섹션 없으면 그냥 텍스트
    const p = document.createElement('div');
    p.innerHTML = linkifyCaseNums(escapeHtml(rawText));
    result.appendChild(p);
    return result;
  }

  // 첫 섹션 이전 텍스트
  const preText = remaining.slice(0, positions[0].idx).trim();
  if (preText) {
    const p = document.createElement('div');
    p.style.marginBottom = '8px';
    p.innerHTML = linkifyCaseNums(escapeHtml(preText));
    result.appendChild(p);
  }

  for (let i = 0; i < positions.length; i++) {
    const { idx, sec } = positions[i];
    const end = i + 1 < positions.length ? positions[i + 1].idx : remaining.length;
    const bodyRaw = remaining.slice(idx + sec.length, end).trim();

    const block = document.createElement('div');
    block.className = 'sec-block';

    const header = document.createElement('div');
    header.className = 'sec-header';
    header.textContent = sec;
    block.appendChild(header);

    const body = document.createElement('div');
    body.className = 'sec-body';
    body.innerHTML = linkifyCaseNums(escapeHtml(bodyRaw));
    block.appendChild(body);

    result.appendChild(block);
  }

  return result;
}

function escapeHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function addUserMsg(text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap user';
  const b = document.createElement('div');
  b.className = 'bubble';
  b.textContent = text;
  wrap.appendChild(b);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

function addBotBubble() {
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap bot';
  const b = document.createElement('div');
  b.className = 'bubble typing-dot';
  wrap.appendChild(b);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return b;
}

// ── 판례 모달 ────────────────────────────────────────────────────────
async function showCase(caseNum) {
  modal.classList.add('open');
  document.getElementById('modal-title').textContent = '판례: ' + caseNum;
  document.getElementById('modal-meta').textContent = '불러오는 중...';
  document.getElementById('modal-text').textContent = '';
  document.getElementById('modal-ref').style.display = 'none';

  try {
    const res = await fetch('/api/case/' + encodeURIComponent(caseNum));
    if (!res.ok) {
      document.getElementById('modal-meta').textContent = '해당 판례를 찾을 수 없습니다.';
      return;
    }
    const d = await res.json();

    document.getElementById('modal-title').textContent =
      (d.case_name || '판례') + ' (' + caseNum + ')';
    document.getElementById('modal-meta').textContent =
      [d.court, d.judgment_date].filter(Boolean).join('  |  ');

    if (d.reference_law) {
      const refEl = document.getElementById('modal-ref');
      refEl.style.display = 'block';
      refEl.textContent = '참조조문: ' + d.reference_law.replace(/【참조조문】\n?/, '').trim();
    }

    document.getElementById('modal-text').textContent = d.text;
  } catch (e) {
    document.getElementById('modal-meta').textContent = '오류: ' + e.message;
  }
}

function closeModal() {
  modal.classList.remove('open');
}
modal.addEventListener('click', (e) => {
  if (e.target === modal) closeModal();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeModal();
});

// ── 전송 ────────────────────────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = qInput.value.trim();
  if (!q) return;

  addUserMsg(q);
  qInput.value = '';
  btn.disabled = true;

  const botBubble = addBotBubble();
  let full = '';

  const es = new EventSource('/api/stream?q=' + encodeURIComponent(q));

  es.onmessage = (ev) => {
    if (ev.data === '[DONE]') {
      es.close();
      btn.disabled = false;
      // 스트리밍 완료 후 섹션 렌더링
      botBubble.classList.remove('typing-dot');
      botBubble.innerHTML = '';
      botBubble.appendChild(renderAnswer(full));
      chat.scrollTop = chat.scrollHeight;
      return;
    }
    if (ev.data.startsWith('[ERROR]')) {
      es.close();
      btn.disabled = false;
      botBubble.classList.remove('typing-dot');
      botBubble.textContent = ev.data;
      return;
    }
    full += ev.data;
    // 스트리밍 중에는 raw 텍스트 표시
    botBubble.textContent = full;
    chat.scrollTop = chat.scrollHeight;
  };

  es.onerror = () => {
    es.close();
    btn.disabled = false;
    botBubble.classList.remove('typing-dot');
    if (!full) botBubble.textContent = '오류가 발생했습니다.';
    else {
      botBubble.innerHTML = '';
      botBubble.appendChild(renderAnswer(full));
    }
  };
});
</script>
</body>
</html>
"""
