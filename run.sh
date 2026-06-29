#!/usr/bin/env bash
# =====================================================
#  RagInv  —  Start the web QA server
#  Usage:  bash run.sh
# =====================================================

# Move to project root no matter where you call this from
cd "$(dirname "$0")"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓${NC} $*"; }
warn() { echo -e "${YELLOW}  !${NC} $*"; }
err()  { echo -e "${RED}  ✗ $*${NC}"; }
hdr()  { echo -e "\n${BOLD}$*${NC}"; }

echo ""
echo -e "${BOLD}======================================${NC}"
echo -e "${BOLD}   형사수사 매뉴얼 QA  —  Web Server   ${NC}"
echo -e "${BOLD}======================================${NC}"

# ── 1. Check .venv ────────────────────────────────────────────
hdr "[1/4] Python 환경 확인"
if [[ ! -f ".venv/bin/python" ]]; then
    err ".venv 가 없습니다. 먼저 setup.sh 를 실행하세요:"
    err "  bash setup.sh"
    exit 1
fi
source .venv/bin/activate
ok "venv 활성화 ($(python --version))"

# ── 2. Check OPENAI_API_KEY ───────────────────────────────────
hdr "[2/4] API 키 확인"
if [[ -f ".env" ]]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs) 2>/dev/null
fi

if [[ -z "$OPENAI_API_KEY" ]]; then
    err "OPENAI_API_KEY 가 비어있습니다!"
    err ".env 파일에 OpenAI 키를 입력하세요:"
    err "  OPENAI_API_KEY=sk-..."
    err ""
    err "키가 없으면 검색 벡터 임베딩이 불가합니다."
    err "(현재 DB는 OpenAI ada-002 임베딩으로 구축되어 있음)"
    exit 1
fi
ok "OPENAI_API_KEY 확인됨"

# ── 3. Check / start Ollama ───────────────────────────────────
hdr "[3/4] Ollama (LLM) 확인"
MODEL="${OLLAMA_MODEL:-exaone3.5:7.8b}"

if ! command -v ollama &>/dev/null; then
    err "Ollama가 설치되어 있지 않습니다."
    err "  brew install ollama"
    exit 1
fi

# Start server if not running
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    warn "Ollama 서버 시작 중..."
    ollama serve &>/tmp/ollama.log &
    sleep 3
fi

# Check model is pulled
if ollama list 2>/dev/null | grep -q "^${MODEL}"; then
    ok "모델 준비됨: $MODEL"
else
    warn "모델 다운로드 중: $MODEL  (최초 실행 시 시간이 걸립니다)"
    ollama pull "$MODEL"
fi

# ── 4. Check DB ───────────────────────────────────────────────
hdr "[4/4] DB 확인"
if [[ ! -f "data/db/index.faiss" ]]; then
    err "data/db/index.faiss 가 없습니다."
    err "먼저 DB 마이그레이션을 실행하세요:"
    err "  python scripts/migrate_db.py --src db/ --dst data/db/"
    exit 1
fi
DOC_COUNT=$(python -c "import pickle; d=pickle.load(open('data/db/docs.pkl','rb')); print(len(d))" 2>/dev/null)
ok "DB 로드됨: ${DOC_COUNT:-?}개 문서"

# ── Launch ────────────────────────────────────────────────────
PORT="${PORT:-8000}"
echo ""
echo -e "${GREEN}${BOLD}  서버 시작 → http://localhost:${PORT}${NC}"
echo -e "  종료: Ctrl+C"
echo ""

# Open browser after 2 sec (background)
(sleep 2 && open "http://localhost:${PORT}" 2>/dev/null) &

uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload
