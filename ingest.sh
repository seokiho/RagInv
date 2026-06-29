#!/usr/bin/env bash
# =====================================================
#  RagInv  —  AIHUB → FAISS 인덱스 빌드
#  Usage:  bash ingest.sh [criminal|civil|all]
# =====================================================

cd "$(dirname "$0")"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓${NC} $*"; }
warn() { echo -e "${YELLOW}  !${NC} $*"; }
err()  { echo -e "${RED}  ✗ $*${NC}"; exit 1; }

JURISDICTION="${1:-criminal}"  # 기본: 형사만

echo ""
echo -e "${BOLD}=====================================${NC}"
echo -e "${BOLD}   AIHUB → FAISS 인덱스 빌드        ${NC}"
echo -e "${BOLD}   대상: ${JURISDICTION}              ${NC}"
echo -e "${BOLD}=====================================${NC}"
echo ""

# ── venv 확인 ─────────────────────────────────────────────────
if [[ ! -f ".venv/bin/python3" ]]; then
    err ".venv 없음. 먼저: bash setup.sh"
fi
ok "venv 확인됨"

# ── 예상 시간 안내 ─────────────────────────────────────────────
if [[ "$JURISDICTION" == "criminal" ]]; then
    warn "형사 데이터 약 201,744청크 — 예상 50분 (MPS 사용 시)"
elif [[ "$JURISDICTION" == "civil" ]]; then
    warn "민사 데이터 약 402,114청크 — 예상 100분 (MPS 사용 시)"
else
    warn "전체 603,858청크 — 예상 150분 (MPS 사용 시)"
    JURISDICTION=""
fi

warn "이 창을 닫으면 중단됩니다. 30배치(~960개)마다 자동 저장됩니다."
echo ""

# ── 실행 ──────────────────────────────────────────────────────
JUR_ARG=""
[[ -n "$JURISDICTION" ]] && JUR_ARG="--jurisdiction $JURISDICTION"

.venv/bin/python3 scripts/ingest_aihub.py \
    $JUR_ARG \
    --batch-size 32 \
    --sleep-ms 200 \
    --checkpoint-every 30

echo ""
ok "완료!"
