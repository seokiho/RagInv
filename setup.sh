#!/usr/bin/env bash
# =============================================================
# RagInv  —  one-time project setup
# Usage:  bash setup.sh
# =============================================================
set -e

# ── 0. Colour helpers ─────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn] ${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ── 1. Locate Python (prefer 3.11/3.12 over 3.14) ────────────
for py in python3.12 python3.11 python3.10 python3; do
    if command -v "$py" &>/dev/null; then
        PYTHON=$py; break
    fi
done
[[ -z "$PYTHON" ]] && error "No Python 3.10+ found."
info "Using: $($PYTHON --version) at $(which $PYTHON)"

# ── 2. Create / reuse virtual-env ────────────────────────────
VENV=".venv"
if [[ -d "$VENV" ]]; then
    warn ".venv already exists — skipping creation"
else
    # Try uv first (faster), fall back to venv
    if command -v ~/.local/bin/uv &>/dev/null; then
        info "Creating venv with uv..."
        ~/.local/bin/uv venv "$VENV" --python "$PYTHON"
    else
        info "Creating venv with python -m venv..."
        "$PYTHON" -m venv "$VENV"
    fi
    info "Virtual environment created at $VENV"
fi

# Activate
source "$VENV/bin/activate"
info "Activated: $(python --version)"

# ── 3. Install packages ───────────────────────────────────────
info "Installing requirements..."
if command -v ~/.local/bin/uv &>/dev/null; then
    ~/.local/bin/uv pip install -r requirements.txt
else
    pip install --upgrade pip
    pip install -r requirements.txt
fi
info "Packages installed ✓"

# ── 4. Check Ollama ──────────────────────────────────────────
info "Checking Ollama..."
if ! command -v ollama &>/dev/null; then
    warn "Ollama not found. Install it:"
    warn "  brew install ollama"
    warn "  Then run:  ollama pull llama3.1:8b"
else
    info "Ollama found: $(ollama --version 2>/dev/null || echo 'version unknown')"
    # Check if the model is pulled
    if ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
        info "llama3.1:8b model ✓"
    else
        warn "llama3.1:8b not pulled yet. Run: ollama pull llama3.1:8b"
    fi
fi

# ── 5. Create directories ────────────────────────────────────
info "Creating project directories..."
mkdir -p data/raw data/db results

# ── 6. Migrate existing LangChain FAISS DB ───────────────────
DB_RAW="db/index.faiss"       # original LangChain DB (upper-level db/ folder)
DB_NATIVE="data/db/index.faiss"

if [[ -f "$DB_RAW" && ! -f "$DB_NATIVE" ]]; then
    info "Found existing LangChain FAISS DB at db/ — migrating to data/db/..."
    python scripts/migrate_db.py --src db/ --dst data/db/
    info "Migration complete ✓"
elif [[ -f "$DB_NATIVE" ]]; then
    info "data/db/ already exists — skipping migration"
else
    warn "No DB found at db/ or data/db/. Run scripts/ingest.py after adding docs to data/raw/"
fi

# ── 7. Copy .env if missing ──────────────────────────────────
if [[ ! -f ".env" ]]; then
    warn ".env not found. Create it with your API keys (see .env.example)"
fi

# ── 8. Done ──────────────────────────────────────────────────
echo ""
info "====================================================="
info "  Setup complete!"
info "====================================================="
echo ""
echo "  Activate venv:       source .venv/bin/activate"
echo ""
echo "  Start web server:    uvicorn app.main:app --reload"
echo "  Run research batch:  python research/batch_run.py --questions data/raw/questions.json"
echo "  Migrate LangChain DB: python scripts/migrate_db.py --src db/ --dst data/db/"
echo ""
