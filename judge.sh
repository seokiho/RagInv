#!/usr/bin/env bash
# =====================================================
#  LLM-as-a-Judge  —  Evaluate RAG results
#  Usage:  bash judge.sh  [results/run_*.json]
# =====================================================
cd "$(dirname "$0")"
source .venv/bin/activate
export $(grep -v '^#' .env | grep -v '^$' | xargs) 2>/dev/null

# ── Find results file ──────────────────────────────────────────
if [[ -n "$1" ]]; then
    RESULTS="$1"
else
    # Auto-pick the most recent run_*.json
    RESULTS=$(ls -t results/run_*.json 2>/dev/null | head -1)
fi

if [[ -z "$RESULTS" || ! -f "$RESULTS" ]]; then
    echo "사용법: bash judge.sh [results/run_*.json]"
    echo ""
    echo "results/ 폴더에 batch_run.py 결과 파일이 없습니다."
    echo "먼저 실행하세요:"
    echo "  python research/batch_run.py --questions data/raw/questions.json"
    exit 1
fi

echo ""
echo "======================================="
echo "  LLM-as-a-Judge 평가 시작"
echo "======================================="
echo "  입력 파일: $RESULTS"

# ── Pick backend ──────────────────────────────────────────────
if [[ -n "$OPENAI_API_KEY" ]]; then
    BACKEND="openai"
    echo "  Judge 모델: GPT-4o-mini (OpenAI)"
elif [[ -n "$CLAUDE_API_KEY" ]]; then
    BACKEND="claude"
    echo "  Judge 모델: Claude Sonnet (Anthropic)"
else
    BACKEND="ollama"
    echo "  Judge 모델: qwen2.5:14b (Ollama, 로컬)"
fi
echo ""

# ── Optional: sample mode for quick test ──────────────────────
SAMPLE_FLAG=""
if [[ "$2" == "--sample" && -n "$3" ]]; then
    SAMPLE_FLAG="--sample $3"
    echo "  샘플 모드: 처음 $3 개만 평가"
fi

python research/llm_judge.py \
    --results  "$RESULTS" \
    --output   results/ \
    --backend  "$BACKEND" \
    $SAMPLE_FLAG

echo ""
echo "완료 ✓  결과: results/judge_*.csv"
