#!/usr/bin/env bash
# m5_sectional_pipeline.sh
# M5에서 실행 — db_sectional × 7B 모델 3개 × 전 조건 순차 실행
#
# 모니터링: tail -f ~/Desktop/RagInv/logs/m5_sectional.log

set -uo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

ROOT=~/Desktop/RagInv
PYTHON=$ROOT/.venv/bin/python3
LOG=$ROOT/logs/m5_sectional.log
GOLD=data/eval_criminal.jsonl
DB=data/db_sectional
M2_IP=192.168.149.188

mkdir -p "$ROOT/logs" "$ROOT/results"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

run_eval() {
    local pred=$1
    local tag=$2

    if [ ! -f "$ROOT/$pred" ]; then
        log "SKIP (파일 없음): $pred"
        return
    fi

    log "  generation eval..."
    $PYTHON scripts/eval_metrics.py generation \
        --pred "$pred" \
        --gold "$GOLD" \
        --out "results/generation_${tag}.json" \
        >> "$LOG" 2>&1 && log "  generation OK" || log "  generation FAIL"

    log "  hallucination eval..."
    $PYTHON scripts/eval_metrics.py hallucination \
        --pred "$pred" \
        --db "$DB" \
        --out "results/hallucination_${tag}.json" \
        >> "$LOG" 2>&1 && log "  hallucination OK" || log "  hallucination FAIL"
}

cd "$ROOT"

log "=========================================="
log "M5 db_sectional 파이프라인 시작"
log "=========================================="

# ollama serve 시작
if ! pgrep -x ollama > /dev/null; then
    log "ollama serve 시작..."
    nohup ollama serve >> "$LOG" 2>&1 &
    sleep 5
fi

declare -a MODELS=("qwen2.5:7b" "llama3.1:8b" "exaone3.5:7.8b")

for model in "${MODELS[@]}"; do
    model_tag="${model//:/_}"

    # ── standard RAG ──
    log "=== ${model} standard RAG 시작 ==="
    $PYTHON scripts/run_inference.py \
        --model "$model" --db "$DB" --eval "$GOLD" \
        >> "$LOG" 2>&1 \
        && log "=== ${model} standard 완료 ===" \
        || log "=== ${model} standard FAIL ==="
    run_eval "results/preds_${model_tag}_db_sectional.jsonl" "${model_tag}_db_sectional"

    # ── zeroshot ──
    log "=== ${model} zeroshot 시작 ==="
    $PYTHON scripts/run_inference_zeroshot.py \
        --model "$model" --db "$DB" \
        >> "$LOG" 2>&1 \
        && log "=== ${model} zeroshot 완료 ===" \
        || log "=== ${model} zeroshot FAIL ==="
    run_eval "results/preds_${model_tag}_zeroshot_db_sectional.jsonl" "${model_tag}_zeroshot_db_sectional"

    # ── fewshot ──
    log "=== ${model} fewshot 시작 ==="
    $PYTHON scripts/run_inference_fewshot.py \
        --model "$model" --db "$DB" \
        >> "$LOG" 2>&1 \
        && log "=== ${model} fewshot 완료 ===" \
        || log "=== ${model} fewshot FAIL ==="
    run_eval "results/preds_${model_tag}_fewshot_db_sectional.jsonl" "${model_tag}_fewshot_db_sectional"

    log "=== ${model} 전체 완료 ==="
done

log "=========================================="
log "M5 파이프라인 완료. 결과를 M2로 전송합니다."
log "=========================================="

# 완료 후 M2로 결과 자동 전송
rsync -avz \
    results/*_db_sectional* \
    seokiho@${M2_IP}:~/Desktop/RagInv/results/ \
    >> "$LOG" 2>&1 \
    && log "M2 전송 완료" \
    || log "M2 전송 실패 — 수동으로 rsync 필요"