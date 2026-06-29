#!/usr/bin/env bash
# auto_pipeline.sh
# PID 14852 (zeroshot/fewshot 6개 잡) 완료 후 자동 순차 실행
#
# 실행 순서:
#   1. 7B 모델 zeroshot/fewshot eval_metrics (generation + hallucination)
#   2. 신규 모델 (qwen3:32b, exaone4.5:33b) standard RAG inference + eval
#   3. 신규 모델 zeroshot inference + eval
#   4. 신규 모델 fewshot inference + eval
#
# 모니터링: tail -f logs/auto_pipeline.log

set -uo pipefail

ROOT=/Users/seokiho/Desktop/RagInv
PYTHON=$ROOT/.venv/bin/python3
LOG=$ROOT/logs/auto_pipeline.log
GOLD=data/eval_criminal.jsonl
DB=data/db_aihub
WAIT_PID=14852

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

run_eval() {
    local pred=$1
    local model_tag=$2
    local prompt_tag=$3
    local suffix="${prompt_tag:+_${prompt_tag}}"

    if [ ! -f "$ROOT/$pred" ]; then
        log "SKIP (파일 없음): $pred"
        return
    fi

    log "  generation eval: $pred"
    $PYTHON scripts/eval_metrics.py generation \
        --pred "$pred" \
        --gold "$GOLD" \
        --out "results/generation_${model_tag}${suffix}_db_aihub.json" \
        >> "$LOG" 2>&1 && log "  generation OK" || log "  generation FAIL"

    log "  hallucination eval: $pred"
    $PYTHON scripts/eval_metrics.py hallucination \
        --pred "$pred" \
        --db "$DB" \
        --out "results/hallucination_${model_tag}${suffix}_db_aihub.json" \
        >> "$LOG" 2>&1 && log "  hallucination OK" || log "  hallucination FAIL"
}

cd "$ROOT"

# ── 1. 현재 잡 완료 대기 ────────────────────────────────────────────────────
log "=========================================="
log "auto_pipeline 시작. PID ${WAIT_PID} 완료 대기 중..."
log "=========================================="

while kill -0 "$WAIT_PID" 2>/dev/null; do
    DONE=$(wc -l < results/preds_qwen2.5_7b_zeroshot_db_aihub.jsonl 2>/dev/null || echo 0)
    log "  대기 중... qwen2.5:7b zeroshot ${DONE}/1563 완료"
    sleep 300
done

log "PID ${WAIT_PID} 완료. 파이프라인 시작."

# ── 2. 7B 모델 eval_metrics ─────────────────────────────────────────────────
log "=========================================="
log "STEP 1: 7B 모델 zeroshot/fewshot eval_metrics"
log "=========================================="

declare -a SMALL_MODELS=("qwen2.5_7b" "llama3.1_8b" "exaone3.5_7.8b")
declare -a PROMPTS=("zeroshot" "fewshot")

for model_tag in "${SMALL_MODELS[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        log "--- ${model_tag} ${prompt} ---"
        run_eval "results/preds_${model_tag}_${prompt}_db_aihub.jsonl" "$model_tag" "$prompt"
    done
done

log "STEP 1 완료."

# ── 3. 신규 모델 inference + eval ───────────────────────────────────────────
log "=========================================="
log "STEP 2: 신규 모델 (qwen3:32b, exaone4.5:33b) db_aihub inference"
log "=========================================="

declare -a NEW_MODELS=("qwen3:32b" "exaone4.5:33b")

for model in "${NEW_MODELS[@]}"; do
    model_tag="${model//:/_}"

    # ── standard RAG ──
    log "=== ${model} standard RAG 시작 ==="
    $PYTHON scripts/run_inference.py \
        --model "$model" \
        --db "$DB" \
        --eval "$GOLD" \
        --top-k 10 \
        >> "$LOG" 2>&1 && log "=== ${model} standard RAG 완료 ===" || log "=== ${model} standard RAG FAIL ==="

    run_eval "results/preds_${model_tag}_db_aihub.jsonl" "$model_tag" ""

    # ── zeroshot ──
    log "=== ${model} zeroshot 시작 ==="
    $PYTHON scripts/run_inference_zeroshot.py \
        --model "$model" \
        --db "$DB" \
        --top-k 10 \
        >> "$LOG" 2>&1 && log "=== ${model} zeroshot 완료 ===" || log "=== ${model} zeroshot FAIL ==="

    run_eval "results/preds_${model_tag}_zeroshot_db_aihub.jsonl" "$model_tag" "zeroshot"

    # ── fewshot ──
    log "=== ${model} fewshot 시작 ==="
    $PYTHON scripts/run_inference_fewshot.py \
        --model "$model" \
        --db "$DB" \
        --top-k 10 \
        >> "$LOG" 2>&1 && log "=== ${model} fewshot 완료 ===" || log "=== ${model} fewshot FAIL ==="

    run_eval "results/preds_${model_tag}_fewshot_db_aihub.jsonl" "$model_tag" "fewshot"

    log "=== ${model} 전체 완료 ==="
done

log "=========================================="
log "전체 파이프라인 완료."
log "다음 단계: LoRA 학습, db_sectional inference"
log "=========================================="