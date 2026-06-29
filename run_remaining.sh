#!/bin/bash
# Qwen 완료 후 남은 실험 순차 실행
# 사용법: bash run_remaining.sh

cd ~/Desktop/RagInv
source .venv/bin/activate 2>/dev/null || true

echo "=== $(date) === Qwen 크로스 시작 ===" && \
.venv/bin/python3 scripts/run_inference_cross.py --model qwen2.5:7b && \
echo "=== $(date) === EXAONE 제로샷 시작 ===" && \
.venv/bin/python3 scripts/run_inference_zeroshot.py --model exaone3.5:7.8b --limit 300 && \
echo "=== $(date) === EXAONE 5-shot 시작 ===" && \
.venv/bin/python3 scripts/run_inference_fewshot.py --model exaone3.5:7.8b --limit 300 && \
echo "=== $(date) === LLaMA 제로샷 시작 ===" && \
.venv/bin/python3 scripts/run_inference_zeroshot.py --model llama3.1:8b --limit 300 && \
echo "=== $(date) === LLaMA 5-shot 시작 ===" && \
.venv/bin/python3 scripts/run_inference_fewshot.py --model llama3.1:8b --limit 300 && \
echo "=== $(date) === Qwen 제로샷 시작 ===" && \
.venv/bin/python3 scripts/run_inference_zeroshot.py --model qwen2.5:7b --limit 300 && \
echo "=== $(date) === Qwen 5-shot 시작 ===" && \
.venv/bin/python3 scripts/run_inference_fewshot.py --model qwen2.5:7b --limit 300 && \
echo "=== $(date) === 전체 완료 ==="
