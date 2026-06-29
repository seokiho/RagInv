#!/usr/bin/env bash
# M5에서 직접 실행: bash scripts/monitor.sh
cd "$(dirname "$0")/.."
N=1563

echo "=============================="
echo "  M5 db_sectional 진행 현황"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================="

models=("qwen2.5_7b" "llama3.1_8b" "exaone3.5_7.8b")
conds=("" "_zeroshot" "_fewshot")
cond_names=("standard" "zeroshot" "fewshot")

for model in "${models[@]}"; do
    for i in 0 1 2; do
        cond="${conds[$i]}"
        cname="${cond_names[$i]}"
        f="results/preds_${model}${cond}_db_sectional.jsonl"
        if [ -f "$f" ]; then
            cnt=$(wc -l < "$f")
            if [ "$cnt" -ge "$N" ]; then
                echo "  ✅ ${model} ${cname}: 완료 (${cnt}건)"
            else
                pct=$((cnt * 100 / N))
                echo "  🔄 ${model} ${cname}: ${cnt}/${N} (${pct}%)"
            fi
        else
            echo "  ⏳ ${model} ${cname}: 대기 중"
        fi
    done
done

echo ""
echo "프로세스:"
ps aux | grep run_inference | grep -v grep | awk '{print "  PID:"$2, $11, $12, $13}' || echo "  (없음)"
echo ""
echo "최근 로그:"
tail -3 logs/m5_sectional.log 2>/dev/null
