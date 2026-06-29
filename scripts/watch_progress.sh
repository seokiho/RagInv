#!/usr/bin/env bash
# 실시간 추론 진행 현황 — watch -n 10 bash scripts/watch_progress.sh
cd "$(dirname "$0")/.."
N=1563

print_file() {
  local model=$1 cond=$2 cname=$3 db=$4
  local f="results/preds_${model}${cond}_${db}.jsonl"
  if [ -f "$f" ]; then
    cnt=$(wc -l < "$f")
    if [ "$cnt" -ge "$N" ]; then
      printf "  ✅ %-28s %-10s %s\n" "${model}" "${cname}" "완료 (${cnt}건)"
    else
      local bar_len=20
      local filled=$((cnt * bar_len / N))
      local bar=$(printf '%0.s█' $(seq 1 $filled))$(printf '%0.s░' $(seq 1 $((bar_len - filled))))
      local pct=$((cnt * 100 / N))
      printf "  🔄 %-28s %-10s [%s] %d/%d (%d%%)\n" "${model}" "${cname}" "${bar}" "${cnt}" "${N}" "${pct}"
    fi
  else
    printf "  ⏳ %-28s %-10s 대기 중\n" "${model}" "${cname}"
  fi
}

echo "══════════════════════════════════════════════════"
echo "  추론 진행 현황   $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════"

echo ""
echo "[ M2 — db_aihub ]"
for model in qwen2.5_7b llama3.1_8b exaone3.5_7.8b; do
  print_file $model ""          "standard" "db_aihub"
  print_file $model "_zeroshot" "zeroshot" "db_aihub"
  print_file $model "_fewshot"  "fewshot"  "db_aihub"
  print_file $model "_cross"    "cross"    "db_aihub"
done

echo ""
echo "[ M5 — db_sectional (SSH) ]"
M5_RESULTS=$(SSH_AUTH_SOCK="" ssh -o IdentitiesOnly=yes -o ConnectTimeout=3 \
  -i /Users/seokiho/.ssh/id_ed25519 seokiho@192.168.145.93 \
  "wc -l ~/Desktop/RagInv/results/preds_*_db_sectional.jsonl 2>/dev/null" 2>/dev/null)
if [ -z "$M5_RESULTS" ]; then
  echo "  (M5 연결 안 됨)"
else
  for model in qwen2.5_7b llama3.1_8b exaone3.5_7.8b; do
    for cond in "" "_zeroshot" "_fewshot"; do
      fname="preds_${model}${cond}_db_sectional.jsonl"
      cnt=$(echo "$M5_RESULTS" | grep "$fname" | awk '{print $1}')
      cname="standard"; [ "$cond" = "_zeroshot" ] && cname="zeroshot"; [ "$cond" = "_fewshot" ] && cname="fewshot"
      if [ -z "$cnt" ]; then
        printf "  ⏳ %-28s %-10s 대기 중\n" "${model}" "${cname}"
      elif [ "$cnt" -ge "$N" ]; then
        printf "  ✅ %-28s %-10s 완료 (%s건)\n" "${model}" "${cname}" "${cnt}"
      else
        pct=$((cnt * 100 / N))
        printf "  🔄 %-28s %-10s %s/%s (%d%%)\n" "${model}" "${cname}" "${cnt}" "${N}" "${pct}"
      fi
    done
  done
fi

echo ""
echo "[ 실행 중인 프로세스 ]"
ps aux | grep -E "run_inference" | grep -v grep \
  | awk '{for(i=11;i<=NF;i++) printf $i" "; print ""}' \
  | sed 's|.*/||' | head -5 || echo "  (없음)"

echo ""
echo "[ 최근 로그 ]"
if [ -f logs/inference_zeroshot_fewshot.log ]; then
  tail -2 logs/inference_zeroshot_fewshot.log | grep -v "^$" | sed 's/^/  /'
elif [ -f logs/pipeline.log ]; then
  tail -2 logs/pipeline.log | grep -v "^$" | sed 's/^/  /'
fi

echo ""
echo "  갱신: watch -n 10 bash scripts/watch_progress.sh"
