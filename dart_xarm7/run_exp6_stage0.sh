#!/usr/bin/env bash
# 실험 6 단계 0 — v0 (white, panda 동역학) 에서 θ* α=0.5 의 모듈 마스크 평가. 서버는 모두 serve_masked.py (run_eval_masked.sh).
#   bash run_exp6_stage0.sh check     정합성: task 0 × 10 에피소드, modules none (= θ0) / all (= θ*) → v0_check_none / v0_check_all
#   setsid nohup bash run_exp6_stage0.sh main > /dev/null 2>&1 < /dev/null &
#                                     본 평가 6개 순차 (서버 1개씩): 마스크 4개 + none(v0_theta0) + all(v0_dart_a0.5)
# 진행: $DART_ROOT/eval/status_exp6.txt, 실행 로그: $DART_ROOT/eval/run_exp6.log
set -uo pipefail
MODE=${1:?check | main}
source ~/hayeon/dart_xarm7/env.sh
H=~/hayeon/dart_xarm7
T0=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero   # θ0 (pi05_libero, fp32)
TS=$DART_ROOT/ckpts/dart_a0.5                                 # v0 θ* α=0.5 (HF 백업 dart_a0.5/, bf16)
ST=$DART_ROOT/eval/status_exp6.txt; mkdir -p "$(dirname "$ST")"
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
ev() {  # ev <태그> <modules>
  log "EVAL $1 (v0, modules=$2, tasks=${TASKS:-0..9}) start"
  TAG=$1 COND=v0 BASE=$T0 TUNED=$TS MODULES=$2 bash $H/run_eval_masked.sh >> "$DART_ROOT/eval/run_exp6.log" 2>&1
  log "EVAL $1 rc=$?: $(cat $H/results/$1/summary.txt 2>/dev/null)"
}
log "=== stage0 $MODE 시작 (job $PBS_JOBID, $(hostname)) ==="
case $MODE in
  check) export TASKS=0 TRIALS=10
         ev v0_check_none none
         ev v0_check_all all ;;
  main)  ev v0_mask_vision vision
         ev v0_mask_vlm vlm
         ev v0_mask_expert expert
         ev v0_mask_visionvlm vision,vlm
         ev v0_theta0 none
         ev v0_dart_a0.5 all ;;
  *) echo "MODE=$MODE (check | main)"; exit 2 ;;
esac
log "=== stage0 $MODE ALL_DONE ==="
