#!/usr/bin/env bash
# 2026-10-10 재실행: DART 합성(공식 CPU) α=0.8 → 평가 θ0/θ_tgt/θ*0.8 (xArm7) → α 스윕 0.5, 1.0 (합성→평가).
# 합성과 평가 클라이언트는 겹치지 않게 순서대로 (6코어 경쟁 시 합성이 수십 배 느려짐). θ* 는 ckpts(/dev/shm 링크)에 보관.
# libero_spatial 10 태스크 × 10 에피소드 (task 0 = adaptation, 1~9 = held-out). 결과: ~/hayeon/dart_xarm7/results/<tag>/
set -uo pipefail
source ~/hayeon/dart_xarm7/env.sh
H=~/hayeon/dart_xarm7; CK=$DART_ROOT/ckpts
T0=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
TGT=$CK/oneshot_tgt/pi05_libero_oneshotft/original/999
export TASKS="0 1 2 3 4 5 6 7 8 9" TRIALS=10
ST=$DART_ROOT/eval/status.txt; mkdir -p "$(dirname "$ST")"
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
ev() { log "EVAL $1 ($4) start"
  TAG=$1 CONFIG=$2 DIR=$3 ROBOT=$4 bash $H/run_eval_model.sh >> "$DART_ROOT/eval/run_all.log" 2>&1
  log "EVAL $1 done: $(cat $H/results/$1/summary.txt 2>/dev/null)"; }
merge() { bash $H/run_dart_merge.sh $1; log "MERGE a=$1: $(grep 'END rc=' "$DART_ROOT/logs/dart_a$1.log")"; grep -q "END rc=0" "$DART_ROOT/logs/dart_a$1.log"; }
log "=== 시작 (job $PBS_JOBID, $(hostname)) ==="
merge 0.8 || { log "merge 0.8 실패 — 중단"; exit 1; }
ev theta0 pi05_libero "$T0" XArm7
ev theta_tgt pi05_libero_oneshotft "$TGT" XArm7
ev dart_a0.8 pi05_libero "$CK/dart_a0.8" XArm7
for A in 0.5 1.0; do merge $A && ev dart_a$A pi05_libero "$CK/dart_a$A" XArm7; done
log "=== ALL_DONE ==="
