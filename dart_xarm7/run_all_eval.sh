#!/usr/bin/env bash
# cross-embodiment 평가 전체: xArm7 에서 θ0 / θ_tgt / θ*(α=0.8, 공식 CPU 합성) → 참고용 θ0 on Panda → α 스윕(0.5, 1.0, GPU 합성).
# GPU 합성은 공식 main() 을 래퍼로 부른 것 (run_dart_merge_gpu.sh). 먼저 α=0.8 을 GPU 로 다시 만들어 CPU 결과와 비교해 기록.
# 합성(공식 dart.py 는 CPU 강제)과 평가 클라이언트(OSMesa, CPU)를 겹치면 6코어 경쟁으로 합성이 수십 배 느려지므로 겹치지 않게 순서대로.
# 이미 100 에피소드가 끝난 태그는 건너뜀.
# libero_spatial 10 태스크 × 10 에피소드 (task 0 = adaptation 태스크, 1~9 = held-out).
#   setsid nohup bash ~/hayeon/dart_xarm7/run_all_eval.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
source ~/hayeon/dart_xarm7/env.sh
H=~/hayeon/dart_xarm7
CK=$DART_ROOT/ckpts
T0=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
TGT=$CK/oneshot_tgt/pi05_libero_oneshotft/original/999
export TASKS="0 1 2 3 4 5 6 7 8 9" TRIALS=10
ST=$DART_ROOT/eval/status.txt; mkdir -p "$(dirname "$ST")"
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
ev() {  # tag config dir robot
  if [ "$(cat $H/results/$1/client*.jsonl 2>/dev/null | wc -l)" -ge 100 ]; then log "EVAL $1 skip (이미 완료)"; return; fi
  log "EVAL $1 ($4) start"
  TAG=$1 CONFIG=$2 DIR=$3 ROBOT=$4 bash $H/run_eval_model.sh >> "$DART_ROOT/eval/run_all.log" 2>&1
  log "EVAL $1 done: $(cat $H/results/$1/summary.txt 2>/dev/null)"
}
wait_merge() {  # alpha
  until grep -q "END rc=" "$DART_ROOT/logs/dart_a$1.log" 2>/dev/null; do sleep 30; done
  log "MERGE a=$1: $(grep 'END rc=' "$DART_ROOT/logs/dart_a$1.log")"
  grep -q "END rc=0" "$DART_ROOT/logs/dart_a$1.log"
}

log "=== 평가 시작 ==="
ev theta0 pi05_libero "$T0" XArm7
wait_merge 0.8 || { log "merge 0.8 실패 — 중단"; exit 1; }
# GPU 합성 검증: α=0.8 을 GPU 로 다시 만들어 공식 CPU 결과와 비교 → 사본 삭제
bash $H/run_dart_merge_gpu.sh 0.8 _gpu
if wait_merge 0.8_gpu; then
  ( cd "$DART_REPO" && JAX_PLATFORMS=cpu .venv/bin/python $H/compare_merge.py "$CK/dart_a0.8" "$CK/dart_a0.8_gpu" "$T0" ) \
    > "$DART_ROOT/eval/compare_cpu_gpu_a0.8.txt" 2>&1
  log "CPU vs GPU (α=0.8): $(grep -E '비트 단위 동일|전체 δ 상대차' "$DART_ROOT/eval/compare_cpu_gpu_a0.8.txt" | tr '\n' ' ')"
  cp "$DART_ROOT/eval/compare_cpu_gpu_a0.8.txt" $H/results/ 2>/dev/null
  GPU_OK=1
else
  log "GPU 합성 실패 → 스윕은 CPU 합성으로"; GPU_OK=0
fi
rm -rf "$CK/dart_a0.8_gpu"
ev theta_tgt pi05_libero_oneshotft "$TGT" XArm7
ev dart_a0.8 pi05_libero "$CK/dart_a0.8" XArm7
ev theta0_panda pi05_libero "$T0" Panda
# α 스윕: 디스크를 위해 하나씩 합성 → 평가 → 삭제
for A in 0.5 1.0; do
  if [ "$GPU_OK" = 1 ]; then bash $H/run_dart_merge_gpu.sh $A; else bash $H/run_dart_merge.sh $A; fi
  wait_merge $A && ev dart_a$A pi05_libero "$CK/dart_a$A" XArm7
  rm -rf "$CK/dart_a$A"; log "deleted dart_a$A (free $(df -BG --output=avail /var/tmp | tail -1 | tr -dc 0-9)G)"
done
log "=== ALL_DONE ==="
