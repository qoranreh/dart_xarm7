#!/usr/bin/env bash
# 실험 6 단계 1 — 조건별 xArm7 타깃 데모 (v0 와 같은 명령: make_retarget_demo.py --rot-clip quantile, 나머지 인자 기본값).
#   1) Panda 데모 녹화 (make_demo_pair.py --robots Panda) → $DART_ROOT/demos/v0/  (Panda 는 XARM7_* 와 무관)
#   2) v0 재현 (white, panda) → demos/v0/   — 이전 v0 데이터셋과 같은지 확인용
#   3) v1, v2, v3 → demos/<cond>/ (Panda npz 는 v0 것을 복사). 조건 env 는 exp6_cond.sh 에서만.
# 사용: setsid nohup bash run_exp6_stage1_demos.sh [cond ...] > /dev/null 2>&1 < /dev/null &   (기본: v0 v1 v2 v3)
# 진행: $DART_ROOT/logs/exp6/stage1_status.txt, 로그 $DART_ROOT/logs/exp6/demo_<cond>.log
set -uo pipefail
source ~/hayeon/dart_xarm7/env.sh
source ~/hayeon/dart_xarm7/client_env.sh
source ~/hayeon/dart_xarm7/exp6_cond.sh
H=~/hayeon/dart_xarm7; D=$DART_ROOT/demos; L=$DART_ROOT/logs/exp6; mkdir -p "$L"
ST=$L/stage1_status.txt
TAG=libero_spatial_t0_demo_0
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
CONDS=${*:-v0 v1 v2 v3}
log "=== stage1 데모 시작 (job $PBS_JOBID, $(hostname)) conds: $CONDS ==="

if [ ! -f "$D/v0/${TAG}_Panda.npz" ]; then
  t0=$(date +%s)
  ( exp6_env v0 && $LIBERO_PY $H/make_demo_pair.py --robots Panda --out "$D/v0" ) > "$L/demo_panda.log" 2>&1
  log "Panda 녹화 rc=$? $(( $(date +%s) - t0 ))s: $(grep -o '^Panda {.*}' "$L/demo_panda.log")"
fi
for c in $CONDS; do
  mkdir -p "$D/$c"
  [ "$c" = v0 ] || cp -p "$D/v0/${TAG}_Panda.npz" "$D/$c/"
  t0=$(date +%s)
  ( exp6_env "$c" && echo "env: XARM7_COLOR=$XARM7_COLOR XARM7_DYNAMICS=$XARM7_DYNAMICS" &&
    $LIBERO_PY $H/make_retarget_demo.py --rot-clip quantile --out "$D/$c" ) > "$L/demo_$c.log" 2>&1
  rc=$?
  S=$D/$c/${TAG}_XArm7_retarget_rotq_summary.json
  log "$c rc=$rc $(( $(date +%s) - t0 ))s: $(/usr/bin/python3 -c "import json; s=json.load(open('$S')); print(s['method'].split('[')[-1].rstrip(']'), '| success', s['xarm_success_at_end'], 'first', s['xarm_first_success_step'])" 2>/dev/null || echo 'summary 없음')"
done
log "=== stage1 데모 ALL_DONE ==="
