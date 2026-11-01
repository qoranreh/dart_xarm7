#!/usr/bin/env bash
# 실험 4-후속1 — t4·t5 정답 궤적 실행 검사 러너 (학습·합성·정책 평가 없음). 단계별로:
#   bash ~/hayeon/dart_xarm7/run_oracle_t45.sh check0     # t0 demo_0 재현 확인 (DEMO_PAIR.md 값과 비교용) → demos/oracle_t0_check/demo0
#   bash ~/hayeon/dart_xarm7/run_oracle_t45.sh panda      # 2번: Panda 재생 task × demo
#   bash ~/hayeon/dart_xarm7/run_oracle_t45.sh init       # 3번: 시작 장면 Panda | xArm7 → results/oracle_t45/init_t<N>.png
#   bash ~/hayeon/dart_xarm7/run_oracle_t45.sh retarget   # 4번: xArm7 retarget task × demo × {quantile, none}
# 조건 env = exp6_cond.sh 의 v0 (white / panda). 출력 $DART_ROOT/demos/oracle_t<N>/demo<k>/. 출력 npz 가 이미 있으면 그 실행은 건너뜀 (덮어쓰기 없음).
# 서버(PBS 잡 안)면 env.sh + client_env.sh, 아니면 env_laptop.sh. TASKS / DEMOS / P(동시 실행 수) 로 범위 조절.
set -uo pipefail
H=~/hayeon/dart_xarm7
if [ -n "${PBS_JOBID:-}" ]; then source $H/env.sh && source $H/client_env.sh; else source $H/env_laptop.sh; fi
source $H/exp6_cond.sh && exp6_env v0 || exit 2
TASKS=${TASKS:-"4 5"}; DEMOS=${DEMOS:-"0 1 2 3 4"}; P=${P:-6}
L=$DART_ROOT/logs/oracle_t45; mkdir -p "$L"
ST=$L/status.txt
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
npz() { echo "$DART_ROOT/demos/spatial_t$1_hdf5_demo0-4.npz"; }
run1() {  # run1 <로그 이름> <출력 파일> <명령...>
  local name=$1 outf=$2; shift 2
  if [ -e "$outf" ]; then log "$name SKIP (이미 있음: $outf)"; return 0; fi
  local t0=$(date +%s); "$@" > "$L/$name.log" 2>&1; local rc=$?
  log "$name rc=$rc $(( $(date +%s) - t0 ))s"
}
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do wait -n; done; }
cd "$H"
log "=== $1 시작 ($(hostname), ${PBS_JOBID:-잡 없음}, XARM7_COLOR=$XARM7_COLOR XARM7_DYNAMICS=$XARM7_DYNAMICS MUJOCO_GL=$MUJOCO_GL, TASKS=$TASKS DEMOS=$DEMOS P=$P) ==="
case $1 in
  check0)
    O=$DART_ROOT/demos/oracle_t0_check/demo0; T=libero_spatial_t0_demo_0
    run1 check0_panda "$O/${T}_Panda.npz" $LIBERO_PY make_demo_pair.py --robots Panda --task 0 --demo demo_0 --hdf5-npz "$(npz 0)" --out "$O"
    for RC in quantile none; do
      S=$([ $RC = none ] && echo XArm7_retarget || echo XArm7_retarget_rotq)
      run1 check0_retarget_$RC "$O/${T}_$S.npz" $LIBERO_PY make_retarget_demo.py --task 0 --demo demo_0 --hdf5-npz "$(npz 0)" --out "$O" --rot-clip $RC &
    done ;;
  panda)
    for N in $TASKS; do for K in $DEMOS; do throttle
      O=$DART_ROOT/demos/oracle_t$N/demo$K
      run1 panda_t${N}_demo$K "$O/libero_spatial_t${N}_demo_${K}_Panda.npz" \
        $LIBERO_PY make_demo_pair.py --robots Panda --task $N --demo demo_$K --hdf5-npz "$(npz $N)" --out "$O" &
    done; done ;;
  init)
    for N in $TASKS; do throttle
      run1 init_t$N "$H/results/oracle_t45/init_t$N.png" $LIBERO_PY oracle_init_view.py --task $N --hdf5-npz "$(npz $N)" &
    done ;;
  retarget)
    for N in $TASKS; do for K in $DEMOS; do for RC in quantile none; do throttle
      O=$DART_ROOT/demos/oracle_t$N/demo$K
      S=$([ $RC = none ] && echo XArm7_retarget || echo XArm7_retarget_rotq)
      [ -e "$O/libero_spatial_t${N}_demo_${K}_Panda.npz" ] || { log "retarget_t${N}_demo${K}_$RC 거부: Panda npz 없음 (panda 단계 먼저)"; continue; }
      run1 retarget_t${N}_demo${K}_$RC "$O/libero_spatial_t${N}_demo_${K}_$S.npz" \
        $LIBERO_PY make_retarget_demo.py --task $N --demo demo_$K --hdf5-npz "$(npz $N)" --out "$O" --rot-clip $RC &
    done; done; done ;;
  *) echo "사용법: $0 check0 | panda | init | retarget" >&2; exit 2 ;;
esac
wait
log "=== $1 끝 ==="
