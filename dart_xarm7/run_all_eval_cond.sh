#!/usr/bin/env bash
# 실험 6 단계 3 — 조건 하나의 7개 모델을 그 조건의 환경(색·동역학)에서 순차 평가 (run_all_eval_v2.sh 참고).
#   theta0    → serve_ckpt.py, config pi05_libero,           dir θ0              (run_eval_model.sh)
#   theta_tgt → serve_ckpt.py, config pi05_libero_oneshotft, dir ckpts/<cond>/oneshot_tgt/…/999
#   dart_a0.5 → serve_ckpt.py, config pi05_libero,           dir ckpts/<cond>/dart_a0.5
#   mask_{vision,vlm,expert,visionvlm} → run_eval_masked.sh (BASE=θ0, TUNED=ckpts/<cond>/dart_a0.5)
# 7개 모두 exp6_cond.sh 의 그 조건 XARM7_COLOR·XARM7_DYNAMICS 를 export 한 셸에서 띄운다 (클라이언트가 상속, 마스크 러너는 COND 로 같은 값).
# libero_spatial 10 태스크 × 10 에피소드, seed 7 (eval_xarm7.py 기본), 클라이언트 5개. 태그 <cond>_<model>.
# 이미 100 에피소드가 끝난 태그는 건너뛴다 (잡이 끝나 다시 돌릴 때 이어서). 끝나면 summary 생성 + results/<cond>_* 를 git commit.
#   setsid nohup bash -c 'for c in v1 v2 v3; do bash ~/hayeon/dart_xarm7/run_all_eval_cond.sh $c; done' > /dev/null 2>&1 < /dev/null &
# 진행: $DART_ROOT/eval/status_exp6.txt, 실행 로그 $DART_ROOT/eval/run_exp6_<cond>.log
set -uo pipefail
C=${1:?cond (v1 | v2 | v3)}
[ "$C" = v0 ] && { echo "v0 는 단계 0 에서 평가함"; exit 2; }
source ~/hayeon/dart_xarm7/env.sh
source ~/hayeon/dart_xarm7/exp6_cond.sh
exp6_env "$C" || exit 2   # 이 조건의 XARM7_COLOR / XARM7_DYNAMICS export → 아래 모든 평가의 클라이언트가 상속
H=~/hayeon/dart_xarm7; CK=$DART_ROOT/ckpts
T0=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
TGT=$CK/$C/oneshot_tgt/pi05_libero_oneshotft/original/999
TS=$CK/$C/dart_a0.5
export TASKS="0 1 2 3 4 5 6 7 8 9" TRIALS=10 NCLIENT=5
ST=$DART_ROOT/eval/status_exp6.txt; RL=$DART_ROOT/eval/run_exp6_$C.log; mkdir -p "$(dirname "$ST")"
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }

for d in "$T0/params" "$TGT/params" "$TS/params"; do [ -d "$d" ] || { log "$C: 체크포인트 없음 $d — 중단"; exit 1; }; done
if ps -eo args | grep -qE "[d]omain_arithmetic|[s]cripts/train.py"; then log "$C: 합성/FT 가 실행 중 — 평가 안 함"; exit 1; fi
( source $H/client_env.sh; $LIBERO_PY -c "
import os, libero_xarm7 as X
assert (X.COLOR, X.DYNAMICS) == (os.environ['XARM7_COLOR'], os.environ['XARM7_DYNAMICS']), (X.COLOR, X.DYNAMICS)
print('client env OK', X.COLOR, X.DYNAMICS)" 2>&1 | tail -1 ) | tee -a "$ST" | grep -q "client env OK" || { log "$C: 클라이언트 env 확인 실패 — 중단"; exit 1; }

done_already() { [ "$(cat $H/results/$1/client*.jsonl 2>/dev/null | wc -l)" -eq 100 ] && grep -q "/100" $H/results/$1/summary.txt 2>/dev/null; }

ev_ckpt() {  # ev_ckpt <model> <config> <dir> — serve_ckpt.py 경로 (run_eval_model.sh)
  local tag=${C}_$1 R=$H/results/${C}_$1
  done_already "$tag" && { log "EVAL $tag 이미 완료 — 건너뜀"; return 0; }
  log "EVAL $tag ($C: $XARM7_COLOR/$XARM7_DYNAMICS, serve_ckpt $2) start"
  mkdir -p "$R"; local t0=$(date +%s)
  printf 'tag=%s\ncond=%s XARM7_COLOR=%s XARM7_DYNAMICS=%s\nserver=serve_ckpt config=%s dir=%s\nsuite=libero_spatial tasks=%s trials=%s nclient=%s\njob=%s host=%s start=%s\n' \
    "$tag" "$C" "$XARM7_COLOR" "$XARM7_DYNAMICS" "$2" "$3" "$TASKS" "$TRIALS" "$NCLIENT" "$PBS_JOBID" "$(hostname)" "$(date '+%F %T')" > "$R/meta.txt"
  EVAL_CKPT_INFO="serve_ckpt config=$2 dir=$3" TAG=$tag CONFIG=$2 DIR=$3 ROBOT=XArm7 bash $H/run_eval_model.sh >> "$RL" 2>&1
  local rc=$?
  echo "end=$(date '+%F %T') wall=$(( $(date +%s) - t0 ))s rc=$rc" >> "$R/meta.txt"
  log "EVAL $tag rc=$rc: $(cat $R/summary.txt 2>/dev/null)"
}
ev_mask() {  # ev_mask <model> <modules> — serve_masked.py 경로 (run_eval_masked.sh, COND 로 같은 env)
  local tag=${C}_$1
  done_already "$tag" && { log "EVAL $tag 이미 완료 — 건너뜀"; return 0; }
  log "EVAL $tag ($C: $XARM7_COLOR/$XARM7_DYNAMICS, serve_masked modules=$2) start"
  OVERWRITE=1 TAG=$tag COND=$C BASE=$T0 TUNED=$TS MODULES=$2 bash $H/run_eval_masked.sh >> "$RL" 2>&1
  log "EVAL $tag rc=$?: $(cat $H/results/$tag/summary.txt 2>/dev/null)"
}

log "=== stage3 $C 시작 (job $PBS_JOBID, $(hostname), XARM7_COLOR=$XARM7_COLOR XARM7_DYNAMICS=$XARM7_DYNAMICS) ==="
ev_ckpt theta0 pi05_libero "$T0"
ev_ckpt theta_tgt pi05_libero_oneshotft "$TGT"
ev_ckpt dart_a0.5 pi05_libero "$TS"
ev_mask mask_vision vision
ev_mask mask_vlm vlm
ev_mask mask_expert expert
ev_mask mask_visionvlm vision,vlm

TAGS="${C}_theta0 ${C}_theta_tgt ${C}_dart_a0.5 ${C}_mask_vision ${C}_mask_vlm ${C}_mask_expert ${C}_mask_visionvlm"
SUMMARY_JSON=$H/results/${C}_summary.json /usr/bin/python3 $H/summarize_eval.py $TAGS > $H/results/${C}_summary.txt 2>&1
/usr/bin/python3 $H/summarize_exp6.py --cond "$C" > $H/results/${C}_exp6_tables.md 2>&1
log "summary: results/${C}_summary.txt, results/${C}_exp6_tables.md"
( cd ~/hayeon && git add dart_xarm7/results/${C}_* dart_xarm7/results/exp6_summary.json && git commit -q -F - <<EOF
실험 6 단계 3: $C ($XARM7_COLOR, $XARM7_DYNAMICS) 평가 결과 7개 + 요약

$(cat $H/results/${C}_*/summary.txt)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
) && log "git commit $(git -C ~/hayeon log --oneline -1)" || log "git commit 실패"
log "=== stage3 $C ALL_DONE ==="
