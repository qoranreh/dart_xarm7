#!/usr/bin/env bash
# [실험 4-후속3 a5] FT(run_oneshot_ft_a5.sh) 가 끝나면 → 합성(run_dart_merge_a5.sh 0.5, 공식 CPU 경로) → HF a5/dart_a0.5 백업
#   → 평가 6개 (순차, exp6_env v0, seed 7 = eval_xarm7.py 기본, 클라이언트 5개) → results/a5_* · 로그 git commit.
# FT 와 같은 잡에서 띄우면 FT status 의 ALL_DONE rc=0 을 기다렸다가 시작한다 (FT 중 평가 금지, 합성·평가 동시 실행 금지 — 모두 순차).
# 다른 잡(A6000)에서는 FROM_HF=1: 홈의 logs/oneshot_ft_a5/status.txt 에 src·tgt HF 백업 완료가 찍히면 HF a5/oneshot_{src,tgt} 를 받아 시작.
#   (A6000 잡: restore_env.sh 로 복원 후) FROM_HF=1 setsid nohup bash ~/hayeon/dart_xarm7/run_a5_after_ft.sh > /dev/null 2>&1 < /dev/null &
#   setsid nohup bash ~/hayeon/dart_xarm7/run_a5_after_ft.sh > /dev/null 2>&1 < /dev/null &
# 진행: $DART_ROOT/eval/status_a5.txt, 평가 실행 로그 $DART_ROOT/eval/run_a5.log
# 이미 끝난 단계(합성 END rc=0, 에피소드 수 다 차고 에러 0 인 평가 태그)는 건너뛰므로 중간에 죽으면 다시 띄우면 이어서 한다.
set -uo pipefail
source ~/hayeon/dart_xarm7/env.sh
source ~/hayeon/dart_xarm7/exp6_cond.sh
exp6_env v0 || exit 2   # XARM7_COLOR=white XARM7_DYNAMICS=panda export → 평가 클라이언트가 상속
H=~/hayeon/dart_xarm7; CK=$DART_ROOT/ckpts
T0=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
T0STAR=$CK/dart_a0.5                                           # θ*_t0 = v0 (restore_env.sh 가 HF dart_a0.5/ 에서 받은 것)
SRC=$CK/a5/oneshot_src/pi05_libero_oneshotft/original/999
TGT=$CK/a5/oneshot_tgt/pi05_libero_oneshotft/original/999
TS=$CK/a5/dart_a0.5                                            # θ*_t5
FTST=$DART_ROOT/logs/oneshot_ft_a5/status.txt
MLOG=$DART_ROOT/logs/dart_a5_a0.5.log
REST="0 1 2 3 4 6 7 8 9"
ST=$DART_ROOT/eval/status_a5.txt; RL=$DART_ROOT/eval/run_a5.log; mkdir -p "$(dirname "$ST")"
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
hf_env() { HF_TOKEN_PATH=$HOME/.cache/huggingface/token HF_HUB_DISABLE_PROGRESS_BARS=1 "$@"; }

log "=== [a5] 합성·평가 러너 시작 (job $PBS_JOBID, $(hostname), XARM7_COLOR=$XARM7_COLOR XARM7_DYNAMICS=$XARM7_DYNAMICS) — FT 끝 대기 ==="

# ---- 1. FT 끝 대기 ----
if [ "${FROM_HF:-0}" = 1 ]; then
  # 다른 잡(A6000 등)에서: FT 잡이 홈으로 복사한 status 에 src·tgt HF 백업 "크기 일치 True" 가 찍히면 HF 에서 받는다 (2026-11-08 사용자 결정: FT 만 A100).
  HST=$H/logs/oneshot_ft_a5/status.txt
  log "FROM_HF=1 — $HST 에 src·tgt HF 백업 완료가 찍힐 때까지 대기"
  until [ "$(grep -cE '(src|tgt) HF 백업 rc=0: a5/oneshot_(src|tgt): .*크기 일치 True' "$HST" 2>/dev/null)" -ge 2 ]; do
    grep -qE "중단|실패|단계를 모두|HF 백업 rc=[1-9]" "$HST" 2>/dev/null && { log "FT 잡이 실패/중단으로 끝남 — 중단: $(grep -E '중단|실패' "$HST" | tail -1)"; exit 1; }
    sleep 120
  done
  for r in src tgt; do
    d=$CK/a5/oneshot_$r/pi05_libero_oneshotft/original/999
    [ -d "$d/params" ] && { log "$r 체크포인트 이미 있음 — 받기 건너뜀"; continue; }
    ( cd "$DART_REPO" && hf_env .venv/bin/python $H/fetch_ckpts_hf.py "$CK/a5/_hf" "a5/oneshot_$r" ) >> $DART_ROOT/logs/a5_fetch_hf.log 2>&1 ||
      { log "HF a5/oneshot_$r 받기/대조 실패 — 중단 ($DART_ROOT/logs/a5_fetch_hf.log)"; exit 1; }
    mkdir -p "$(dirname "$d")" && mv "$CK/a5/_hf/a5/oneshot_$r" "$d"
    log "HF a5/oneshot_$r → $d ($(grep "^a5/oneshot_$r" $DART_ROOT/logs/a5_fetch_hf.log | tail -1 | sed 's/.*: 로컬/로컬/'))"
  done
  [ -d "$SRC/params" ] && [ -d "$TGT/params" ] || { log "체크포인트 없음 — 중단"; exit 1; }
else
  until grep -q "=== ALL_DONE" "$FTST" 2>/dev/null; do sleep 60; done
  if ! grep -q "=== ALL_DONE rc=0" "$FTST" || [ ! -d "$SRC/params" ] || [ ! -d "$TGT/params" ]; then
    log "FT 실패 또는 체크포인트 없음 — 중단 ($(grep 'ALL_DONE' "$FTST" | tail -1))"; exit 1
  fi
fi
log "FT 체크포인트 확인 — src $(du -sh "$SRC" | cut -f1), tgt $(du -sh "$TGT" | cut -f1)"

# ---- 2. 합성 (CPU) ----
if grep -q "END rc=0" "$MLOG" 2>/dev/null && [ -d "$TS/params" ]; then
  log "MERGE 이미 완료 — 건너뜀"
else
  log "MERGE start (α=0.5, 공식 CPU 경로 run_dart_merge_a5.sh)"
  bash $H/run_dart_merge_a5.sh 0.5
  log "MERGE $(tail -1 "$MLOG")"
  { grep -q "END rc=0" "$MLOG" && [ -d "$TS/params" ]; } || { log "합성 실패 — 중단 ($MLOG)"; exit 1; }
fi
mkdir -p $H/logs/oneshot_ft_a5 && cp "$MLOG" $DART_ROOT/logs/dart_a5_a0.5.rss $H/logs/oneshot_ft_a5/ 2>/dev/null
# HF 백업은 평가와 겹쳐도 됨 (네트워크·해시만, 합성과는 안 겹침)
( cd "$DART_REPO" && hf_env .venv/bin/python $H/upload_folder_hf.py "$TS" a5/dart_a0.5 "(실험 4-후속3 a5 θ*_t5, α=0.5)" > $DART_ROOT/logs/dart_a5_hf_upload.log 2>&1
  log "MERGE HF 백업 rc=$?: $(grep '^a5/dart_a0.5:' $DART_ROOT/logs/dart_a5_hf_upload.log | tail -1)" ) &
UP=$!

# ---- 3. 평가 6개 (순차) ----
for d in "$T0/params" "$T0STAR/params" "$TGT/params" "$TS/params"; do [ -d "$d" ] || { log "체크포인트 없음 $d — 평가 중단"; exit 1; }; done
if ps -eo args | grep -qE "[d]omain_arithmetic|[s]cripts/train.py"; then log "합성/FT 가 실행 중 — 평가 안 함"; exit 1; fi
[ -d "$DART_ROOT/osmesa/lib" ] || { log "osmesa 없음 ($DART_ROOT/osmesa) — 평가 중단 (restore 의 osmesa 단계 확인)"; exit 1; }
( source $H/client_env.sh; $LIBERO_PY -c "
import os, libero_xarm7 as X
assert (X.COLOR, X.DYNAMICS) == (os.environ['XARM7_COLOR'], os.environ['XARM7_DYNAMICS']), (X.COLOR, X.DYNAMICS)
print('client env OK', X.COLOR, X.DYNAMICS)" 2>&1 | tail -1 ) | tee -a "$ST" | grep -q "client env OK" || { log "클라이언트 env 확인 실패 — 중단"; exit 1; }

# 완료 = 기대 에피소드 수 + 에러 0
done_already() { [ "$(cat $H/results/$1/client*.jsonl 2>/dev/null | wc -l)" -eq "$2" ] && grep -q "/$2 = .*errors=0" $H/results/$1/summary.txt 2>/dev/null; }
ev() {  # ev <태그> <config> <dir> <tasks> <trials> — serve_ckpt.py 경로 (run_eval_model.sh)
  local tag=$1 cfg=$2 dir=$3 tasks=$4 trials=$5 R=$H/results/$1
  local n=$(( $(wc -w <<<"$tasks") * trials ))
  done_already "$tag" "$n" && { log "EVAL $tag 이미 완료 — 건너뜀"; return 0; }
  log "EVAL $tag start (config=$cfg tasks=[$tasks] trials=$trials → $n ep)"
  mkdir -p "$R"; local t0=$(date +%s)
  printf 'tag=%s\ncond=v0 XARM7_COLOR=%s XARM7_DYNAMICS=%s\nserver=serve_ckpt config=%s dir=%s\nsuite=libero_spatial tasks=%s trials=%s nclient=5 seed=7\njob=%s host=%s start=%s\n' \
    "$tag" "$XARM7_COLOR" "$XARM7_DYNAMICS" "$cfg" "$dir" "$tasks" "$trials" "$PBS_JOBID" "$(hostname)" "$(date '+%F %T')" > "$R/meta.txt"
  TAG=$tag CONFIG=$cfg DIR=$dir ROBOT=XArm7 TASKS="$tasks" TRIALS=$trials NCLIENT=5 bash $H/run_eval_model.sh >> "$RL" 2>&1
  local rc=$?
  echo "end=$(date '+%F %T') wall=$(( $(date +%s) - t0 ))s rc=$rc" >> "$R/meta.txt"
  log "EVAL $tag rc=$rc wall=$(( $(date +%s) - t0 ))s: $(cat $R/summary.txt 2>/dev/null)"
  done_already "$tag" "$n" || { log "EVAL $tag 에러/미완료 → 1회 재시도"; rm -f "$R"/client*.jsonl
    TAG=$tag CONFIG=$cfg DIR=$dir ROBOT=XArm7 TASKS="$tasks" TRIALS=$trials NCLIENT=5 bash $H/run_eval_model.sh >> "$RL" 2>&1
    log "EVAL $tag 재시도 rc=$?: $(cat $R/summary.txt 2>/dev/null)"; done_already "$tag" "$n" || log "EVAL $tag 재시도 후에도 에러/미완료 — 확인 필요"; }
}
ev a5_theta0_t5      pi05_libero          "$T0"     "5"     50
ev a5_dartT0_t5      pi05_libero          "$T0STAR" "5"     50
ev a5_theta_tgt_t5   pi05_libero_oneshotft "$TGT"   "5"     50
ev a5_dart_t5        pi05_libero          "$TS"     "5"     50
ev a5_theta_tgt_rest pi05_libero_oneshotft "$TGT"   "$REST" 10
ev a5_dart_rest      pi05_libero          "$TS"     "$REST" 10

# ---- 4. 기록 ----
wait $UP
cp "$ST" $H/logs/oneshot_ft_a5/status_a5_eval.txt
( cd ~/hayeon && git add dart_xarm7/results/a5_* dart_xarm7/logs/oneshot_ft_a5 &&
  git commit -q -m "실험 4-후속3 (a5): FT 2회·합성 α=0.5·평가 6개 결과 (잡 $PBS_JOBID, $(hostname))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "git commit $(git -C ~/hayeon log --oneline -1)" ) || log "git commit 실패/변경 없음"
log "=== [a5] ALL_DONE ==="
