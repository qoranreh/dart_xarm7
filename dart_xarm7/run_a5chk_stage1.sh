#!/usr/bin/env bash
# [실험 4-후속4 a5chk] STOP1 카드 B — t5 × 10회, 모델 3개 순차, 모든 에피소드 영상 저장 (--save-all-videos, 기본 off 플래그).
# exp6_env v0, seed 7 (eval_xarm7.py 기본), 클라이언트 5개 — 4-후속3 평가와 같은 설정. 태그 a5chk_*.
#   setsid nohup bash ~/hayeon/dart_xarm7/run_a5chk_stage1.sh > /dev/null 2>&1 < /dev/null &
# 진행: $DART_ROOT/eval/status_a5chk.txt
set -uo pipefail
source ~/hayeon/dart_xarm7/env.sh
source ~/hayeon/dart_xarm7/exp6_cond.sh
exp6_env v0 || exit 2
# 2026-11-09 15:15 첫 시도에서 클라이언트 5개가 홈(NFS) ~/.cache/mesa_shader_cache_db 파일 잠금에 막혀 멈춤 → 렌더러 캐시를 잡 로컬 디스크로 (결과 무관)
export MESA_SHADER_CACHE_DIR=$DART_ROOT/mesa-cache; mkdir -p "$MESA_SHADER_CACHE_DIR"
H=~/hayeon/dart_xarm7; CK=$DART_ROOT/ckpts
T0=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
TGT=$CK/a5/oneshot_tgt/pi05_libero_oneshotft/original/999
TS=$CK/a5/dart_a0.5
ST=$DART_ROOT/eval/status_a5chk.txt; RL=$DART_ROOT/eval/run_a5chk.log; mkdir -p "$(dirname "$ST")"
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }

log "=== [a5chk] STOP1 카드 B 시작 (job $PBS_JOBID, $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader), XARM7_COLOR=$XARM7_COLOR XARM7_DYNAMICS=$XARM7_DYNAMICS) ==="
for d in "$T0/params" "$TGT/params" "$TS/params"; do [ -d "$d" ] || { log "체크포인트 없음 $d — 중단"; exit 1; }; done
if ps -eo args | grep -qE "[d]omain_arithmetic|[s]cripts/train.py|[e]val_xarm7.py"; then log "합성/FT/다른 평가 실행 중 — 중단"; exit 1; fi

ev() {  # ev <태그> <config> <dir> [tasks] [trials]
  local tag=$1 cfg=$2 dir=$3 tasks=${4:-5} trials=${5:-10} R=$H/results/$1
  [ -e "$R/summary.txt" ] && { log "EVAL $tag 이미 있음 — 덮어쓰지 않고 건너뜀"; return 0; }
  log "EVAL $tag start (config=$cfg, tasks=[$tasks] × $trials, --save-all-videos)"
  mkdir -p "$R"; local t0=$(date +%s)
  printf 'tag=%s\ncond=v0 XARM7_COLOR=%s XARM7_DYNAMICS=%s\nserver=serve_ckpt config=%s dir=%s\nsuite=libero_spatial tasks=%s trials=%s nclient=5 seed=7 --save-all-videos MESA_SHADER_CACHE_DIR=local\njob=%s host=%s gpu=%s start=%s\n' \
    "$tag" "$XARM7_COLOR" "$XARM7_DYNAMICS" "$cfg" "$dir" "$tasks" "$trials" "$PBS_JOBID" "$(hostname)" "$(nvidia-smi --query-gpu=name --format=csv,noheader)" "$(date '+%F %T')" > "$R/meta.txt"
  EVAL_EXTRA_ARGS=--save-all-videos TAG=$tag CONFIG=$cfg DIR=$dir ROBOT=XArm7 TASKS="$tasks" TRIALS=$trials NCLIENT=5 bash $H/run_eval_model.sh >> "$RL" 2>&1
  local rc=$?
  echo "end=$(date '+%F %T') wall=$(( $(date +%s) - t0 ))s rc=$rc" >> "$R/meta.txt"
  log "EVAL $tag rc=$rc wall=$(( $(date +%s) - t0 ))s: $(cat $R/summary.txt 2>/dev/null) | 영상 $(ls $DART_ROOT/eval/$tag/videos/*.mp4 2>/dev/null | wc -l)개"
}
ev a5chk_theta0_t5v    pi05_libero           "$T0"
ev a5chk_theta_tgt_t5v pi05_libero_oneshotft "$TGT"
ev a5chk_dart_t5v      pi05_libero           "$TS"
log "=== [a5chk] STOP1 카드 B 평가 끝 ==="

# ---- STOP1 자동 통과 기준 (b) errors=0·영상 30개, (c) 4-후속3 같은 에피소드 0~9 성공 수 ±3 → 카드 A ----
chk=$(/usr/bin/python3 - "$H" "$DART_ROOT" <<'PY'
import json, glob, os, sys
H, D = sys.argv[1], sys.argv[2]
ok, out = True, []
for new, old in [("a5chk_theta0_t5v", "a5_theta0_t5"), ("a5chk_theta_tgt_t5v", "a5_theta_tgt_t5"), ("a5chk_dart_t5v", "a5_dart_t5")]:
    r = [json.loads(l) for f in glob.glob(f"{H}/results/{new}/client*.jsonl") for l in open(f)]
    o = [json.loads(l) for f in glob.glob(f"{H}/results/{old}/client*.jsonl") for l in open(f)]
    n_err = sum(1 for x in r if x.get("error")); n_vid = len(glob.glob(f"{D}/eval/{new}/videos/*.mp4"))
    s_new = sum(x["success"] for x in r); s_old = sum(x["success"] for x in o if x["episode"] < 10)
    good = len(r) == 10 and n_err == 0 and n_vid == 10 and abs(s_new - s_old) <= 3
    ok &= good
    out.append(f"{new}: {s_new}/{len(r)} (기존 ep0~9 {s_old}/10) errors={n_err} 영상={n_vid} {'OK' if good else 'FAIL'}")
print(" | ".join(out)); print("PASS" if ok else "STOP")
PY
)
log "STOP1 기준: $(echo "$chk" | head -1)"
echo "$chk" | tail -1 | grep -q PASS || { log "STOP1 기준 미달 — 카드 A 안 함, 멈춤"; exit 0; }
log "STOP1 자동 통과 → 카드 A (t4·t7 × 50, 모델 2개)"
ev a5chk_dartT0_t47 pi05_libero "$CK/dart_a0.5" "4 7" 50
ev a5chk_dart_t47   pi05_libero "$TS"           "4 7" 50
log "=== [a5chk] ALL_DONE ==="
