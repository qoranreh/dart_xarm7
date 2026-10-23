#!/usr/bin/env bash
# 실험 6 — 조건별 DART 산술 θ* = θ0 + α·δ (run_dart_merge.sh 복제, 공식 domain_arithmetic.dart 명령 그대로). 사용: run_dart_merge_cond.sh <cond> <alpha>
# run_dart_merge.sh 와 다른 것: policy_src = v0 θ_src (모든 조건 공유), policy_tgt = ckpts/<cond>/oneshot_tgt/…/999,
#   출력 ckpts/<cond>/dart_a<α>, 로그 logs/dart_<cond>_a<α>.{log,rss}, 시작 전 평가 클라이언트가 돌고 있으면 거부.
# 공식 dart.py 는 jax 를 CPU 로 강제한다 → RAM 사용량(RSS)을 10초마다 기록 (잡 메모리 한도 192GB, 공식 경고 100GB+).
set -uo pipefail
source ~/hayeon/dart_xarm7/env.sh
cd "$DART_REPO"
C=${1:?cond}; A=${2:?alpha}
CK=$DART_ROOT/ckpts
SRC=$CK/oneshot_src/pi05_libero_oneshotft/original/999        # v0 θ_src
TGT=$CK/$C/oneshot_tgt/pi05_libero_oneshotft/original/999     # 조건 θ_tgt
OUT=$CK/$C/dart_a$A
LOG=$DART_ROOT/logs/dart_${C}_a$A.log
MEM=$DART_ROOT/logs/dart_${C}_a$A.rss
[ -d "$TGT/params" ] || { echo "θ_tgt 없음: $TGT"; exit 1; }
if pgrep -f eval_xarm7.py > /dev/null; then echo "평가 클라이언트가 실행 중 — 합성과 겹치면 안 됨, 중단"; exit 1; fi
free_gb() { df -BG --output=avail "$CK/" | tail -1 | tr -dc 0-9; }
echo "$(date '+%F %T') START cond=$C alpha=$A src=$SRC tgt=$TGT out=$OUT free=$(free_gb)G" > "$LOG"
t0=$(date +%s)
uv run -m domain_arithmetic.dart \
  --cfg.base_policy.dir "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero" \
  --cfg.base_policy.config pi05_libero \
  --cfg.policy_src.dir "$SRC" \
  --cfg.policy_src.config pi05_libero_oneshotft \
  --cfg.policy_tgt.dir "$TGT" \
  --cfg.policy_tgt.config pi05_libero_oneshotft \
  --cfg.scaling_coef "$A" \
  --cfg.output_dir "$OUT" \
  --cfg.overwrite >> "$LOG" 2>&1 &
PID=$!
: > "$MEM"
while kill -0 $PID 2>/dev/null; do
  # uv 아래 python 자식들까지 합친 RSS (GiB)
  rss=$(ps -o rss= --ppid $PID -p $PID 2>/dev/null | awk '{s+=$1} END{printf "%.1f", s/1048576}')
  echo "$(date +%T) $rss" >> "$MEM"; sleep 10
done
wait $PID; rc=$?
peak=$(awk '{if ($2>m) m=$2} END{print m}' "$MEM")
echo "$(date '+%F %T') END rc=$rc wall=$(( $(date +%s) - t0 ))s peak_rss=${peak}GiB size=$(du -sh "$OUT" 2>/dev/null | cut -f1) free=$(free_gb)G" >> "$LOG"
exit $rc
