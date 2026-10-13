#!/usr/bin/env bash
# DART 산술 θ* = θ0 + α·δ (공식 domain_arithmetic.dart, 저장소 안내 명령 그대로). 사용: run_dart_merge.sh <alpha>
# 공식 dart.py 는 jax 를 CPU 로 강제한다 → RAM 사용량(RSS)을 10초마다 기록 (잡 메모리 한도 192GB, 공식 경고 100GB+).
set -uo pipefail
source ~/hayeon/dart_xarm7/env.sh
cd "$DART_REPO"
A=${1:?alpha}
OUT=$DART_ROOT/ckpts/dart_a$A
LOG=$DART_ROOT/logs/dart_a$A.log
MEM=$DART_ROOT/logs/dart_a$A.rss
CK=$DART_ROOT/ckpts
echo "$(date '+%F %T') START alpha=$A out=$OUT free=$(df -BG --output=avail /var/tmp | tail -1 | tr -dc 0-9)G" > "$LOG"
t0=$(date +%s)
uv run -m domain_arithmetic.dart \
  --cfg.base_policy.dir "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero" \
  --cfg.base_policy.config pi05_libero \
  --cfg.policy_src.dir "$CK/oneshot_src/pi05_libero_oneshotft/original/999" \
  --cfg.policy_src.config pi05_libero_oneshotft \
  --cfg.policy_tgt.dir "$CK/oneshot_tgt/pi05_libero_oneshotft/original/999" \
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
echo "$(date '+%F %T') END rc=$rc wall=$(( $(date +%s) - t0 ))s peak_rss=${peak}GiB size=$(du -sh "$OUT" 2>/dev/null | cut -f1) free=$(df -BG --output=avail /var/tmp | tail -1 | tr -dc 0-9)G" >> "$LOG"
