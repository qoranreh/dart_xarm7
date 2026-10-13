#!/usr/bin/env bash
# DART 산술을 GPU 에서 실행 (α 스윕용). 공식 domain_arithmetic.dart 의 main() 을 그대로 부르되,
# `python -m domain_arithmetic.dart` 로 실행할 때만 걸리는 CPU 강제(jax_platform_name=cpu)를 피한다. 저장소 코드 무수정.
# 사용: run_dart_merge_gpu.sh <alpha> [출력 접미사]   → $DART_ROOT/ckpts/dart_a<alpha><접미사>
set -uo pipefail
source ~/hayeon/dart_xarm7/env.sh
# 2026-10-09: GPU 합성이 공식 CPU 합성과 크게 달라서(TF32: δ 상대차 17%, highest: 71%) 플래그가 있으면 공식 CPU 스크립트로 넘긴다.
if [ -f "$DART_ROOT/eval/USE_CPU_MERGE" ]; then exec bash ~/hayeon/dart_xarm7/run_dart_merge.sh "$1"; fi
cd "$DART_REPO"
A=${1:?alpha}; SUF=${2:-}
OUT=$DART_ROOT/ckpts/dart_a$A$SUF
LOG=$DART_ROOT/logs/dart_a$A$SUF.log
CK=$DART_ROOT/ckpts
echo "$(date '+%F %T') START alpha=$A backend=GPU out=$OUT free=$(df -BG --output=avail /var/tmp | tail -1 | tr -dc 0-9)G" > "$LOG"
t0=$(date +%s)
# 평가 서버(XLA mem 0.5)와 GPU 를 같이 쓰므로 선할당 끄기.
# JAX_DEFAULT_MATMUL_PRECISION=highest: A100 기본(TF32) 행렬곱으로는 CPU 합성과 δ 가 17% 달랐음 (2026-10-09 비교).
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_DEFAULT_MATMUL_PRECISION=${JAX_DEFAULT_MATMUL_PRECISION:-highest} .venv/bin/python -c "
import logging, tyro
from domain_arithmetic import dart
logging.basicConfig(level=logging.INFO)
tyro.cli(dart.main)
" \
  --cfg.base_policy.dir "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero" --cfg.base_policy.config pi05_libero \
  --cfg.policy_src.dir "$CK/oneshot_src/pi05_libero_oneshotft/original/999" --cfg.policy_src.config pi05_libero_oneshotft \
  --cfg.policy_tgt.dir "$CK/oneshot_tgt/pi05_libero_oneshotft/original/999" --cfg.policy_tgt.config pi05_libero_oneshotft \
  --cfg.scaling_coef "$A" --cfg.output_dir "$OUT" --cfg.overwrite >> "$LOG" 2>&1
rc=$?
echo "$(date '+%F %T') END rc=$rc wall=$(( $(date +%s) - t0 ))s size=$(du -sh "$OUT" 2>/dev/null | cut -f1) free=$(df -BG --output=avail /var/tmp | tail -1 | tr -dc 0-9)G" >> "$LOG"
