#!/usr/bin/env bash
# 실험 6 단계 2 전체 (순차): 타깃 FT v1→v2→v3 (run_oneshot_ft_cond.sh, 각 HF 백업 포함)
#   → DART 합성 α=0.5 v1→v2→v3 (run_dart_merge_cond.sh, 공식 CPU 경로) + 각 HF 백업 <cond>/dart_a0.5/
#   → v0 θ* 대비 δ 모듈별 비교 (compare_merge.py)
# 합성은 FT·업로드가 모두 끝난 뒤에만 시작한다 (CPU 경쟁 방지). 평가 클라이언트는 이 스크립트가 돌 동안 띄우지 말 것.
#   setsid nohup bash ~/hayeon/dart_xarm7/run_exp6_stage2.sh [cond ...] > /dev/null 2>&1 < /dev/null &
# 진행: $DART_ROOT/logs/exp6/stage2_status.txt (FT 세부는 $DART_ROOT/logs/oneshot_ft/status.txt)
set -uo pipefail
source ~/hayeon/dart_xarm7/env.sh
H=~/hayeon/dart_xarm7; CK=$DART_ROOT/ckpts; A=0.5
L=$DART_ROOT/logs/exp6; mkdir -p "$L"; ST=$L/stage2_status.txt
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
CONDS=${*:-v1 v2 v3}
log "=== stage2 시작 (job $PBS_JOBID, $(hostname)) conds: $CONDS ==="

bash $H/run_oneshot_ft_cond.sh $CONDS
grep -q "ALL_DONE rc=0" "$DART_ROOT/logs/oneshot_ft/status.txt" || { log "FT 실패 — 합성 안 함 ($(grep ALL_DONE "$DART_ROOT/logs/oneshot_ft/status.txt" | tail -1))"; exit 1; }
log "FT 완료: $(grep -E 'DONE ckpt|HF 백업' "$DART_ROOT/logs/oneshot_ft/status.txt" | tail -6 | tr '\n' ' ')"

for c in $CONDS; do
  bash $H/run_dart_merge_cond.sh "$c" "$A"; rc=$?
  log "MERGE $c a=$A rc=$rc: $(grep 'END rc=' "$DART_ROOT/logs/dart_${c}_a$A.log")"
  [ $rc -eq 0 ] || { log "합성 $c 실패 — 중단"; exit 1; }
  ( cd "$DART_REPO" && HF_TOKEN_PATH=$HOME/.cache/huggingface/token HF_HUB_DISABLE_PROGRESS_BARS=1 \
      .venv/bin/python $H/upload_folder_hf.py "$CK/$c/dart_a$A" "$c/dart_a$A" "(실험 6 $c θ*, α=$A, 공식 CPU 합성)" ) >> "$L/hf_upload_merges.log" 2>&1
  log "HF 백업 $c/dart_a$A rc=$?: $(grep "^$c/dart_a$A:" "$L/hf_upload_merges.log" | tail -1)"
done

for c in $CONDS; do
  ( cd "$DART_REPO" && JAX_PLATFORMS=cpu .venv/bin/python $H/compare_merge.py "$CK/dart_a$A" "$CK/$c/dart_a$A" \
      "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero" ) > "$L/compare_v0_${c}_a$A.txt" 2>&1
  log "COMPARE v0 vs $c rc=$?: $(grep -A3 '^모듈별' "$L/compare_v0_${c}_a$A.txt" | tail -3 | tr -s ' ' | tr '\n' ';')"
done
mkdir -p $H/logs/exp6_stage2 && cp "$L"/stage2_status.txt "$L"/compare_v0_*.txt "$DART_ROOT"/logs/dart_v[0-9]_a*.log $H/logs/exp6_stage2/ 2>/dev/null
log "=== stage2 ALL_DONE ==="
