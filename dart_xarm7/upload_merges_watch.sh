#!/usr/bin/env bash
# DART 합성(α=0.8, 0.5, 1.0)이 끝날 때마다 θ* 를 HF 비공개 저장소 dart_a<α>/ 로 올린다.
source ~/hayeon/dart_xarm7/env.sh
ST=$DART_ROOT/eval/status.txt; LOG=$DART_ROOT/logs/hf_upload_merges.log
for A in 0.8 0.5 1.0; do
  until grep -q "END rc=" "$DART_ROOT/logs/dart_a$A.log" 2>/dev/null; do sleep 30; done
  if grep -q "END rc=0" "$DART_ROOT/logs/dart_a$A.log"; then
    ( cd "$DART_REPO" && HF_TOKEN_PATH=$HOME/.cache/huggingface/token HF_HUB_DISABLE_PROGRESS_BARS=1 \
      .venv/bin/python ~/hayeon/dart_xarm7/upload_folder_hf.py "$DART_ROOT/ckpts/dart_a$A" "dart_a$A" "(DART θ*, α=$A, 공식 CPU 합성)" ) >> "$LOG" 2>&1
    echo "$(date '+%F %T') HF upload dart_a$A rc=$?: $(grep "^dart_a$A:" "$LOG" | tail -1)" >> "$ST"
  else
    echo "$(date '+%F %T') HF upload dart_a$A 건너뜀 (합성 실패)" >> "$ST"
  fi
done
