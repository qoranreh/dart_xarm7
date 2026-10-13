#!/usr/bin/env bash
# θ0(pi05_libero) one-shot full FT 2회를 순차 실행: source(Panda 데모) → target(xArm7 데모, MimicGen+회전 제한).
# 각 러닝이 끝나면 params 를 bf16 으로 변환 (DART·서빙은 bf16 으로 읽으므로 결과 동일, 디스크 11.6G → 4.9G).
#
# 실행 (GPU 컨테이너, 터미널과 분리):
#   setsid nohup bash ~/hayeon/dart_xarm7/run_oneshot_ft.sh > /dev/null 2>&1 < /dev/null &
# 진행 상황: $DART_ROOT/logs/oneshot_ft/status.txt, 러닝별 로그 같은 폴더 (끝나면 ~/hayeon/dart_xarm7/logs/oneshot_ft/ 로 복사)
#
# OOM 대응 순서 (사용자 지정): XLA mem fraction 0.9 → 0.95 → batch 축소(32). EMA 는 config 에서 이미 꺼짐(ema_decay=None),
# gradient checkpointing 은 이미 켜짐(gemma/siglip nn.remat nothing_saveable). 그래도 OOM 이면 중단.
set -uo pipefail
source ~/hayeon/dart_xarm7/env.sh
cd "$DART_REPO"

OUT=$DART_ROOT/ckpts
LOGDIR=$DART_ROOT/logs/oneshot_ft
mkdir -p "$OUT" "$LOGDIR"
STATUS=$LOGDIR/status.txt
DATA=$HOME/hayeon/dart_xarm7/data/lerobot
DATA_SRC=$DATA/libero_spatial_t0_demo_0_src_panda
DATA_TGT=$DATA/libero_spatial_t0_demo_0_tgt_xarm7_retarget_rotclip

export OPENPI_SAVE_TRAIN_STATE=0                       # params/assets 만 저장 (checkpoints.py 수정분)
export JAX_COMPILATION_CACHE_DIR=$DART_ROOT/jax-cache  # 기본값 ~/.cache/jax 는 홈 쿼터 위험 (train.py 수정분)
export WANDB_CACHE_DIR=$DART_ROOT/wandb/cache WANDB_DATA_DIR=$DART_ROOT/wandb/data
export EXTRA_ARGS="--log-interval=10 --overwrite"     # 로그 10스텝마다(학습 결과 무관), 재시도 시 같은 경로 덮어씀
NEED_GB=14                                             # fp32 params 11.6G + 컴파일 캐시·여유

log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
free_gb() { df -BG --output=avail "$OUT/" | tail -1 | tr -dc 0-9; }  # 체크포인트가 놓이는 파일시스템 (ckpts 가 /dev/shm 링크일 수 있음)

run_one() {
  local name=$1 data=$2 base=$OUT/oneshot_$1
  local step_dir=$base/pi05_libero_oneshotft/original/999
  for attempt in "0.9 64" "0.95 64" "0.95 32"; do
    set -- $attempt
    local mem=$1 bs=$2 lg=$LOGDIR/${name}_mem${1}_bs${2}.log
    if [ "$(free_gb)" -lt $NEED_GB ]; then log "$name 중단: /var/tmp 여유 $(free_gb)G < ${NEED_GB}G"; return 2; fi
    log "$name START mem_fraction=$mem batch=$bs data=$data free=$(free_gb)G"
    local t0=$(date +%s)
    XLA_PYTHON_CLIENT_MEM_FRACTION=$mem BATCH_SIZE=$bs DATA_REPO_ID=$data CHECKPOINT_BASE_DIR=$base CAMERA_VIEW=original \
      bash exec/finetune.sh > "$lg" 2>&1
    local rc=$? t1=$(date +%s)
    log "$name END rc=$rc train_wall=$((t1 - t0))s free=$(free_gb)G"
    if [ $rc -eq 0 ] && [ -d "$step_dir/params" ]; then
      .venv/bin/python ~/hayeon/dart_xarm7/convert_ckpt_bf16.py "$step_dir" >> "$lg" 2>&1 || { log "$name bf16 변환 실패 (로그 확인)"; return 3; }
      log "$name DONE ckpt=$step_dir ($(du -sh "$step_dir" | cut -f1)) free=$(free_gb)G | $(grep -E '^Step [0-9]+:' "$lg" | tail -1)"
      return 0
    fi
    if grep -q -E "RESOURCE_EXHAUSTED|[Oo]ut of memory" "$lg"; then log "$name OOM → 다음 설정으로 재시도"; continue; fi
    log "$name 실패 (OOM 아님) — 중단, 로그: $lg"; return 1
  done
  log "$name OOM 대응 단계를 모두 썼는데도 실패 — 중단"; return 1
}

log "=== one-shot FT 시작 (job $PBS_JOBID, $(hostname)) ==="
run_one src "$DATA_SRC" && run_one tgt "$DATA_TGT"
rc=$?
mkdir -p ~/hayeon/dart_xarm7/logs/oneshot_ft && cp "$LOGDIR"/*.log "$LOGDIR"/status.txt ~/hayeon/dart_xarm7/logs/oneshot_ft/ 2>/dev/null
log "=== ALL_DONE rc=$rc ==="
