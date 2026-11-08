#!/usr/bin/env bash
# [실험 4-후속3 a5] run_oneshot_ft.sh 복제 — 데이터(t5 demo_0)·출력 경로(ckpts/a5)·로그 경로만 다르고 학습 설정은 한 글자도 안 바꿈.
# 추가: 시작 때 데이터 meta/info.json·method 기록, 러닝마다 bf16 변환 후 HF 백업 (a5/oneshot_{src,tgt}/).
# 원본 설명: θ0(pi05_libero) one-shot full FT 2회를 순차 실행: source(Panda 데모) → target(xArm7 데모, MimicGen+회전 제한).
# 각 러닝이 끝나면 params 를 bf16 으로 변환 (DART·서빙은 bf16 으로 읽으므로 결과 동일, 디스크 11.6G → 4.9G).
#
# 실행 (GPU 컨테이너, 터미널과 분리):
#   setsid nohup bash ~/hayeon/dart_xarm7/run_oneshot_ft_a5.sh > /dev/null 2>&1 < /dev/null &
# 진행 상황: $DART_ROOT/logs/oneshot_ft_a5/status.txt (끝나면 ~/hayeon/dart_xarm7/logs/oneshot_ft_a5/ 로 복사)
#
# OOM 대응 순서 (사용자 지정): XLA mem fraction 0.9 → 0.95 → batch 축소(32). EMA 는 config 에서 이미 꺼짐(ema_decay=None),
# gradient checkpointing 은 이미 켜짐(gemma/siglip nn.remat nothing_saveable). 그래도 OOM 이면 중단.
set -uo pipefail
source ~/hayeon/dart_xarm7/env.sh
cd "$DART_REPO"

OUT=$DART_ROOT/ckpts/a5
LOGDIR=$DART_ROOT/logs/oneshot_ft_a5
mkdir -p "$OUT" "$LOGDIR"
STATUS=$LOGDIR/status.txt
DATA=$HOME/hayeon/dart_xarm7/data/lerobot/a5
DATA_SRC=$DATA/libero_spatial_t5_demo_0_src_panda
DATA_TGT=$DATA/libero_spatial_t5_demo_0_tgt_xarm7_retarget_rotclip

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

# ---- 이하 a5 추가분 (학습 설정과 무관: 데이터 확인·loss 요약·HF 백업) ----
DEMOS=$DART_ROOT/hf-a5/a5/demos   # STOP1 원본 npz·summary (잡 133746 에서 만들어 HF a5/demos/ 에 올린 것)
hf_env() { HF_TOKEN_PATH=$HOME/.cache/huggingface/token HF_HUB_DISABLE_PROGRESS_BARS=1 "$@"; }

fetch_demos() {  # HF a5/demos → $DEMOS (데이터셋이 그 npz 로 만든 것인지 대조하려고)
  [ -f "$DEMOS/libero_spatial_t5_demo_0_XArm7_retarget_rotq.npz" ] && return 0
  hf_env .venv/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('qoranreh/dart-xarm7-oneshot', allow_patterns=['a5/demos/*'], local_dir='$DART_ROOT/hf-a5'))" \
    >> "$LOGDIR/data_check.log" 2>&1
}

check_data() {  # check_data <src|tgt> <데이터셋> <npz 경로(확장자 없이)> <summary json> — info.json·method 를 로그에 남기고 t5 데모와 같은지 확인
  .venv/bin/python - "$@" >> "$LOGDIR/data_check.log" 2>&1 <<'EOF'
import io, json, pathlib, sys
import numpy as np, pyarrow.parquet as pq
from PIL import Image
role, data, demo, summ_path = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
info = json.loads((data / "meta/info.json").read_text())
tasks = [json.loads(l)["task"] for l in (data / "meta/tasks.jsonl").read_text().splitlines() if l.strip()]
z, summ = np.load(demo + ".npz"), json.load(open(summ_path))
print(f"[info.json {role}] {json.dumps(info, ensure_ascii=False)}")
print(f"[data] {role} DATA={data}")
print(f"[data] {role} meta/info.json: robot_type={info['robot_type']} total_frames={info['total_frames']} episodes={info['total_episodes']} "
      f"fps={info['fps']} features={list(info['features'])} | tasks.jsonl={tasks}")
if role == "tgt":
    print(f"[data] {role} npz={pathlib.Path(demo).name}.npz env_meta={z['env_meta']} | method: {summ['method']}")
    assert str(z["env_meta"]) == "XARM7_COLOR=white, XARM7_DYNAMICS=panda", f"npz 조건 {z['env_meta']} != v0"
    assert summ["xarm_success_at_end"] and summ["frames"] == 137
else:
    print(f"[data] {role} npz={pathlib.Path(demo).name}.npz method: Panda 재생 (hdf5 {summ['demo']} 액션 그대로) | Panda {summ['Panda']}")
    assert summ["Panda"]["success_at_end"]
assert "t5_demo_0" in str(data) and info["total_frames"] == 137 and info["total_episodes"] == 1
assert tasks == [str(z["task"])] and "ramekin" in tasks[0], f"태스크 문장 {tasks} != npz {z['task']}"
t = pq.read_table(next((data / "data/chunk-000").glob("*.parquet"))).to_pydict()
img = np.stack([np.array(Image.open(io.BytesIO(x["bytes"]))) for x in t["image"]])
assert np.array_equal(img, z["image"]), "데이터셋 이미지 != npz 이미지"
assert np.array_equal(np.array(t["actions"], np.float32), z["actions"].astype(np.float32)), "데이터셋 액션 != npz 액션"
print(f"[data] {role} OK — 데이터셋 이미지·액션 {len(img)} 프레임 = {pathlib.Path(demo).name}.npz")
EOF
  local rc=$?
  grep "^\[data\] $1 " "$LOGDIR/data_check.log" | tail -4 | while read -r l; do log "$l"; done
  [ $rc -eq 0 ] || log "[data] $1 확인 실패 rc=$rc — $LOGDIR/data_check.log"
  return $rc
}

loss_summary() {  # loss_summary <src|tgt> — 성공한 시도의 로그에서 step 0 → 990, 로그된 900~990 스텝 10개 평균 (v0 ONESHOT_FT.md 와 같은 정의)
  local lg; lg=$(ls -t "$LOGDIR/$1"_mem*_bs*.log | head -1)
  log "$1 LOSS $(awk -F'loss=' '/^Step [0-9]+:/{split($1,a," "); s=a[2]+0; split($2,b,","); l=b[1]+0; if(s==0) f=l; if(s==990) e=l; if(s>=900){n++; t+=l}}
    END{printf "step0=%.4f step990=%.4f last100_mean(900~990, n=%d)=%.4f", f, e, n, (n ? t/n : 0)}' "$lg") ($(basename "$lg"))"
}

hf_backup() {  # hf_backup <src|tgt> <step_dir> — HF 비공개 저장소 a5/oneshot_<src|tgt>/ 로 올리고 파일 수·크기 대조 (백그라운드로 부름)
  local hl=$LOGDIR/${1}_hf_upload.log
  hf_env .venv/bin/python ~/hayeon/dart_xarm7/upload_folder_hf.py "$2" "a5/oneshot_$1" "(실험 4-후속3 a5 θ_$1, t5 demo_0, bf16)" >> "$hl" 2>&1
  log "$1 HF 백업 rc=$?: $(grep "^a5/oneshot_$1:" "$hl" | tail -1)"
}

log "=== [a5] one-shot FT 시작 (job $PBS_JOBID, $(hostname)) | ckpt 위치 $(readlink -f "$OUT") 여유 $(free_gb)G, /var/tmp 여유 $(df -BG --output=avail /var/tmp | tail -1 | tr -dc 0-9)G ==="
# dart_oneshot_ft.patch (저장·로그 3곳) 가 적용돼 있어야 v0 와 같은 설정. restore_env.sh 는 적용하지 않으므로 확인 (없으면 train_state ~23GiB 저장, JAX 캐시가 홈으로).
git -C "$DART_REPO" apply --reverse --check ~/hayeon/dart_xarm7/patches/dart_oneshot_ft.patch 2>/dev/null ||
  { log "dart_oneshot_ft.patch 미적용 — 학습 안 함 (cd \$DART_REPO && git apply ~/hayeon/dart_xarm7/patches/dart_oneshot_ft.patch)"; exit 2; }
log "dart_oneshot_ft.patch 적용 확인 | 저장소 $(git -C "$DART_REPO" rev-parse --short HEAD), 수정 파일: $(git -C "$DART_REPO" diff --name-only | tr '\n' ' ')"
fetch_demos || { log "HF a5/demos 받기 실패 — 중단 ($LOGDIR/data_check.log)"; exit 2; }
check_data src "$DATA_SRC" "$DEMOS/libero_spatial_t5_demo_0_Panda" "$DEMOS/libero_spatial_t5_demo_0_summary.json" &&
  check_data tgt "$DATA_TGT" "$DEMOS/libero_spatial_t5_demo_0_XArm7_retarget_rotq" "$DEMOS/libero_spatial_t5_demo_0_XArm7_retarget_rotq_summary.json" ||
  { log "데이터 확인 실패 — 학습 안 함"; exit 2; }
[ "${DRY_RUN:-0}" = 1 ] && { log "DRY_RUN — 데이터 확인만, 학습 안 함"; exit 0; }
UPLOADS=()
rc=0
for r in src tgt; do
  data=$DATA_SRC; [ $r = tgt ] && data=$DATA_TGT
  run_one $r "$data" || { rc=$?; break; }
  loss_summary $r
  hf_backup $r "$OUT/oneshot_$r/pi05_libero_oneshotft/original/999" &
  UPLOADS+=($!)
done
[ ${#UPLOADS[@]} -gt 0 ] && wait "${UPLOADS[@]}"
mkdir -p ~/hayeon/dart_xarm7/logs/oneshot_ft_a5 && cp "$LOGDIR"/*.log "$LOGDIR"/status.txt ~/hayeon/dart_xarm7/logs/oneshot_ft_a5/ 2>/dev/null
log "=== ALL_DONE rc=$rc ==="
