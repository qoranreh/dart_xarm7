#!/usr/bin/env bash
# 실험 6 단계 2 — 조건별 타깃 one-shot full FT (run_oneshot_ft.sh 복제). 소스 FT 는 건너뛰고 v0 θ_src 를 모든 조건이 공유한다.
# run_oneshot_ft.sh 와 다른 것 (그 외 학습 설정·OOM 대응 순서·bf16 변환/검증·환경 변수는 한 글자도 바꾸지 않음):
#   - 타깃만 학습, DATA_TGT = data/lerobot/<cond>/libero_spatial_t0_demo_0_tgt_xarm7_retarget_rotclip
#   - 출력 $DART_ROOT/ckpts/<cond>/oneshot_tgt/pi05_libero_oneshotft/original/999, 로그 $DART_ROOT/logs/oneshot_ft/<cond>_*
#   - 시작 시 데이터셋이 그 조건의 데모가 맞는지 확인 (info.json, npz 조건 기록, 이미지·액션 대조) — 틀리면 학습 안 함
#   - 끝나면 HF 백업 qoranreh/dart-xarm7-oneshot 의 <cond>/oneshot_tgt/ (백그라운드, 파일 수·크기 대조)
#
# 실행 (GPU 컨테이너, 터미널과 분리, 조건 여러 개면 순차):
#   setsid nohup bash ~/hayeon/dart_xarm7/run_oneshot_ft_cond.sh v1 v2 v3 > /dev/null 2>&1 < /dev/null &
# 데이터 확인만: DRY_RUN=1 bash run_oneshot_ft_cond.sh v1 v2 v3
# 진행 상황: $DART_ROOT/logs/oneshot_ft/status.txt (끝나면 ~/hayeon/dart_xarm7/logs/oneshot_ft_exp6/ 로 복사)
set -uo pipefail
[ $# -ge 1 ] || { echo "사용: $0 <cond> [cond ...]  (v1 | v2 | v3)"; exit 2; }
source ~/hayeon/dart_xarm7/env.sh
source ~/hayeon/dart_xarm7/exp6_cond.sh
cd "$DART_REPO"

OUT=$DART_ROOT/ckpts
LOGDIR=$DART_ROOT/logs/oneshot_ft
mkdir -p "$OUT" "$LOGDIR"
STATUS=$LOGDIR/status.txt
DATA=$HOME/hayeon/dart_xarm7/data/lerobot

export OPENPI_SAVE_TRAIN_STATE=0                       # params/assets 만 저장 (checkpoints.py 수정분)
export JAX_COMPILATION_CACHE_DIR=$DART_ROOT/jax-cache  # 기본값 ~/.cache/jax 는 홈 쿼터 위험 (train.py 수정분)
export WANDB_CACHE_DIR=$DART_ROOT/wandb/cache WANDB_DATA_DIR=$DART_ROOT/wandb/data
export EXTRA_ARGS="--log-interval=10 --overwrite"     # 로그 10스텝마다(학습 결과 무관), 재시도 시 같은 경로 덮어씀
NEED_GB=14                                             # fp32 params 11.6G + 컴파일 캐시·여유

log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
free_gb() { df -BG --output=avail "$OUT/" | tail -1 | tr -dc 0-9; }  # 체크포인트가 놓이는 파일시스템 (ckpts 가 /dev/shm 링크일 수 있음)

run_one() {  # run_one <이름> <데이터> <출력 base>
  local name=$1 data=$2 base=$3
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

check_data() {  # check_data <cond> <데이터셋> — 데이터셋이 그 조건의 데모(npz)에서 만든 것인지 확인하고 로그에 남김
  ( exp6_env "$1" && .venv/bin/python - "$1" "$2" "$DART_ROOT/demos/$1/libero_spatial_t0_demo_0_XArm7_retarget_rotq" <<'EOF'
import io, json, os, pathlib, sys
import numpy as np, pyarrow.parquet as pq
from PIL import Image
c, data, demo = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
info = json.loads((data / "meta/info.json").read_text())
z, summ = np.load(demo + ".npz"), json.load(open(demo + "_summary.json"))
want = f"XARM7_COLOR={os.environ['XARM7_COLOR']}, XARM7_DYNAMICS={os.environ['XARM7_DYNAMICS']}"
print(f"[data] cond={c} DATA_TGT={data}")
print(f"[data] meta/info.json: robot_type={info['robot_type']} total_frames={info['total_frames']} episodes={info['total_episodes']} "
      f"fps={info['fps']} features={list(info['features'])}")
print(f"[data] npz env_meta: {z['env_meta']} | summary method: {summ['method']}")
assert f"/{c}/" in str(data), f"데이터 경로에 /{c}/ 없음"
assert str(z["env_meta"]) == want, f"npz 조건 {z['env_meta']} != {c} 의 {want}"
t = pq.read_table(next((data / "data/chunk-000").glob("*.parquet"))).to_pydict()
img = np.stack([np.array(Image.open(io.BytesIO(x["bytes"]))) for x in t["image"]])
assert np.array_equal(img, z["image"]), "데이터셋 이미지 != 조건 npz 이미지"
assert np.array_equal(np.array(t["actions"], np.float32), z["actions"].astype(np.float32)), "데이터셋 액션 != 조건 npz 액션"
print(f"[data] OK — 데이터셋 이미지·액션 {len(img)} 프레임 = {c} 데모 ({want})")
EOF
  ) 2>&1 | grep -v -i warn | tee -a "$STATUS"
  return ${PIPESTATUS[0]}
}

hf_backup() {  # hf_backup <cond> <step_dir> — HF 비공개 저장소 <cond>/oneshot_tgt/ 로 올리고 대조 (백그라운드로 부름)
  local hl=$LOGDIR/${1}_hf_upload.log
  HF_TOKEN_PATH=$HOME/.cache/huggingface/token HF_HUB_DISABLE_PROGRESS_BARS=1 \
    .venv/bin/python ~/hayeon/dart_xarm7/upload_folder_hf.py "$2" "$1/oneshot_tgt" "(실험 6 $1 θ_tgt, bf16)" >> "$hl" 2>&1
  log "$1 HF 백업 rc=$?: $(grep "^$1/oneshot_tgt:" "$hl" | tail -1)"
}

log "=== 조건별 타깃 FT 시작 (job $PBS_JOBID, $(hostname)) conds: $* | ckpt 파일시스템 여유 $(free_gb)G, /var/tmp 여유 $(df -BG --output=avail /var/tmp | tail -1 | tr -dc 0-9)G ==="
UPLOADS=()
rc=0
for c in "$@"; do
  [ "$c" = v0 ] && { log "v0 는 이미 있음 — 건너뜀"; continue; }
  exp6_env "$c" || { rc=2; break; }
  DATA_TGT=$DATA/$c/libero_spatial_t0_demo_0_tgt_xarm7_retarget_rotclip
  check_data "$c" "$DATA_TGT" || { log "$c 데이터 확인 실패 — 중단"; rc=2; break; }
  [ "${DRY_RUN:-0}" = 1 ] && { log "$c DRY_RUN — 데이터 확인만, 학습 안 함"; continue; }
  run_one "${c}_tgt" "$DATA_TGT" "$OUT/$c/oneshot_tgt" || { rc=$?; break; }
  hf_backup "$c" "$OUT/$c/oneshot_tgt/pi05_libero_oneshotft/original/999" &
  UPLOADS+=($!)
done
[ ${#UPLOADS[@]} -gt 0 ] && wait "${UPLOADS[@]}"
mkdir -p ~/hayeon/dart_xarm7/logs/oneshot_ft_exp6 && cp "$LOGDIR"/v[0-9]_*.log "$LOGDIR"/status.txt ~/hayeon/dart_xarm7/logs/oneshot_ft_exp6/ 2>/dev/null
log "=== ALL_DONE rc=$rc ==="
