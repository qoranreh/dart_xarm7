# DART (Panda→xArm7) 작업용 환경 변수. GPU 컨테이너 안에서:  source ~/hayeon/dart_xarm7/env.sh
# 큰 것은 전부 잡 전용 노드 로컬 디스크(/var/tmp/pbs.<잡ID>/dart, 휘발성)에 둔다. 홈 쿼터(≈28GB)가 꽉 차 있음.
: "${PBS_JOBID:?PBS 잡(GPU 컨테이너) 안에서만 source 할 것}"

export DART_ROOT=/var/tmp/pbs.${PBS_JOBID}/dart
export DART_REPO=$DART_ROOT/dart              # snumprlab/dart @1b23c4f, venv = $DART_REPO/.venv (uv run 기본 위치)

# uv: 캐시를 홈(~/.cache/uv)에 쓰지 않게. Python 3.11 은 ~/.local/share/uv/python 의 것을 씀.
export UV_CACHE_DIR=$DART_ROOT/uv-cache
export UV_PYTHON_DOWNLOADS=never
export GIT_LFS_SKIP_SMUDGE=1

# openpi 체크포인트·토크나이저 캐시 (기본값 ~/.cache/openpi 는 쿼터 초과)
export OPENPI_DATA_HOME=$DART_ROOT/openpi-cache
# LeRobot 데이터셋 / HF 캐시
export HF_HOME=$DART_ROOT/hf-home
export HF_LEROBOT_HOME=$DART_ROOT/lerobot

# wandb: 로그인 안 되어 있음(~/.netrc 없음). finetune.sh 기본값(online, entity=domain-arithmetic=저자 팀)은 실패하므로 offline.
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=$DART_ROOT/wandb

# JAX. CUDA_VISIBLE_DEVICES 는 런처가 GPU UUID 로 이미 설정 → finetune.sh 의 기본값 0,1 은 적용되지 않음.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}

mkdir -p "$UV_CACHE_DIR" "$OPENPI_DATA_HOME" "$HF_HOME" "$HF_LEROBOT_HOME" "$WANDB_DIR"
[ -d "$DART_REPO/.venv" ] || echo "[env.sh] 경고: $DART_REPO/.venv 없음 — 새 잡이면 ENV_STATUS.md 의 '새 잡에서 복원' 절차부터"
