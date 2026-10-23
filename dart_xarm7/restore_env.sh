#!/usr/bin/env bash
# 새 잡에서 DART xArm7 작업 환경을 복원한다 (ENV_STATUS.md "새 잡에서 복원" + 평가 클라이언트 + OSMesa + HF 체크포인트).
# GPU 컨테이너 안에서, 터미널과 분리해 실행:
#   DART_SHM=/dev/shm/introai34_dart_$(cut -d. -f1 <<<"$PBS_JOBID") setsid nohup bash ~/hayeon/dart_xarm7/restore_env.sh > /dev/null 2>&1 < /dev/null &
# DART_SHM 을 주면 $DART_ROOT(/var/tmp/pbs.<잡ID>/dart) 를 그 경로로 가는 링크로 만든다 (/var/tmp 여유가 ~37G 보다 적을 때).
#   /dev/shm 은 노드 RAM 이고 노드 공용 → 잡 끝에 rm -rf 로 비울 것.
# 진행: $DART_ROOT/logs/restore/status.txt
set -uo pipefail
: "${PBS_JOBID:?PBS 잡(GPU 컨테이너) 안에서만 실행}"
JOBDIR=/var/tmp/pbs.$PBS_JOBID
if [ -n "${DART_SHM:-}" ] && [ ! -e "$JOBDIR/dart" ]; then mkdir -p "$DART_SHM" && ln -s "$DART_SHM" "$JOBDIR/dart"; fi
source ~/hayeon/dart_xarm7/env.sh
LOGDIR=$DART_ROOT/logs/restore; mkdir -p "$LOGDIR"
ST=$LOGDIR/status.txt
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
step() {  # step <이름> <명령...> — 소요 시간·rc 기록
  local name=$1; shift; local t0=$(date +%s)
  log "$name START"
  "$@" > "$LOGDIR/$name.log" 2>&1; local rc=$?
  log "$name END rc=$rc $(( $(date +%s) - t0 ))s"; return $rc
}

repo() {
  cd "$DART_ROOT"
  [ -d dart ] || git clone --recurse-submodules https://github.com/snumprlab/dart.git
  cd dart && git checkout 1b23c4f && git submodule update --init --recursive && git submodule status
}
uv_sync_fn() {  # opencv-python(GUI)+headless 가 같이 깔려 cv2 가 libX11 을 찾다 실패 → FT DataLoader 무한 대기. headless 재설치로 해결 (2026-10-08, 10-22 재발)
  cd "$DART_REPO" && uv sync --python 3.11 && uv pip install --reinstall --no-deps opencv-python-headless==4.11.0.86 &&
  .venv/bin/python -c "import cv2, cv2.typing; print('cv2', cv2.__version__, 'OK')"
}
theta0() { cd "$DART_REPO" && uv run python -c "from openpi.shared import download as d; print(d.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))"; }
hf_ckpts() { cd "$DART_REPO" && HF_TOKEN_PATH=$HOME/.cache/huggingface/token HF_HUB_DISABLE_PROGRESS_BARS=1 .venv/bin/python ~/hayeon/dart_xarm7/fetch_ckpts_hf.py "$DART_ROOT/ckpts" oneshot_src oneshot_tgt dart_a0.5; }
libero_venv() {
  local V=$DART_ROOT/venvs/dart-libero
  [ -x "$V/bin/python" ] || /usr/bin/python3 -m venv --system-site-packages --without-pip "$V"
  "$V/bin/python" -m pip install --no-cache-dir -c ~/venvs/dart-libero-constraints.txt -r ~/venvs/dart-libero-requirements.txt &&
  "$V/bin/python" -m pip install --no-cache-dir --no-deps robosuite==1.4.1 gym==0.25.2 "$DART_REPO/packages/openpi-client" &&
  "$V/bin/python" -c "import torch; print(torch.__version__, torch.__file__)"
}
osmesa() {
  export CONDA_PKGS_DIRS=$DART_ROOT/conda-pkgs
  [ -d "$DART_ROOT/osmesa" ] || /tools/anaconda3/bin/conda create -y -p "$DART_ROOT/osmesa" -c conda-forge 'mesalib<25' &&
  /tools/anaconda3/bin/conda clean -y --index-cache --tarballs --packages && ls "$DART_ROOT/osmesa/lib" | grep -i osmesa
}

log "=== restore 시작 (job $PBS_JOBID, $(hostname), DART_ROOT=$DART_ROOT -> $(readlink -f "$DART_ROOT")) ==="
step repo repo || { log "저장소 실패 — 중단"; exit 1; }
step osmesa osmesa &  P1=$!
step uv_sync uv_sync_fn || { log "uv sync 실패 — 중단"; exit 1; }
step libero_venv libero_venv &  P2=$!
step theta0 theta0 &  P3=$!
step hf_ckpts hf_ckpts &  P4=$!
wait $P1 $P2 $P3 $P4
log "=== ALL_DONE (df: $(df -BG --output=avail "$(readlink -f "$DART_ROOT")" | tail -1 | tr -dc 0-9)G free, $(du -sh "$(readlink -f "$DART_ROOT")" | cut -f1) used) ==="
