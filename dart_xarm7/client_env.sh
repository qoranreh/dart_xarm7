# LIBERO-view 평가 클라이언트 환경 (이미지 Python venv + OSMesa). env.sh 다음에 source:
#   source ~/hayeon/dart_xarm7/env.sh && source ~/hayeon/dart_xarm7/client_env.sh
# EGL 은 이 컨테이너에서 불가 (NVIDIA_DRIVER_CAPABILITIES 에 graphics 없음, libEGL 없음) → OSMesa(CPU) 사용
: "${DART_ROOT:?먼저 env.sh 를 source 할 것}"
export LIBERO_VENV=$DART_ROOT/venvs/dart-libero
export LIBERO_PY=$LIBERO_VENV/bin/python          # 그냥 python 은 /tools/anaconda3 이므로 경로로 부를 것
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$DART_ROOT/osmesa/lib
export LP_NUM_THREADS=${LP_NUM_THREADS:-4}         # llvmpipe 가 cpu_count=128 을 보고 스레드를 과하게 띄움
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export LIBERO_CONFIG_PATH=$DART_ROOT/libero_cfg
LV=$DART_REPO/third_party/libero-view
export PYTHONPATH=$HOME/hayeon/dart_xarm7/src:$DART_REPO:$DART_REPO/packages/openpi-client/src:$LV${PYTHONPATH:+:$PYTHONPATH}
mkdir -p $LIBERO_CONFIG_PATH
[ -f $LIBERO_CONFIG_PATH/config.yaml ] || cat > $LIBERO_CONFIG_PATH/config.yaml <<YAML
benchmark_root: $LV/libero/libero
bddl_files: $LV/libero/libero/bddl_files
init_states: $LV/libero/libero/init_files
datasets: $LV/libero/datasets
assets: $LV/libero/libero/assets
YAML
