# 노트북(PBS 잡 없음) 용 env.sh + client_env.sh 대체. 실험 4-후속1 (oracle t4·t5) 데모 재생·retarget 전용 (정책 서버·학습 없음).
#   source ~/hayeon/dart_xarm7/env_laptop.sh
# 서버와 다른 점: $DART_ROOT 가 영구 경로(~/dart_laptop), 렌더 EGL (서버는 OSMesa), 클라이언트 venv 를 uv 로 새로 구성 (ORACLE_T45_LOG.md 0절).
export DART_ROOT=${DART_ROOT:-$HOME/dart_laptop}
export DART_REPO=$DART_ROOT/dart              # snumprlab/dart @1b23c4f, 서브모듈은 libero-view 만
export UV_CACHE_DIR=$DART_ROOT/uv-cache
# make_retarget_demo.py 가 읽는 norm_stats: $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/... → ~/ckpts/pi05_libero 링크
export OPENPI_DATA_HOME=$DART_ROOT/openpi-cache
export LIBERO_VENV=$DART_ROOT/venvs/dart-libero
export LIBERO_PY=$LIBERO_VENV/bin/python
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export LIBERO_CONFIG_PATH=$DART_ROOT/libero_cfg
LV=$DART_REPO/third_party/libero-view
export PYTHONPATH=$HOME/hayeon/dart_xarm7/src:$DART_REPO:$DART_REPO/packages/openpi-client/src:$LV${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$LIBERO_CONFIG_PATH" "$OPENPI_DATA_HOME/openpi-assets/checkpoints"
[ -e "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero" ] || ln -s "$HOME/ckpts/pi05_libero" "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"
[ -f $LIBERO_CONFIG_PATH/config.yaml ] || cat > $LIBERO_CONFIG_PATH/config.yaml <<YAML
benchmark_root: $LV/libero/libero
bddl_files: $LV/libero/libero/bddl_files
init_states: $LV/libero/libero/init_files
datasets: $LV/libero/datasets
assets: $LV/libero/libero/assets
YAML
