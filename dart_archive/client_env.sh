# source 해서 쓰는 평가 클라이언트 환경 (이미지 Python venv + OSMesa)
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/var/tmp/pbs.128706.ECE-util1/dart/osmesa/lib
export LIBERO_CONFIG_PATH=/var/tmp/pbs.128706.ECE-util1/dart/libero_cfg
export PYTHONPATH=/var/tmp/pbs.128706.ECE-util1/dart/dart:/var/tmp/pbs.128706.ECE-util1/dart/dart/packages/openpi-client/src:/var/tmp/pbs.128706.ECE-util1/dart/dart/third_party/libero-view
export OMP_NUM_THREADS=4
