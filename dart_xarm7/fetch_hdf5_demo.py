"""LIBERO 원본 hdf5 (기본 libero_spatial task 0) 에서 demo_0~4 의 액션·상태만 HTTP 범위 요청으로 읽어 npz 로 저장한다 (509MB 전체를 받지 않음).
2026-10-08 잡 129138 에서 실행한 코드를 그대로 스크립트로 옮긴 것 (DEMO_PAIR.md 1번). make_demo_pair.py / make_retarget_demo.py 의 --hdf5-npz 입력.
hdf5 파일명 = benchmark 의 suite.get_task(task).name + "_demo.hdf5". 기본 출력 $DART_ROOT/demos/<suite 에서 libero_ 뺀 것>_t<task>_hdf5_demo0-4.npz.

  source ~/hayeon/dart_xarm7/env.sh && cd $DART_REPO && uv run python ~/hayeon/dart_xarm7/fetch_hdf5_demo.py   (task 0, h5py·fsspec 은 dart .venv 에 있음)
  source ~/hayeon/dart_xarm7/env.sh && source ~/hayeon/dart_xarm7/client_env.sh && $LIBERO_PY ~/hayeon/dart_xarm7/fetch_hdf5_demo.py --task 4
    (task 이름을 benchmark 에서 읽으므로 task 0 이 아니면 libero 가 import 되는 $LIBERO_PY 로)
"""
import argparse
import os
import sys
import time

import fsspec
import h5py
import numpy as np

T0_NAME = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"  # libero_spatial task 0 (dart .venv 에는 libero 가 없음)
p = argparse.ArgumentParser()
p.add_argument("out", nargs="?", default=None, help="출력 npz (생략 시 $DART_ROOT/demos/<suite>_t<task>_hdf5_demo0-4.npz)")
p.add_argument("--suite", default="libero_spatial")
p.add_argument("--task", type=int, default=0)
args = p.parse_args()
try:
    from libero.libero import benchmark
    name = benchmark.get_benchmark_dict()[args.suite]().get_task(args.task).name
except ImportError:
    if (args.suite, args.task) != ("libero_spatial", 0):
        sys.exit("libero 를 import 할 수 없음 — task 이름을 benchmark 에서 읽어야 하므로 $LIBERO_PY 로 실행할 것")
    name = T0_NAME
URL = f"https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/{args.suite}/{name}_demo.hdf5"
short = args.suite.removeprefix("libero_")
out_path = args.out or os.path.expandvars(f"$DART_ROOT/demos/{short}_t{args.task}_hdf5_demo0-4.npz")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
print("url:", URL)
t = time.time()
try:
    fobj = fsspec.open(URL, block_size=2**20).open()
except FileNotFoundError:
    sys.exit(f"404: URL 없음 — 중단 ({URL})")
f = h5py.File(fobj, "r")
d = f["data"]
names = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
print("n demos:", len(names), "| demo_0 keys:", list(d["demo_0"].keys()))
out = {}
for n in names[:5]:
    g = d[n]
    out[n] = dict(actions=g["actions"][()], states=g["states"][()], rewards=g["rewards"][()], dones=g["dones"][()],
                  ee_states=g["obs/ee_states"][()], gripper_states=g["obs/gripper_states"][()], joint_states=g["obs/joint_states"][()])
np.savez(out_path, **{f"{n}/{k}": v for n, dd in out.items() for k, v in dd.items()},
         problem_info=d.attrs.get("problem_info", ""), env_args=d.attrs.get("env_args", ""))
print("saved", out_path, "| demo_0 states", out["demo_0"]["states"].shape, "actions", out["demo_0"]["actions"].shape, f"| {time.time() - t:.0f}s")
