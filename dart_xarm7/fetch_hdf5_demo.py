"""LIBERO 원본 hdf5 (libero_spatial task 0) 에서 demo_0~4 의 액션·상태만 HTTP 범위 요청으로 읽어 npz 로 저장한다 (509MB 전체를 받지 않음).
2026-10-08 잡 129138 에서 실행한 코드를 그대로 스크립트로 옮긴 것 (DEMO_PAIR.md 1번). make_demo_pair.py / make_retarget_demo.py 의 --hdf5-npz 입력.

  source ~/hayeon/dart_xarm7/env.sh && cd $DART_REPO && uv run python ~/hayeon/dart_xarm7/fetch_hdf5_demo.py   (h5py·fsspec 은 dart .venv 에 있음)
"""
import os
import sys
import time

import fsspec
import h5py
import numpy as np

URL = ("https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_spatial/"
       "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5")
out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars("$DART_ROOT/demos/spatial_t0_hdf5_demo0-4.npz")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
t = time.time()
f = h5py.File(fsspec.open(URL, block_size=2**20).open(), "r")
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
