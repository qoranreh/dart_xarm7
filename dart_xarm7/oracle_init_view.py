"""실험 4-후속1 3번: 한 태스크의 demo 시작 장면을 Panda (set_init_state) 와 xArm7 (transfer_init_state) 로 만들고 agentview 를 나란히 저장한다.

이미지 방향은 make_demo_pair.py 녹화와 같게 180° 회전 (모델 입력 방향). 대기 스텝 없이 set_init_state 직후 장면.
IK 오차·옮긴 물체 조인트 수는 같은 이름의 .json 에.

  $LIBERO_PY ~/hayeon/dart_xarm7/oracle_init_view.py --task 4 --hdf5-npz $DART_ROOT/demos/spatial_t4_hdf5_demo0-4.npz
"""
import argparse
import json
import os
import pathlib

import numpy as np
from PIL import Image, ImageDraw

import libero_xarm7 as X
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

p = argparse.ArgumentParser()
p.add_argument("--suite", default="libero_spatial")
p.add_argument("--task", type=int, required=True)
p.add_argument("--demo", default="demo_0")
p.add_argument("--hdf5-npz", required=True)
p.add_argument("--out", default=None, help="기본 ~/hayeon/dart_xarm7/results/oracle_t45/init_t<task>.png")
args = p.parse_args()
out = pathlib.Path(args.out or os.path.expanduser(f"~/hayeon/dart_xarm7/results/oracle_t45/init_t{args.task}.png"))
out.parent.mkdir(parents=True, exist_ok=True)

state0 = np.load(args.hdf5_npz)[f"{args.demo}/states"][0]
task = benchmark.get_benchmark_dict()[args.suite]().get_task(args.task)
bddl = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)


def make(robot):
    env = OffScreenRenderEnv(bddl_file_name=bddl, robots=[robot], camera_heights=256, camera_widths=256,
                             camera_names=["agentview", "robot0_eye_in_hand"])
    env.seed(0)
    env.reset()
    return env


panda = make("Panda")
img_p = panda.set_init_state(state0)["agentview_image"][::-1, ::-1]
xarm = make("XArm7")
flat, info = X.transfer_init_state(panda, state0, xarm)
img_x = xarm.set_init_state(flat)["agentview_image"][::-1, ::-1]

tiles = []
for img, label in ((img_p, "Panda"), (img_x, "xArm7 (transfer_init_state)")):
    im = Image.fromarray(np.ascontiguousarray(img))
    ImageDraw.Draw(im).text((4, 2), f"t{args.task} {args.demo} {label}", fill=(255, 255, 0))
    tiles.append(np.asarray(im))
Image.fromarray(np.concatenate(tiles, 1)).save(out)
meta = {"task": task.language, "demo": args.demo, "env_meta": f"XARM7_COLOR={X.COLOR}, XARM7_DYNAMICS={X.DYNAMICS}",
        "ik_pos_err": info["ik_pos_err"], "ik_rot_err": info["ik_rot_err"], "n_copied_joints": len(info["copied_joints"]),
        "missing_joints": info["missing_joints"], "pixel_diff_mean": float(np.abs(img_p.astype(float) - img_x.astype(float)).mean())}
out.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
print(out, json.dumps(meta, ensure_ascii=False))
panda.close()
xarm.close()
