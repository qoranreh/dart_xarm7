"""LIBERO-view 환경을 지정한 로봇으로 만들고 agentview / wrist 이미지 1장씩 저장 + 액션·state 공간 출력.

실행 (GPU 컨테이너, headless OSMesa):
  source ~/hayeon/dart_xarm7/env.sh && source ~/hayeon/dart_xarm7/client_env.sh
  $LIBERO_PY ~/hayeon/dart_xarm7/render_test.py --robot Panda
  $LIBERO_PY ~/hayeon/dart_xarm7/render_test.py --robot XArm7
"""
import argparse
import json
import os
import pathlib

import imageio
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--robot", default="Panda")
p.add_argument("--suite", default="libero_spatial")
p.add_argument("--task", type=int, default=0)
p.add_argument("--res", type=int, default=256)
p.add_argument("--out", default=os.path.expanduser("~/hayeon/dart_xarm7/renders"))
p.add_argument("--init-state", action="store_true", help="공식 평가처럼 set_init_state(init_states[0]) 적용 (Panda 전용)")
args = p.parse_args()

if args.robot != "Panda":
    import libero_xarm7  # noqa: F401  (MountedXArm7 / XArm7Gripper 등록)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()[args.suite]()
task = suite.get_task(args.task)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(
    bddl_file_name=str(bddl),
    robots=[args.robot],
    camera_heights=args.res,
    camera_widths=args.res,
    camera_names=["agentview", "robot0_eye_in_hand"],
)
env.seed(7)
obs = env.reset()
if args.init_state:
    obs = env.set_init_state(suite.get_task_init_states(args.task)[0])
# 공식 평가처럼 물체가 떨어져 자리 잡도록 10 스텝 대기 (LIBERO_DUMMY_ACTION = [0]*6 + [-1])
for _ in range(10):
    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

inner = env.env
robot = inner.robots[0]
out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
tag = f"{args.robot}_{args.suite}_t{args.task}" + ("_init" if args.init_state else "")
for cam in ["agentview", "robot0_eye_in_hand"]:
    img = obs[f"{cam}_image"][::-1, ::-1]  # 공식 평가와 같은 180도 회전 (모델 입력과 같은 방향)
    imageio.imwrite(out / f"{tag}_{cam}.png", img)

low, high = env.env.action_spec
info = {
    "task": task.language,
    "robot_model": type(robot.robot_model).__name__,
    "gripper_model": type(robot.gripper).__name__,
    "controller": type(robot.controller).__name__,
    "controller_cfg": {k: robot.controller_config.get(k) for k in ("type", "control_delta", "output_max", "output_min", "kp")},
    "action_dim": int(inner.action_dim),
    "action_low": low.tolist(),
    "action_high": high.tolist(),
    "arm_joints": list(robot.robot_model.joints),
    "gripper_joints": list(robot.gripper.joints),
    "init_qpos": np.round(robot.init_qpos, 4).tolist(),
    "base_pos": np.round(inner.sim.data.get_body_xpos(robot.robot_model.root_body), 4).tolist(),
    "eef_pos": np.round(obs["robot0_eef_pos"], 4).tolist(),
    "eef_quat": np.round(obs["robot0_eef_quat"], 4).tolist(),
    "gripper_qpos": np.round(obs["robot0_gripper_qpos"], 4).tolist(),
    "state_dim(eef_pos+axisangle+gripper_qpos)": 3 + 3 + len(obs["robot0_gripper_qpos"]),
    "images": [str(out / f"{tag}_{c}.png") for c in ["agentview", "robot0_eye_in_hand"]],
}
print(json.dumps(info, indent=1, ensure_ascii=False))
(out / f"{tag}_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
env.close()
