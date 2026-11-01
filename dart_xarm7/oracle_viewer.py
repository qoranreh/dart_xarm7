"""MuJoCo 대화형 뷰어(GLFW 창)로 데모 재생을 실시간으로 본다 — 실험 4-후속1 확인용 (노트북 화면).

같은 시작 장면 (Panda: hdf5 states[0], xArm7: transfer_init_state) → 대기 10스텝 → 저장된 액션을 20Hz(제어 주기)로 재생.
카메라 관측을 끈 env 라 물리는 녹화(make_demo_pair / make_retarget_demo)와 같다. 창은 마우스로 회전·확대 가능, 닫으면 끝.

  source ~/hayeon/dart_xarm7/env_laptop.sh && source ~/hayeon/dart_xarm7/exp6_cond.sh && exp6_env v0
  MUJOCO_GL=glfw $LIBERO_PY ~/hayeon/dart_xarm7/oracle_viewer.py --task 4 --demo 0 --robot XArm7 --variant quantile --loop
  --robot Panda 면 원본 데모 액션 (같은 폴더의 _Panda.npz), XArm7 이면 retarget npz 액션. --speed 0.5 = 절반 속도.
"""
import argparse
import os
import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np

import libero_xarm7 as X
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

p = argparse.ArgumentParser()
p.add_argument("--suite", default="libero_spatial")
p.add_argument("--task", type=int, required=True)
p.add_argument("--demo", type=int, required=True)
p.add_argument("--robot", choices=["Panda", "XArm7"], required=True)
p.add_argument("--variant", choices=["quantile", "none"], default="quantile")
p.add_argument("--speed", type=float, default=1.0)
p.add_argument("--wait", type=int, default=10)
p.add_argument("--loop", action="store_true", help="끝나면 2초 멈춘 뒤 처음부터 다시 (창을 닫을 때까지)")
args = p.parse_args()

DUMMY = [0.0] * 6 + [-1.0]
tag = f"{args.suite}_t{args.task}_demo_{args.demo}"
folder = pathlib.Path(os.path.expandvars(f"$DART_ROOT/demos/oracle_t{args.task}/demo{args.demo}"))
suffix = "Panda" if args.robot == "Panda" else ("XArm7_retarget_rotq" if args.variant == "quantile" else "XArm7_retarget")
actions = np.load(folder / f"{tag}_{suffix}.npz")["actions"].astype(np.float64)
state0 = np.load(os.path.expandvars(f"$DART_ROOT/demos/{args.suite.removeprefix('libero_')}_t{args.task}_hdf5_demo0-4.npz"))[f"demo_{args.demo}/states"][0]
task = benchmark.get_benchmark_dict()[args.suite]().get_task(args.task)
bddl = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)


def make(robot):
    env = ControlEnv(bddl_file_name=bddl, robots=[robot], use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
    env.seed(0)
    env.reset()
    return env


env = make(args.robot)
if args.robot == "Panda":
    init = state0
else:
    pa = make("Panda")
    init, _ = X.transfer_init_state(pa, state0, env)
    pa.close()
dt = env.env.control_timestep / args.speed
print(f"{task.language} | {suffix} | 액션 {len(actions)}개 | env {X.COLOR}/{X.DYNAMICS}")

# LIBERO 의 env.reset() 은 MjSim 을 새로 만들므로 (hard_reset) 뷰어를 띄운 뒤에는 set_init_state 로만 처음 장면으로 돌린다
env.set_init_state(init)
m, d = env.env.sim.model._model, env.env.sim.data._data
with mujoco.viewer.launch_passive(m, d) as v:
    with v.lock():  # agentview 와 비슷한 방향 (로봇 정면, 위에서)
        v.cam.lookat[:] = [-0.1, 0.0, 0.95]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.5, 180.0, -35.0
    while v.is_running():
        env.set_init_state(init)
        env.env.timestep, env.env.done = 0, False  # 반복 재생 시 horizon 초과로 step 이 막히지 않게
        v.sync()
        first = None
        for t, a in enumerate([DUMMY] * args.wait + list(actions)):
            if not v.is_running():
                break
            t0 = time.time()
            env.step(a)
            v.sync()
            if first is None and t >= args.wait and env.env._check_success():
                first = t - args.wait + 1
            time.sleep(max(0.0, dt - (time.time() - t0)))
        print(f"끝: 성공={bool(env.env._check_success())} (첫 성공 스텝 {first})")
        if not args.loop:
            while v.is_running():
                time.sleep(0.1)
            break
        time.sleep(2.0)
env.close()
