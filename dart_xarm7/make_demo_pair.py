"""LIBERO hdf5 데모 1개의 액션을 Panda(source) 와 xArm7(target) 에 재생하며 256×256 관측을 녹화한다.

공식 DART 데이터(SNUMPR/dart-libero-viewpoints)와 같은 절차:
  env.reset() → set_init_state(hdf5 states[0]) → 대기 10스텝(LIBERO_DUMMY_ACTION) → 액션마다 [관측 기록 → step]
  이미지는 공식 평가와 같이 180° 회전, state = eef_pos + axis-angle + gripper_qpos (8차원)
xArm7 은 같은 시작 장면을 transfer_init_state 로 만든다 (물체 상태 복사 + Panda 그리퍼 포즈에 IK), state 의 그리퍼는 Panda 형식으로 환산.

  source ~/hayeon/dart_xarm7/env.sh && source ~/hayeon/dart_xarm7/client_env.sh
  $LIBERO_PY ~/hayeon/dart_xarm7/make_demo_pair.py
"""
import argparse
import json
import os
import pathlib

import numpy as np

import libero_xarm7 as X
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

p = argparse.ArgumentParser()
p.add_argument("--suite", default="libero_spatial")
p.add_argument("--task", type=int, default=0)
p.add_argument("--demo", default="demo_0")
p.add_argument("--hdf5-npz", default=os.path.expandvars("$DART_ROOT/demos/spatial_t0_hdf5_demo0-4.npz"))
p.add_argument("--wait", type=int, default=10)
p.add_argument("--extra-steps", type=int, default=0, help="액션이 끝난 뒤 성공 판정까지 더 기다릴 스텝 (녹화는 안 함)")
p.add_argument("--out", default=os.path.expandvars("$DART_ROOT/demos"))
p.add_argument("--robots", default="Panda,XArm7")
args = p.parse_args()

DUMMY = [0.0] * 6 + [-1.0]
z = np.load(args.hdf5_npz)
actions = z[f"{args.demo}/actions"].astype(np.float64)
state0 = z[f"{args.demo}/states"][0]
suite = benchmark.get_benchmark_dict()[args.suite]()
task = suite.get_task(args.task)
bddl = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
tag = f"{args.suite}_t{args.task}_{args.demo}"


def make(robot):
    env = OffScreenRenderEnv(bddl_file_name=bddl, robots=[robot], camera_heights=256, camera_widths=256,
                             camera_names=["agentview", "robot0_eye_in_hand"])
    env.seed(0)
    env.reset()
    return env


envs, summary = {}, {"task": task.language, "demo": args.demo, "n_actions": len(actions), "wait": args.wait}
for robot in args.robots.split(","):
    env = envs[robot] = make(robot)
    if robot == "Panda":
        obs = env.set_init_state(state0)
    else:
        flat, info = X.transfer_init_state(envs["Panda"], state0, env)
        obs = env.set_init_state(flat)
        summary[f"{robot}_transfer"] = {k: v for k, v in info.items() if k != "copied_joints"} | {"n_copied_joints": len(info["copied_joints"])}
        np.save(out / f"{tag}_{robot}_init_state.npy", flat)
    for _ in range(args.wait):
        obs, _, _, _ = env.step(DUMMY)

    rec = {k: [] for k in ("image", "wrist_image", "state", "actions", "eef_pos", "eef_quat", "gripper_qpos_raw", "pad_gap", "done")}
    first_success = None
    for t, a in enumerate(actions):
        rec["image"].append(obs["agentview_image"][::-1, ::-1].copy())
        rec["wrist_image"].append(obs["robot0_eye_in_hand_image"][::-1, ::-1].copy())
        rec["state"].append(X.policy_state(env, obs))
        rec["actions"].append(a.astype(np.float32))
        rec["eef_pos"].append(obs["robot0_eef_pos"].copy())
        rec["eef_quat"].append(obs["robot0_eef_quat"].copy())
        rec["gripper_qpos_raw"].append(obs["robot0_gripper_qpos"].copy())
        rec["pad_gap"].append(X.pad_surface_gap(env))
        obs, _, done, _ = env.step(a)
        rec["done"].append(bool(done))
        if done and first_success is None:
            first_success = t + 1
    success_end = bool(env.env._check_success())
    extra_success = None
    for k in range(args.extra_steps):
        obs, _, done, _ = env.step(DUMMY)
        if env.env._check_success():
            extra_success = k + 1
            break
    arr = {k: np.array(v) for k, v in rec.items()}
    np.savez_compressed(out / f"{tag}_{robot}.npz", **arr, task=task.language)
    summary[robot] = {"success_at_end": success_end, "first_success_step": first_success,
                      "success_after_extra_wait": extra_success, "frames": len(actions)}
    print(robot, json.dumps(summary[robot]))

if "Panda" in envs and len(envs) > 1:
    P, Q = (np.load(out / f"{tag}_{r}.npz") for r in ("Panda", "XArm7"))
    dpos = np.linalg.norm(P["eef_pos"] - Q["eef_pos"], axis=1)
    summary["eef_pos_diff_m"] = {"start": float(dpos[0]), "mean": float(dpos.mean()), "max": float(dpos.max()), "argmax": int(dpos.argmax())}
    summary["state_diff_max_per_dim"] = np.abs(P["state"] - Q["state"]).max(0).round(4).tolist()
(out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
print(json.dumps({k: v for k, v in summary.items() if k not in ("Panda", "XArm7")}, indent=1, ensure_ascii=False))
for e in envs.values():
    e.close()
