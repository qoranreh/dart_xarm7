"""MimicGen 방식으로 xArm7 target 데모를 만든다: Panda 데모의 '명령 목표 포즈' 궤적을 xArm7 이 따라가며 액션을 새로 계산.

MimicGen (NVlabs/mimicgen) 과 같은 규칙:
  - 목표 포즈 = 원본 데모의 target_pose = 그 스텝의 현재 grip_site 포즈 + 액션×(0.05m, 0.5rad)   (datagen/data_generator.py:298-311)
  - 액션 = clip((목표 위치 − 현재 위치)/0.05), clip(axis-angle(목표 회전·현재 회전ᵀ)/0.5)  (env_interfaces/robosuite.py target_pose_to_action)
  - 그리퍼 명령은 원본 데모 것을 그대로 복사, 끝에서 성공한 궤적만 채택
  - 같은 시작 장면이라 물체 기준 변환(transform_source_data_segment_using_object_pose)은 항등 → 생략
  - MimicGen 기본 설정의 action_noise(0.05) 는 넣지 않음 (깨끗한 데모 1개가 목적)
그리퍼 좌표계 보정: Panda 는 grip_site 가 right_hand 대비 접근축으로 -90° 돌아가 있으므로,
  xArm7 의 목표 자세는 "Panda right_hand 목표 자세" 로 바꿔서 준다 (손가락 방향이 물리적으로 같아지도록).

  source ~/hayeon/dart_xarm7/env.sh && source ~/hayeon/dart_xarm7/client_env.sh
  $LIBERO_PY ~/hayeon/dart_xarm7/make_retarget_demo.py
"""
import argparse
import json
import os
import pathlib

import numpy as np
from robosuite.utils import transform_utils as T

import libero_xarm7 as X
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

p = argparse.ArgumentParser()
p.add_argument("--suite", default="libero_spatial")
p.add_argument("--task", type=int, default=0)
p.add_argument("--demo", default="demo_0")
p.add_argument("--hdf5-npz", default=os.path.expandvars("$DART_ROOT/demos/spatial_t0_hdf5_demo0-4.npz"))
p.add_argument("--wait", type=int, default=10)
p.add_argument("--out", default=os.path.expandvars("$DART_ROOT/demos"))
p.add_argument("--rot-clip", choices=["none", "quantile"], default="none",
               help="quantile: 회전 액션을 pi05_libero norm stats 의 q01~q99 로 제한 (MimicGen 원래 규칙 아님, 비교용 변형)")
args = p.parse_args()
SUFFIX = "XArm7_retarget" if args.rot_clip == "none" else "XArm7_retarget_rotq"

DUMMY = [0.0] * 6 + [-1.0]
z = np.load(args.hdf5_npz)
src_actions = z[f"{args.demo}/actions"].astype(np.float64)
state0 = z[f"{args.demo}/states"][0]
suite = benchmark.get_benchmark_dict()[args.suite]()
task = suite.get_task(args.task)
bddl = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
out = pathlib.Path(args.out)
tag = f"{args.suite}_t{args.task}_{args.demo}"


def make(robot):
    env = OffScreenRenderEnv(bddl_file_name=bddl, robots=[robot], camera_heights=256, camera_widths=256,
                             camera_names=["agentview", "robot0_eye_in_hand"])
    env.seed(0)
    env.reset()
    return env


def frames(env):
    """(grip_site 위치, grip_site 회전, right_hand 회전) — grip_site 가 OSC 제어 프레임."""
    sim, rb = env.env.sim, env.env.robots[0]
    m, d = sim.model._model, sim.data._data
    return (d.site_xpos[rb.eef_site_id].copy(), d.site_xmat[rb.eef_site_id].reshape(3, 3).copy(),
            d.xmat[m.body(rb.robot_model.eef_name).id].reshape(3, 3).copy())


# 1) Panda: 원본 액션을 재생하며 MimicGen 의 target_pose (명령 목표 포즈) 궤적을 기록
panda = make("Panda")
obs = panda.set_init_state(state0)
for _ in range(args.wait):
    obs, _, _, _ = panda.step(DUMMY)
ctrl = panda.env.robots[0].controller
max_dpos, max_drot = float(ctrl.output_max[0]), float(ctrl.output_max[3])
tgt_pos, tgt_hand_rot, off_err = [], [], []
for t, a in enumerate(src_actions):
    pp, Rg, Rh = frames(panda)
    if t == 0:
        R_off_p = Rg.T @ Rh  # Panda: grip_site → right_hand (고정)
    off_err.append(float(np.abs(Rg.T @ Rh - R_off_p).max()))
    Rt = T.quat2mat(T.axisangle2quat(a[3:6] * max_drot)) @ Rg  # action_to_target_pose 와 같은 식
    tgt_pos.append(pp + a[:3] * max_dpos)
    tgt_hand_rot.append(Rt @ R_off_p)  # Panda right_hand 의 목표 자세
    obs, _, _, _ = panda.step(a)
panda_success = bool(panda.env._check_success())

# 2) xArm7: 같은 시작 장면에서 목표 포즈 궤적을 따라가며 액션 계산
xarm = make("XArm7")
flat, tinfo = X.transfer_init_state(panda, state0, xarm)
obs = xarm.set_init_state(flat)
for _ in range(args.wait):
    obs, _, _, _ = xarm.step(DUMMY)
_, Rg_x, Rh_x = frames(xarm)
R_off_x = Rg_x.T @ Rh_x  # xArm7: grip_site → right_hand (단위행렬이어야 함)

rec = {k: [] for k in ("image", "wrist_image", "state", "actions", "eef_pos", "eef_quat", "gripper_qpos_raw", "pad_gap",
                       "track_err_pos", "track_err_rot", "clipped", "pos_raw", "rot_raw")}
NS = json.load(open(os.path.expandvars("$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json")))["norm_stats"]["actions"]
ROT_Q01, ROT_Q99 = np.array(NS["q01"][3:6]), np.array(NS["q99"][3:6])
first_success = None
for t in range(len(src_actions)):
    px, Rgx, _ = frames(xarm)
    tgt_grip_rot = tgt_hand_rot[t] @ R_off_x.T
    dpos_raw = (tgt_pos[t] - px) / max_dpos
    drot_raw = T.quat2axisangle(T.mat2quat(tgt_grip_rot @ Rgx.T)) / max_drot
    drot = np.clip(drot_raw, -1, 1)
    if args.rot_clip == "quantile":
        drot = np.clip(drot, ROT_Q01, ROT_Q99)
    a = np.concatenate([np.clip(dpos_raw, -1, 1), drot, [src_actions[t][6]]])
    rec["image"].append(obs["agentview_image"][::-1, ::-1].copy())
    rec["wrist_image"].append(obs["robot0_eye_in_hand_image"][::-1, ::-1].copy())
    rec["state"].append(X.policy_state(xarm, obs))
    rec["actions"].append(a.astype(np.float32))
    rec["eef_pos"].append(obs["robot0_eef_pos"].copy())
    rec["eef_quat"].append(obs["robot0_eef_quat"].copy())
    rec["gripper_qpos_raw"].append(obs["robot0_gripper_qpos"].copy())
    rec["pad_gap"].append(X.pad_surface_gap(xarm))
    rec["clipped"].append(bool(np.any(np.abs(np.concatenate([dpos_raw, drot_raw])) > 1)))
    rec["pos_raw"].append(dpos_raw.copy())  # 자르기 전 액션 (기록용, 액션 계산과 무관)
    rec["rot_raw"].append(drot_raw.copy())
    obs, _, done, _ = xarm.step(a)
    pa, Ra, _ = frames(xarm)  # 스텝 뒤 목표 포즈와의 차이 (추종 오차)
    rec["track_err_pos"].append(float(np.linalg.norm(tgt_pos[t] - pa)))
    rec["track_err_rot"].append(float(np.linalg.norm(T.quat2axisangle(T.mat2quat(tgt_grip_rot @ Ra.T)))))
    if done and first_success is None:
        first_success = t + 1
success = bool(xarm.env._check_success())
arr = {k: np.array(v) for k, v in rec.items()}
# 이 프로세스에서 libero_xarm7 가 실제로 읽은 조건 값 (실험 6: 환경변수가 데모 생성에 적용됐는지 기록)
ENV_META = f"XARM7_COLOR={X.COLOR}, XARM7_DYNAMICS={X.DYNAMICS}"
np.savez_compressed(out / f"{tag}_{SUFFIX}.npz", **arr, task=task.language,
                    target_pos=np.array(tgt_pos), target_hand_rot=np.array(tgt_hand_rot),
                    xarm7_color=X.COLOR, xarm7_dynamics=X.DYNAMICS, env_meta=ENV_META)

src = np.load(out / f"{tag}_Panda.npz")
summ = {
    "method": "MimicGen-style retarget (target_pose tracking, gripper copied, no noise)" + (" + rotation clipped to norm q01..q99" if args.rot_clip == "quantile" else "")
              + f" [{ENV_META}]",
    "xarm7_color": X.COLOR, "xarm7_dynamics": X.DYNAMICS,
    "panda_success": panda_success, "xarm_success_at_end": success, "xarm_first_success_step": first_success,
    "frames": len(src_actions), "R_off_panda_const_err": max(off_err),
    "R_off_xarm_is_identity_err": float(np.abs(R_off_x - np.eye(3)).max()),
    "panda_actions_match_previous_recording": bool(np.allclose(src["actions"], src_actions.astype(np.float32))),
    "steps_with_clipped_action": int(arr["clipped"].sum()),
    "track_err_pos_cm": {"mean": float(arr["track_err_pos"].mean() * 100), "max": float(arr["track_err_pos"].max() * 100)},
    "track_err_rot_deg": {"mean": float(np.degrees(arr["track_err_rot"].mean())), "max": float(np.degrees(arr["track_err_rot"].max()))},
    "eef_vs_panda_cm": {"mean": float(np.linalg.norm(arr["eef_pos"] - src["eef_pos"], axis=1).mean() * 100),
                        "max": float(np.linalg.norm(arr["eef_pos"] - src["eef_pos"], axis=1).max() * 100)},
    "action_absdiff_vs_source_mean": np.abs(arr["actions"] - src["actions"]).mean(0).round(4).tolist(),
    "action_absdiff_vs_source_max": np.abs(arr["actions"] - src["actions"]).max(0).round(4).tolist(),
    "transfer": {k: v for k, v in tinfo.items() if k != "copied_joints"},
}
(out / f"{tag}_{SUFFIX}_summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
print(json.dumps(summ, indent=1, ensure_ascii=False))
panda.close()
xarm.close()
