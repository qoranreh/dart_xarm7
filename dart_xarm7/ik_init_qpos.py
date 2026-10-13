"""LIBERO MountedPanda 의 초기 그리퍼 포즈를 목표로 xArm7 IK 를 풀어 init_qpos 를 구한다.

목표: grip_site 위치 = Panda grip_site 위치, right_hand 자세 = Panda right_hand 자세.
(grip_site 자세를 맞추면 안 된다 — Panda 는 grip_site 가 right_hand 대비 접근축으로 -90° 돌아가 있고 xArm7 은 같아서,
 grip_site 를 맞추면 손가락 방향이 90° 어긋난다. right_hand 를 맞추면 손가락 축과 eef_quat 관측값이 둘 다 Panda 와 같아진다.
 xArm7 은 grip_site 와 right_hand 의 자세가 같으므로 grip_site 자세 목표 = Panda right_hand 자세.)

같은 씬(libero_spatial task 0)에서 Panda env 를 만들어 목표 포즈와 비교용 프레임들을 기록하고,
xArm7 env 에서 damped least squares IK (null-space 로 기본 자세 쪽 유도, 관절 한계 클립) 를 푼다.

  source ~/hayeon/dart_xarm7/env.sh && source ~/hayeon/dart_xarm7/client_env.sh
  $LIBERO_PY ~/hayeon/dart_xarm7/ik_init_qpos.py
"""
import json
import pathlib

import mujoco
import numpy as np

import libero_xarm7
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
BDDL = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)


def make(robot):
    env = OffScreenRenderEnv(bddl_file_name=BDDL, robots=[robot], camera_heights=64, camera_widths=64,
                             camera_names=["agentview"])
    env.seed(7)
    env.reset()
    return env


def frames(env):
    """grip_site / right_hand / 손목 카메라 / 손가락 패드 중심의 월드 포즈."""
    sim, rb = env.env.sim, env.env.robots[0]
    m, d = sim.model._model, sim.data._data
    sid = rb.eef_site_id
    hand = m.body(rb.robot_model.eef_name).id
    cam = m.camera("robot0_eye_in_hand").id
    pad = lambda k: np.mean([d.geom_xpos[m.geom(g).id] for g in rb.gripper.important_geoms[k]], axis=0)
    return {
        "grip_pos": d.site_xpos[sid].copy(),
        "grip_mat": d.site_xmat[sid].reshape(3, 3).copy(),
        "hand_mat": d.xmat[hand].reshape(3, 3).copy(),
        "cam_pos": d.cam_xpos[cam].copy(),
        "cam_mat": d.cam_xmat[cam].reshape(3, 3).copy(),
        "finger_dir": (lambda v: v / np.linalg.norm(v))(pad("left_fingerpad") - pad("right_fingerpad")),
    }


def rot_err(R_goal, R):
    """R → R_goal 회전 오차 (월드 프레임 axis-angle 벡터)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (R_goal @ R.T).flatten())
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def ik(env, pos_goal, mat_goal, q_nominal, iters=2000):
    sim, rb = env.env.sim, env.env.robots[0]
    m, d = sim.model._model, sim.data._data
    qidx = np.array(rb._ref_joint_pos_indexes)
    didx = np.array(rb._ref_joint_vel_indexes)
    jids = [m.joint(n).id for n in rb.robot_model.joints]
    lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
    sid = rb.eef_site_id
    q = q_nominal.copy()
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for it in range(iters):
        d.qpos[qidx] = q
        mujoco.mj_forward(m, d)
        e = np.concatenate([pos_goal - d.site_xpos[sid], rot_err(mat_goal, d.site_xmat[sid].reshape(3, 3))])
        if np.linalg.norm(e[:3]) < 1e-5 and np.linalg.norm(e[3:]) < 1e-4:
            break
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        J = np.vstack([jacp[:, didx], jacr[:, didx]])
        lam = 1e-4
        Jp = J.T @ np.linalg.inv(J @ J.T + lam * np.eye(6))
        dq = Jp @ e + 0.05 * (np.eye(7) - Jp @ J) @ (q_nominal - q)  # null-space: 기본 자세 쪽으로
        q = np.clip(q + 0.5 * dq, lo, hi)
    return q, np.linalg.norm(e[:3]), np.linalg.norm(e[3:]), it


panda = make("Panda")
P = frames(panda)
panda.close()

xarm = make("XArm7")
X0 = frames(xarm)
q0 = np.array(xarm.env.robots[0].init_qpos)
# 시드 여러 개 (손목 joint7 을 ±π 돌린 것 포함) → 오차가 수렴한 것 중 시드에서 가장 덜 움직인 해
best = None
for dj in (0.0, np.pi, -np.pi):
    seed = q0 + np.array([0, 0, 0, 0, 0, 0, dj])
    q, ep, er, it = ik(xarm, P["grip_pos"], P["hand_mat"], seed)
    score = (ep > 1e-4 or er > 1e-3, np.linalg.norm(q - seed))
    print(f"seed j7+{dj:+.2f}: pos_err={ep:.2e} rot_err={er:.2e} iters={it} |q-seed|={score[1]:.3f} q={np.round(q, 3)}")
    if best is None or score < best[0]:
        best = (score, q, ep, er, it)
_, q, ep, er, it = best
sim = xarm.env.sim
sim.data._data.qpos[np.array(xarm.env.robots[0]._ref_joint_pos_indexes)] = q
mujoco.mj_forward(sim.model._model, sim.data._data)
X = frames(xarm)

def ang(Ra, Rb):
    return float(np.degrees(np.linalg.norm(rot_err(Ra, Rb))))

hand_quat = lambda R: (lambda q: np.round([q[1], q[2], q[3], q[0]], 4).tolist())(  # robosuite xyzw
    (lambda q: (mujoco.mju_mat2Quat(q, R.flatten()), q)[1])(np.zeros(4)))
out = {
    "ik_qpos": np.round(q, 6).tolist(),
    "ik_iters": it, "ik_pos_err_m": ep, "ik_rot_err_rad": er,
    "panda_grip_pos": np.round(P["grip_pos"], 4).tolist(),
    "xarm_grip_pos(before→after)": [np.round(X0["grip_pos"], 4).tolist(), np.round(X["grip_pos"], 4).tolist()],
    "grip_site 자세차 deg (before→after)": [ang(P["grip_mat"], X0["grip_mat"]), ang(P["grip_mat"], X["grip_mat"])],
    "손가락 벌어지는 방향 (panda / xarm before / after)": [np.round(P["finger_dir"], 3).tolist(),
                                                    np.round(X0["finger_dir"], 3).tolist(), np.round(X["finger_dir"], 3).tolist()],
    "eef_quat 관측값 xyzw (panda / xarm after)": [hand_quat(P["hand_mat"]), hand_quat(X["hand_mat"])],
    "right_hand 자세차 deg (after)": ang(P["hand_mat"], X["hand_mat"]),
    "손목카메라: grip_site 기준 위치 (panda / xarm)": [np.round(P["grip_mat"].T @ (P["cam_pos"] - P["grip_pos"]), 4).tolist(),
                                              np.round(X["grip_mat"].T @ (X["cam_pos"] - X["grip_pos"]), 4).tolist()],
    "손목카메라: grip_site 기준 자세차 deg": ang(P["grip_mat"].T @ P["cam_mat"], X["grip_mat"].T @ X["cam_mat"]),
}
print(json.dumps(out, indent=1, ensure_ascii=False))
pathlib.Path(__file__).with_name("ik_init_qpos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
xarm.close()
