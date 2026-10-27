"""실험 7 평가 클라이언트 — eval_xarm7.py 에 LIBERO-Plus 식 섭동 3가지를 더한 것 (eval_xarm7.py 는 수정하지 않음).

eval_xarm7.py 와 같은 것: make_env, 초기 상태 이식 (libero_xarm7.transfer_init_state), env.seed(7), np.random.seed(7),
대기 10스텝, 5스텝마다 재계획, 이미지 180° 회전 + 224 resize_with_pad, max_steps, JSONL 기록.
다른 것:
  --init-offset  : 초기 상태 번호를 offset ~ offset+trials-1 로 (실험 0~6 은 0~9 를 씀). 기록의 episode = 실제 초기 상태 번호.
  --perturb/--level (LIBERO-Plus, arXiv 2510.13626, github.com/sylvestf/LIBERO-plus 의 정의를 xArm7 클라이언트에 새로 구현):
    none     : 섭동 없음 → eval_xarm7.py 와 같은 동작.
    cam_dist : C1. agentview 를 광축을 따라 옮겨 장면 중심까지 거리를 level 배로 (LIBERO-Plus 범위 1.01~2.00).
    cam_azim : C2. 장면 중심을 지나는 연직축으로 agentview 를 ±level 도 회전 (부호는 에피소드 난수, 고도 0, LIBERO-Plus 15~75°).
    joint    : R. xArm7 팔 qpos 에 무작위 단위 방향 × level rad (LIBERO-Plus new_init.py 와 같은 식, 범위 0.1~0.5).
               팔·그리퍼가 물체·테이블을 새로 파고들면 같은 난수열에서 다시 뽑는다 (최대 20회, 횟수 기록).
  장면 중심 = agentview 광축과 테이블 윗면 평면의 교점. 손목 카메라는 바꾸지 않는다.
  섭동 난수는 default_rng([seed, task_id, init_idx]) — 모든 모델이 같은 섭동을 받고, 전역 np.random 상태는 건드리지 않는다.
  영상은 태스크마다 첫 초기 상태(ep == init_offset) 만 저장.

  source ~/hayeon/dart_xarm7/env.sh && source ~/hayeon/dart_xarm7/client_env.sh
  $LIBERO_PY ~/hayeon/dart_xarm7/eval_perturb.py --perturb cam_dist --level 1.2 --init-offset 20 --trials 5 --task-ids 0 1 ...
"""
from __future__ import annotations

import collections
import dataclasses
import json
import logging
import os
import pathlib
import time

import imageio
import mujoco
import numpy as np
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

import libero_xarm7 as X
from libero.libero import benchmark

from eval_xarm7 import LIBERO_DUMMY_ACTION, MAX_STEPS, make_env

PERTURBS = ("none", "cam_dist", "cam_azim", "joint")


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 9100
    robot: str = "XArm7"
    task_suite_name: str = "libero_spatial"
    task_ids: tuple[int, ...] = (0,)
    trials: int = 10
    init_offset: int = 0
    perturb: str = "none"
    level: float = 0.0
    shard: int = 0
    nshard: int = 1
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    seed: int = 7
    tag: str = ""
    results_path: str = "results.jsonl"
    video_out_path: str = "videos"


# ---------------------------------------------------------------- 섭동
def table_top_z(sim):
    """테이블 윗면 높이: 이름에 'table' 이 든 상자(box) geom 중 가장 넓은 것의 윗면."""
    m, d = sim.model._model, sim.data._data
    best = None
    for g in range(m.ngeom):
        name = (m.geom(g).name or "") + " " + m.body(m.geom_bodyid[g]).name
        if "table" in name.lower() and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            area = m.geom_size[g][0] * m.geom_size[g][1]
            if best is None or area > best[0]:
                best = (area, float(d.geom_xpos[g][2] + m.geom_size[g][2]), name.strip())
    assert best is not None, "테이블 geom 을 찾지 못함"
    return best[1], best[2]


def apply_camera(env, kind, level, sign):
    """agentview 카메라를 바꾼다 (월드에 붙은 카메라만). 반환: 기록용 정보."""
    sim = env.env.sim
    m, d = sim.model._model, sim.data._data
    cid = m.camera("agentview").id
    assert m.cam_bodyid[cid] == 0, "agentview 가 월드 바디에 붙어 있지 않음"
    p, R = d.cam_xpos[cid].copy(), d.cam_xmat[cid].reshape(3, 3).copy()
    fwd = -R[:, 2]  # MuJoCo 카메라는 자기 -z 방향을 본다
    zt, tname = table_top_z(sim)
    t = (zt - p[2]) / fwd[2]  # 광축 위 장면 중심까지 거리
    c = p + t * fwd
    if kind == "cam_dist":
        p2, R2 = c - level * t * fwd, R
    elif kind == "cam_azim":
        a = np.deg2rad(sign * level)
        Rz = np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
        p2, R2 = c + Rz @ (p - c), Rz @ R
    else:
        raise ValueError(kind)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R2.flatten())
    m.cam_pos[cid], m.cam_quat[cid] = p2, q
    sim.forward()
    return {"cam_pos0": p.round(4).tolist(), "cam_pos": p2.round(4).tolist(), "center": c.round(4).tolist(),
            "dist0": round(float(t), 4), "dist": round(float(np.linalg.norm(p2 - c)), 4), "sign": int(sign), "table": tname}


def _robot_env_penetrations(sim):
    """팔·그리퍼가 환경(물체·테이블 등, 마운트 제외)을 파고든 접촉 수."""
    m, d = sim.model._model, sim.data._data
    robot = lambda b: b.startswith(("robot0_", "gripper0_"))
    n = 0
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
        if robot(b1) != robot(b2) and not (b1.startswith("mount0_") or b2.startswith("mount0_")) and c.dist < -1e-4:
            n += 1
    return n


def perturb_joints(env, level, rng, max_tries=20):
    """현재 sim 상태(이식 직후)의 팔 qpos 에 level × 단위 무작위 방향을 더한 flattened state 를 만든다."""
    sim, rb = env.env.sim, env.env.robots[0]
    m = sim.model._model
    qidx = np.array(rb._ref_joint_pos_indexes)
    jids = [m.joint(n).id for n in rb.robot_model.joints]
    lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
    limited = m.jnt_limited[jids].astype(bool)
    q0 = sim.data._data.qpos[qidx].copy()
    pos0, rot0 = X.gripper_pose(env)
    base = _robot_env_penetrations(sim)
    for k in range(max_tries):
        u = rng.standard_normal(len(q0))
        q = q0 + level * u / np.linalg.norm(u)
        q = np.where(limited, np.clip(q, lo, hi), q)
        sim.data._data.qpos[qidx] = q
        sim.forward()
        if _robot_env_penetrations(sim) <= base:
            break
    pos1, rot1 = X.gripper_pose(env)
    ang = np.degrees(np.arccos(np.clip((np.trace(rot0.T @ rot1) - 1) / 2, -1, 1)))
    info = {"dq": (q - q0).round(5).tolist(), "dq_norm": round(float(np.linalg.norm(q - q0)), 5), "tries": k + 1,
            "eef_shift_cm": round(float(100 * np.linalg.norm(pos1 - pos0)), 2), "eef_rot_deg": round(float(ang), 2)}
    return sim.get_state().flatten(), info


def prepare_episode(env, panda, init, args, task_id, ep):
    """초기 상태 이식 → (joint 섭동) → set_init_state → (카메라 섭동). 반환: (obs, 이식 정보, 섭동 정보)."""
    rng = np.random.default_rng([args.seed, task_id, ep])
    flat, tinfo = X.transfer_init_state(panda, init, env)  # 내부에서 env.reset()
    pinfo = {}
    if args.perturb == "joint":
        flat, pinfo = perturb_joints(env, args.level, rng)
    obs = env.set_init_state(flat)
    if args.perturb in ("cam_dist", "cam_azim"):
        pinfo = apply_camera(env, args.perturb, args.level, sign=int(rng.choice([-1, 1])))
    return obs, tinfo, pinfo


# ---------------------------------------------------------------- 평가 루프 (eval_xarm7.main 과 같은 흐름)
def main(args: Args) -> None:
    assert args.perturb in PERTURBS, args.perturb
    assert args.robot != "Panda", "실험 7 은 xArm7 전용"
    logging.info("eval env: XARM7_COLOR=%s XARM7_DYNAMICS=%s robot=%s ckpt=%s tag=%s perturb=%s level=%s init=%d..%d",
                 X.COLOR, X.DYNAMICS, args.robot, os.environ.get("EVAL_CKPT_INFO", "(미지정)"), args.tag,
                 args.perturb, args.level, args.init_offset, args.init_offset + args.trials - 1)
    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = MAX_STEPS[args.task_suite_name]
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    jobs = [(t, e) for t in args.task_ids for e in range(args.init_offset, args.init_offset + args.trials)]
    jobs = jobs[args.shard::args.nshard]
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    out = open(args.results_path, "a")
    envs = {}
    for task_id, ep in jobs:
        task = suite.get_task(task_id)
        if task_id not in envs:
            for e in envs.values():
                for x in e:
                    x.close()
            envs = {task_id: (make_env(task, args.robot, 256, args.seed), make_env(task, "Panda", 64, args.seed))}
        env, panda = envs[task_id]
        init = suite.get_task_init_states(task_id)[ep]
        t_start = time.time()
        obs, tinfo, pinfo = prepare_episode(env, panda, init, args, task_id, ep)
        action_plan = collections.deque()
        t, done, frames, err = 0, False, [], None
        while t < max_steps + args.num_steps_wait:
            try:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size))
                if ep == args.init_offset:
                    frames.append(np.concatenate([img, wrist], axis=1))
                if not action_plan:
                    element = {"observation/image": img, "observation/wrist_image": wrist,
                               "observation/state": X.policy_state(env, obs), "prompt": str(task.language)}
                    chunk = client.infer(element)["actions"]
                    assert len(chunk) >= args.replan_steps
                    action_plan.extend(chunk[: args.replan_steps])
                obs, _, done, _ = env.step(action_plan.popleft().tolist())
                if done:
                    break
                t += 1
            except Exception as e:  # noqa: BLE001
                err = repr(e)
                logging.error("exception: %s", err)
                break
        rec = {"tag": args.tag, "robot": args.robot, "task_id": task_id, "episode": ep, "task": task.language,
               "success": bool(done), "steps": t - args.num_steps_wait, "sec": round(time.time() - t_start, 1), "error": err,
               "ik_pos_err": tinfo["ik_pos_err"], "xarm7_color": X.COLOR, "xarm7_dynamics": X.DYNAMICS,
               "perturb": args.perturb, "level": args.level, "pinfo": pinfo}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        logging.info("%s task %d ep %d success=%s steps=%d %.0fs %s", args.tag, task_id, ep, done, rec["steps"], rec["sec"], pinfo)
        if ep == args.init_offset and frames:
            imageio.mimwrite(pathlib.Path(args.video_out_path) / f"{args.tag}_t{task_id}_ep{ep}_{'success' if done else 'failure'}.mp4",
                             frames, fps=10)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(tyro.cli(Args))
