"""Panda 와 xArm7 에 같은 7차원 액션 시퀀스를 넣고 그리퍼(eef) 움직임과 손가락 벌림을 비교한다.

  source ~/hayeon/dart_xarm7/env.sh && source ~/hayeon/dart_xarm7/client_env.sh
  $LIBERO_PY ~/hayeon/dart_xarm7/action_space_check.py
"""
import json
import pathlib

import numpy as np

import libero_xarm7  # noqa: F401
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils import transform_utils as T

s = benchmark.get_benchmark_dict()["libero_spatial"]()
t = s.get_task(0)
BDDL = str(pathlib.Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file)

# (이름, 스텝 수, 액션 [dx dy dz droll dpitch dyaw grip])  — 액션 1.0 = OSC 목표 0.05m / 0.5rad 이동
PHASES = [
    ("settle", 10, [0, 0, 0, 0, 0, 0, -1]),
    ("+x", 10, [0.3, 0, 0, 0, 0, 0, -1]),
    ("+y", 10, [0, 0.3, 0, 0, 0, 0, -1]),
    ("+z", 10, [0, 0, 0.3, 0, 0, 0, -1]),
    ("+yaw", 10, [0, 0, 0, 0, 0, 0.3, -1]),
    ("+roll", 10, [0, 0, 0, 0.3, 0, 0, -1]),
    ("hold", 10, [0, 0, 0, 0, 0, 0, -1]),
    ("close", 25, [0, 0, 0, 0, 0, 0, 1]),
    ("open", 25, [0, 0, 0, 0, 0, 0, -1]),
]


def run(robot):
    env = OffScreenRenderEnv(bddl_file_name=BDDL, robots=[robot], camera_heights=64, camera_widths=64,
                             camera_names=["agentview"])
    env.seed(7)
    obs = env.reset()
    inner = env.env
    rb = inner.robots[0]
    m, d = inner.sim.model._model, inner.sim.data._data
    pads = [[m.geom(g).id for g in rb.gripper.important_geoms[k]] for k in ("left_fingerpad", "right_fingerpad")]
    width = lambda: float(np.linalg.norm(d.geom_xpos[pads[0]].mean(0) - d.geom_xpos[pads[1]].mean(0)))
    log = {"pos": [obs["robot0_eef_pos"].copy()], "quat": [obs["robot0_eef_quat"].copy()], "width": [width()], "phase": ["start"]}
    for name, n, a in PHASES:
        for _ in range(n):
            obs, _, _, _ = env.step(np.array(a, dtype=float))
            log["pos"].append(obs["robot0_eef_pos"].copy())
            log["quat"].append(obs["robot0_eef_quat"].copy())
            log["width"].append(width())
            log["phase"].append(name)
    info = {"action_dim": int(inner.action_dim), "action_spec": [x.tolist() for x in inner.action_spec],
            "controller": rb.controller_config["type"], "gripper_dof": int(rb.gripper.dof)}
    env.close()
    return {k: np.array(v) if k != "phase" else v for k, v in log.items()}, info


def phase_delta(log, name):
    idx = [i for i, p in enumerate(log["phase"]) if p == name]
    i0, i1 = idx[0] - 1, idx[-1]
    dp = log["pos"][i1] - log["pos"][i0]
    R0, R1 = T.quat2mat(log["quat"][i0]), T.quat2mat(log["quat"][i1])
    drot = T.quat2axisangle(T.mat2quat(R1 @ R0.T))  # 월드 프레임 회전 변화
    return dp, drot


L, I = {}, {}
for r in ("Panda", "XArm7"):
    L[r], I[r] = run(r)

res = {"spec": I, "phases": {}}
for name, _, _ in PHASES[1:6]:
    (dpP, drP), (dpX, drX) = phase_delta(L["Panda"], name), phase_delta(L["XArm7"], name)
    res["phases"][name] = {
        "Panda dpos(m)": np.round(dpP, 4).tolist(), "XArm7 dpos(m)": np.round(dpX, 4).tolist(),
        "Panda drot(rad,world)": np.round(drP, 4).tolist(), "XArm7 drot(rad,world)": np.round(drX, 4).tolist(),
    }
res["eef_pos 최대 차이(m), 전체 궤적"] = float(np.abs(L["Panda"]["pos"] - L["XArm7"]["pos"]).max())
for r in ("Panda", "XArm7"):
    w, ph = L[r]["width"], L[r]["phase"]
    ci = [i for i, p in enumerate(ph) if p == "close"]
    oi = [i for i, p in enumerate(ph) if p == "open"]
    closed_at = next((k for k, i in enumerate(ci) if w[i] <= w[ci[-1]] + 0.002), None)
    opened_at = next((k for k, i in enumerate(oi) if w[i] >= w[oi[-1]] - 0.002), None)
    res[f"{r} 손가락 패드 간격(m)"] = {"열림": round(float(w[ci[0] - 1]), 4), "닫힘": round(float(w[ci[-1]]), 4),
                                    "닫히는데 걸린 스텝": None if closed_at is None else closed_at + 1,
                                    "다시 열리는데 걸린 스텝": None if opened_at is None else opened_at + 1}
print(json.dumps(res, indent=1, ensure_ascii=False))
pathlib.Path(__file__).with_name("action_space_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
