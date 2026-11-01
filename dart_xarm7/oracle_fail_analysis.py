"""실험 4-후속1 6번: retarget 1개를 같은 시작 장면에서 저장된 액션으로 다시 재생하며 그릇 상태를 기록하고, 영상·프레임 띠·시계열 그림을 만든다.

- Panda 원본 데모(같은 폴더의 _Panda.npz 액션)도 같은 방식으로 재생해 기준으로 쓴다.
- 기록 (매 스텝 뒤): 그릇 위치·기울기(그릇 z축과 월드 z 사이 각), 양쪽 패드가 그릇에 닿음(_check_grasp), 그리퍼-그릇 접촉,
  그릇-접시 xy 거리 (성공 술어 On = 접시와 접촉 + xy < 3cm), grip_site-그릇 거리·높이차,
  그리퍼·팔이 닿은 다른 물체 이름 (예: t4 의 wooden_cabinet_1). 스텝별 배열은 <name>_replay.npz.
- 재생이 원래 실행과 같은지: 매 스텝 전 eef_pos 를 npz 와 비교 (eef_replay_max_diff_m).
- 액션이 끝난 뒤 --extra 스텝 더 기다려(그리퍼 열림) 늦게 성공하는지도 본다 (표의 성공 수에는 반영 안 함).
출력 (--out, 기본 results/oracle_t45/fail/): <name>.json (사건·튀는 스텝·유형 제안), <name>_strip.png (위 Panda / 아래 xArm7 같은 스텝),
<name>_ts.png (시계열), <name>_replay.npz, 영상은 results/oracle_t45/videos/<name>.mp4 ([agentview | 손목], 10fps, git 제외).

  $LIBERO_PY ~/hayeon/dart_xarm7/oracle_fail_analysis.py --task 4 --demo 0 --variant quantile
"""
import argparse
import json
import os
import pathlib

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from robosuite.utils import transform_utils as T

import libero_xarm7 as X
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

p = argparse.ArgumentParser()
p.add_argument("--suite", default="libero_spatial")
p.add_argument("--task", type=int, required=True)
p.add_argument("--demo", type=int, required=True)
p.add_argument("--variant", choices=["quantile", "none"], required=True)
p.add_argument("--obj", default="akita_black_bowl_1")
p.add_argument("--plate", default="plate_1")
p.add_argument("--wait", type=int, default=10)
p.add_argument("--extra", type=int, default=20)
p.add_argument("--out", default=os.path.expanduser("~/hayeon/dart_xarm7/results/oracle_t45/fail"))
args = p.parse_args()

DUMMY = [0.0] * 6 + [-1.0]
tag = f"{args.suite}_t{args.task}_demo_{args.demo}"
folder = pathlib.Path(os.path.expandvars(f"$DART_ROOT/demos/oracle_t{args.task}/demo{args.demo}"))
suffix = "XArm7_retarget_rotq" if args.variant == "quantile" else "XArm7_retarget"
name = f"t{args.task}_demo{args.demo}_{args.variant}"
OUT = pathlib.Path(args.out)
VID = OUT.parent / "videos"
OUT.mkdir(parents=True, exist_ok=True)
VID.mkdir(parents=True, exist_ok=True)
Z = np.load(folder / f"{tag}_{suffix}.npz")
PZ = np.load(folder / f"{tag}_Panda.npz")
state0 = np.load(os.path.expandvars(f"$DART_ROOT/demos/{args.suite.removeprefix('libero_')}_t{args.task}_hdf5_demo0-4.npz"))[f"demo_{args.demo}/states"][0]
task = benchmark.get_benchmark_dict()[args.suite]().get_task(args.task)
bddl = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)


def make(robot):
    env = OffScreenRenderEnv(bddl_file_name=bddl, robots=[robot], camera_heights=256, camera_widths=256,
                             camera_names=["agentview", "robot0_eye_in_hand"])
    env.seed(0)
    env.reset()
    return env


def owner_map(e):
    """geom 이름 → 물체/고정물 이름 (그리퍼·팔 접촉 상대를 이름으로 보기 위함)."""
    m = {}
    for d in (e.fixtures_dict, e.objects_dict):
        for n, o in d.items():
            for g in o.contact_geoms:
                m[g] = n
    return m


def contacts(e, model, owners, skip):
    return sorted({owners.get(g, "other") for g in e.get_contacts(model)} - {skip})


def probe(env):
    e = env.env
    owners = owner_map(e)
    rb, sim = e.robots[0], e.sim
    bowl, plate = e.get_object(args.obj), e.get_object(args.plate)
    bp = sim.data.body_xpos[e.obj_body_id[args.obj]].copy()
    pp = sim.data.body_xpos[e.obj_body_id[args.plate]].copy()
    R = T.quat2mat(T.convert_quat(sim.data.body_xquat[e.obj_body_id[args.obj]], to="xyzw"))
    grip = sim.data.site_xpos[rb.eef_site_id].copy()
    return {"bowl_pos": bp, "bowl_tilt_deg": float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))),
            "grasp": bool(e._check_grasp(gripper=rb.gripper, object_geoms=bowl)), "touch": bool(e.check_contact(rb.gripper, bowl)),
            "bowl_plate_xy": float(np.linalg.norm(bp[:2] - pp[:2])), "bowl_plate_contact": bool(e.check_contact(bowl, plate)),
            "grip_bowl_dist": float(np.linalg.norm(grip - bp)), "grip_minus_bowl_z": float(grip[2] - bp[2]),
            "grip_other": contacts(e, rb.gripper, owners, args.obj), "arm_other": contacts(e, rb.robot_model, owners, None),
            "success": bool(e._check_success())}


def run(robot, actions, eef_ref):
    env = make(robot)
    if robot == "Panda":
        obs = env.set_init_state(state0)
    else:
        pa = make("Panda")
        flat, _ = X.transfer_init_state(pa, state0, env)
        pa.close()
        obs = env.set_init_state(flat)
    for _ in range(args.wait):
        obs, _, _, _ = env.step(DUMMY)
    rec, eef_diff = [], 0.0
    p0 = probe(env)
    for t, a in enumerate(actions):
        eef_diff = max(eef_diff, float(np.abs(obs["robot0_eef_pos"] - eef_ref[t]).max()))
        obs, _, _, _ = env.step(a)
        rec.append(probe(env))
    extra_success = None
    for k in range(args.extra):
        obs, _, _, _ = env.step(DUMMY)
        if env.env._check_success():
            extra_success = k + 1
            break
    after = probe(env)  # 추가 대기 뒤 (그리퍼 열림) 상태
    env.close()
    out = {k: np.array([r[k] for r in rec]) for k in rec[0] if k not in ("grip_other", "arm_other")}
    out["grip_other"] = [r["grip_other"] for r in rec]
    out["arm_other"] = [r["arm_other"] for r in rec]
    after = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in after.items()}
    return p0, out, eef_diff, extra_success, after


def events(p0, r, actions):
    close = np.flatnonzero(actions[:, 6] > 0)
    g = np.flatnonzero(r["grasp"])
    dz = r["bowl_pos"][:, 2] - p0["bowl_pos"][2]
    e = {"close_cmd_step": int(close[0]) if len(close) else None,
         "first_grasp_step": int(g[0]) if len(g) else None, "last_grasp_step": int(g[-1]) if len(g) else None, "grasp_steps": int(len(g)),
         "first_touch_step": int(np.flatnonzero(r["touch"])[0]) if r["touch"].any() else None,
         "grip_bowl_dist_at_close_cm": float(r["grip_bowl_dist"][close[0]] * 100) if len(close) else None,
         "grip_minus_bowl_z_at_close_cm": float(r["grip_minus_bowl_z"][close[0]] * 100) if len(close) else None,
         "grip_contacts_close_window": sorted({n for t in range(max(close[0] - 10, 0), min(close[0] + 15, len(r["grip_other"])))
                                               for n in r["grip_other"][t]}) if len(close) else [],
         "arm_contacts_all": sorted({n for c in r["arm_other"] for n in c}),
         "final_touch": bool(r["touch"][-1]),
         "bowl_lift_max_cm": float(dz.max() * 100), "bowl_lift_argmax": int(dz.argmax()),
         "bowl_tilt_max_while_grasp_deg": float(r["bowl_tilt_deg"][g].max()) if len(g) else None,
         "bowl_tilt_max_deg": float(r["bowl_tilt_deg"].max()),
         "final_bowl_plate_xy_cm": float(r["bowl_plate_xy"][-1] * 100), "final_bowl_plate_contact": bool(r["bowl_plate_contact"][-1]),
         "final_bowl_tilt_deg": float(r["bowl_tilt_deg"][-1]), "final_bowl_dz_cm": float(dz[-1] * 100),
         "min_bowl_plate_xy_cm": float(r["bowl_plate_xy"].min() * 100), "success_end": bool(r["success"][-1]),
         "first_success_step": int(np.flatnonzero(r["success"])[0]) + 1 if r["success"].any() else None}
    if len(g):  # 들고 있는 동안 그릇이 접시 쪽으로 얼마나 갔나 (놓기 직전 xy)
        e["bowl_plate_xy_at_last_grasp_cm"] = float(r["bowl_plate_xy"][g[-1]] * 100)
    return e


def suggest(e, ep):
    """유형 제안 (최종 판정은 프레임 띠를 보고 로그에 적음). 순서: 접근 → 파지 → 들기 → 놓기 중 처음 어긋난 곳.
    파지 성립 = 양 패드 접촉 ≥ 10스텝 이고 그릇이 3cm 이상 들림. 기운 채 = 접촉 중 그릇 기울기 > max(Panda + 10°, 30°)."""
    if e["first_grasp_step"] is None:
        return "접근" if (e["grip_bowl_dist_at_close_cm"] or 99) > 8 else "파지(빈손으로 닫힘)"
    tilted = (e["bowl_tilt_max_while_grasp_deg"] or 0) > max((ep["bowl_tilt_max_while_grasp_deg"] or 0) + 10, 30)
    if tilted:
        return "파지(기운 채 잡힘)"
    if e["grasp_steps"] < 10 or e["bowl_lift_max_cm"] < 3:
        return "파지(빈손으로 닫힘)"
    if e.get("bowl_plate_xy_at_last_grasp_cm", 99) > 8:
        return "들기"
    return "놓기"


pz0, pr, p_eef, p_extra, p_after = run("Panda", PZ["actions"].astype(np.float64), PZ["eef_pos"])
xz0, xr, x_eef, x_extra, x_after = run("XArm7", Z["actions"].astype(np.float64), Z["eef_pos"])
src_actions = PZ["actions"]
ev = {"Panda": events(pz0, pr, src_actions), "XArm7": events(xz0, xr, src_actions)}

# 튀는 스텝: pad_gap 은 그리퍼 명령이 바뀌지 않았는데 한 스텝에 2mm 넘게 변한 곳, track_err 는 평균+2σ 초과 (스텝 번호 = 액션 인덱스)
gap = Z["pad_gap"]
cmd = src_actions[:, 6]
dgap = np.diff(gap)
# 명령이 바뀐 뒤 15스텝(닫힘·열림 진행 중)은 제외
changes = np.flatnonzero(np.diff(cmd)) + 1
settling = np.zeros(len(gap) - 1, bool)
for c in changes:
    settling[max(c - 1, 0):c + 15] = True
gap_jumps = [(int(t + 1), round(float(dgap[t]) * 1000, 1)) for t in np.flatnonzero((np.abs(dgap) > 0.002) & ~settling)]
tp, tr = Z["track_err_pos"], Z["track_err_rot"]
spk = lambda x: [int(t) for t in np.flatnonzero(x > x.mean() + 2 * x.std())]
spikes = {"pad_gap_jumps_step_mm": gap_jumps, "track_err_pos_spike_steps": spk(tp), "track_err_rot_spike_steps": spk(tr),
          "track_err_pos_top3": [(int(t), round(float(tp[t]) * 100, 1)) for t in np.argsort(tp)[-3:][::-1]],
          "track_err_rot_top3": [(int(t), round(float(np.degrees(tr[t])), 1)) for t in np.argsort(tr)[-3:][::-1]],
          "pad_gap_hold_median_mm": {"XArm7": float(np.median(gap[cmd > 0]) * 1000) if (cmd > 0).any() else None,
                                     "Panda": float(np.median(PZ["pad_gap"][cmd > 0]) * 1000) if (cmd > 0).any() else None},
          "pad_gap_min_mm": {"XArm7": float(gap.min() * 1000), "Panda": float(PZ["pad_gap"].min() * 1000)}}
res = {"name": name, "task": task.language, "variant": args.variant, "env_meta": str(Z["env_meta"]),
       "eef_replay_max_diff_m": {"Panda": p_eef, "XArm7": x_eef},
       "replay_success_matches_summary": None, "success_after_extra_wait": {"Panda": p_extra, "XArm7": x_extra},
       "after_extra_wait": {"Panda": p_after, "XArm7": x_after},
       "events": ev, "spikes": spikes, "suggested_failure_point": suggest(ev["XArm7"], ev["Panda"])}
S = json.loads((folder / f"{tag}_{suffix}_summary.json").read_text())
res["replay_success_matches_summary"] = ev["XArm7"]["success_end"] == S["xarm_success_at_end"]
(OUT / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
np.savez_compressed(OUT / f"{name}_replay.npz", **{f"{rb}_{k}": v for rb, rr in (("Panda", pr), ("XArm7", xr))
                                                    for k, v in rr.items() if k not in ("grip_other", "arm_other")})

# 영상: [agentview | 손목] + 글자
frames = []
for t in range(len(Z["actions"])):
    im = Image.fromarray(np.concatenate([Z["image"][t], Z["wrist_image"][t]], 1))
    d = ImageDraw.Draw(im)
    d.text((4, 2), f"{name} t={t} cmd={'close' if cmd[t] > 0 else 'open'}", fill=(255, 255, 0))
    d.text((4, 14), f"gap {gap[t] * 1000:.1f}mm  err {tp[t] * 100:.1f}cm {np.degrees(tr[t]):.1f}deg  grasp={int(xr['grasp'][t])}", fill=(255, 255, 0))
    frames.append(np.asarray(im))
if not (VID / f"{name}.mp4").exists():  # 재생이 결정적이라 내용이 같음 — 재생 중인 파일을 덮지 않음
    imageio.mimsave(VID / f"{name}.mp4", frames, fps=10)

# 프레임 띠: 위 Panda, 아래 xArm7 (같은 스텝). 고르게 6칸 + 사건 스텝
ex = ev["XArm7"]
picks = set(np.linspace(0, len(cmd) - 1, 6).round().astype(int).tolist())
for k in ("close_cmd_step", "first_grasp_step", "last_grasp_step", "bowl_lift_argmax"):
    if ex.get(k) is not None:
        picks.add(int(ex[k]))
picks = sorted(picks)[:12]
rows = []
for img, lab in ((PZ["image"], "Panda"), (Z["image"], "xArm7")):
    tiles = []
    for t in picks:
        im = Image.fromarray(np.ascontiguousarray(img[t]))
        ImageDraw.Draw(im).text((4, 2), f"{lab} t={t}", fill=(255, 255, 0))
        tiles.append(np.asarray(im))
    rows.append(np.concatenate(tiles, 1))
Image.fromarray(np.concatenate(rows, 0)).save(OUT / f"{name}_strip.png")

# 시계열
fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
t = np.arange(len(cmd))
ax[0].plot(t, gap * 1000, label="xArm7 pad_gap")
ax[0].plot(t, PZ["pad_gap"] * 1000, "--", label="Panda pad_gap")
ax[0].fill_between(t, 0, 90, where=cmd > 0, alpha=0.1, color="k", label="close cmd")
ax[0].set_ylabel("mm")
ax[1].plot(t, tp * 100, label="track err pos (cm)")
ax[1].plot(t, np.degrees(tr) / 5, label="track err rot (deg/5)")
ax[2].plot(t, (xr["bowl_pos"][:, 2] - xz0["bowl_pos"][2]) * 100, label="xArm7 bowl dz (cm)")
ax[2].plot(t, (pr["bowl_pos"][:, 2] - pz0["bowl_pos"][2]) * 100, "--", label="Panda bowl dz (cm)")
ax[2].plot(t, xr["grasp"] * 2, ":", label="xArm7 grasp (x2)")
ax[2].plot(t, xr["bowl_tilt_deg"] / 10, label="xArm7 bowl tilt (deg/10)")
ax[3].plot(t, xr["bowl_plate_xy"] * 100, label="xArm7 bowl-plate xy (cm)")
ax[3].plot(t, pr["bowl_plate_xy"] * 100, "--", label="Panda bowl-plate xy (cm)")
ax[3].axhline(3, color="r", lw=0.8, label="On threshold 3cm")
ax[3].set_xlabel("step")
for a in ax:
    a.legend(fontsize=7, loc="upper right")
    a.grid(alpha=0.3)
fig.suptitle(f"{name}  suggested: {res['suggested_failure_point']}")
fig.tight_layout()
fig.savefig(OUT / f"{name}_ts.png", dpi=80)
print(json.dumps({k: res[k] for k in ("name", "eef_replay_max_diff_m", "replay_success_matches_summary", "success_after_extra_wait", "suggested_failure_point")}, ensure_ascii=False))
