"""데모 쌍 검증 리포트 자료: 통계(json), 프레임 샘플 그리드(png), 나란히 비교 영상(mp4/webm).

target 은 MimicGen 방식 + 회전 보정 제한 xArm7 데모(make_retarget_demo.py --rot-clip quantile, 채택).
비교용으로 액션 재생판(make_demo_pair.py)과 MimicGen 원래 방식(제한 없음)의 요약도 같이 계산한다.

  source ~/hayeon/dart_xarm7/env.sh && source ~/hayeon/dart_xarm7/client_env.sh
  $LIBERO_PY ~/hayeon/dart_xarm7/report_assets.py
"""
import json
import os
import pathlib

import imageio
import numpy as np
from PIL import Image, ImageDraw

D = pathlib.Path(os.path.expandvars("$DART_ROOT/demos"))
OUT = pathlib.Path(os.path.expanduser("~/hayeon/dart_xarm7/report"))
OUT.mkdir(parents=True, exist_ok=True)
TAG = "libero_spatial_t0_demo_0"
P = np.load(D / f"{TAG}_Panda.npz")
VARIANTS = {"replay": "XArm7", "retarget": "XArm7_retarget", "retarget_rotq": "XArm7_retarget_rotq"}
V = {k: np.load(D / f"{TAG}_{v}.npz") for k, v in VARIANTS.items() if (D / f"{TAG}_{v}.npz").exists()}
ADOPTED = "retarget_rotq"  # 2026-10-08 사용자 선택: MimicGen 방식 + 회전 보정을 norm q01~q99 로 제한
X = V[ADOPTED]
S_pair = json.loads((D / f"{TAG}_summary.json").read_text())
S_ret = {k: json.loads((D / f"{TAG}_{VARIANTS[k]}_summary.json").read_text()) for k in ("retarget", "retarget_rotq") if k in V}
NS = json.load(open(os.path.expandvars(
    "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json")))["norm_stats"]


def qnorm(x, k):  # openpi 분위수 정규화 (transforms.py)
    q01, q99 = np.array(NS[k]["q01"])[: x.shape[1]], np.array(NS[k]["q99"])[: x.shape[1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2 - 1


def stats(x, k):
    n = qnorm(x, k)
    return {"mean": x.mean(0).round(4).tolist(), "std": x.std(0).round(4).tolist(), "min": x.min(0).round(4).tolist(),
            "max": x.max(0).round(4).tolist(), "norm_min": n.min(0).round(3).tolist(), "norm_max": n.max(0).round(3).tolist(),
            # 1% 허용 (그리퍼 ±1 이 q99=0.9996 경계에 걸리는 것 제외)
            "out_of_q01_q99_pct": (np.abs(n) > 1.01).mean(0).round(3).tolist()}


def first_success(k):
    if k == "replay":
        return S_pair["XArm7"]["first_success_step"], S_pair["XArm7"]["success_at_end"]
    return S_ret[k]["xarm_first_success_step"], S_ret[k]["xarm_success_at_end"]


variants = {}
for k, z in V.items():
    n = qnorm(z["actions"].astype(float), "actions")
    eef = np.linalg.norm(z["eef_pos"] - P["eef_pos"], axis=1) * 100
    fs, se = first_success(k)
    variants[k] = {
        "first_success": fs, "success_end": se,
        "eef_diff_cm_mean": round(float(eef.mean()), 2), "eef_diff_cm_max": round(float(eef.max()), 2),
        "action_absdiff_mean": np.abs(z["actions"] - P["actions"]).mean(0).round(3).tolist(),
        "norm_min": n.min(0).round(2).tolist(), "norm_max": n.max(0).round(2).tolist(),
        "out_pct": (np.abs(n) > 1.01).mean(0).round(3).tolist(),
        "track_rot_deg_mean": round(float(np.degrees(z["track_err_rot"]).mean()), 1) if "track_err_rot" in z else None,
        "track_rot_deg_max": round(float(np.degrees(z["track_err_rot"]).max()), 1) if "track_err_rot" in z else None,
        "clipped_steps": int(z["clipped"].sum()) if "clipped" in z else 0,
    }

rep = {
    "pair_summary": S_pair, "adopted": ADOPTED, "adopted_summary": S_ret.get(ADOPTED), "variants": variants,
    "frames": {"Panda": int(len(P["actions"])), "XArm7": int(len(X["actions"]))},
    "panda_first_success": S_pair["Panda"]["first_success_step"],
    "actions": {"Panda": stats(P["actions"], "actions"), "XArm7": stats(X["actions"], "actions")},
    "state": {"Panda": stats(P["state"], "state"), "XArm7": stats(X["state"], "state")},
    "norm_stats_q01": {k: np.round(NS[k]["q01"][:8], 4).tolist() for k in ("state", "actions")},
    "norm_stats_q99": {k: np.round(NS[k]["q99"][:8], 4).tolist() for k in ("state", "actions")},
    "pad_gap_cm": {"Panda": (np.array(P["pad_gap"]) * 100).round(2).tolist(), "XArm7": (np.array(X["pad_gap"]) * 100).round(2).tolist()},
    "eef_diff_cm": (np.linalg.norm(P["eef_pos"] - X["eef_pos"], axis=1) * 100).round(2).tolist(),
    "grip_action": P["actions"][:, 6].round(2).tolist(),
    "pitch_action": {"Panda": P["actions"][:, 4].round(3).tolist(), "XArm7": X["actions"][:, 4].round(3).tolist()},
}
(OUT / "stats.json").write_text(json.dumps(rep, ensure_ascii=False))

# 프레임 샘플 그리드: 행 = Panda agentview / xArm7 agentview / Panda wrist / xArm7 wrist, 열 = 프레임
T = len(P["actions"])
picks = [0, 25, 50, 75, T - 1]
rows = [("Panda agentview", P["image"]), ("xArm7 agentview", X["image"]), ("Panda wrist", P["wrist_image"]), ("xArm7 wrist", X["wrist_image"])]
s, lw, hh = 192, 120, 22
G = Image.new("RGB", (lw + len(picks) * (s + 4), hh + len(rows) * (s + 4)), "white")
dr = ImageDraw.Draw(G)
for j, f in enumerate(picks):
    dr.text((lw + j * (s + 4) + 4, 5), f"frame {f}", fill="black")
for i, (label, arr) in enumerate(rows):
    dr.text((6, hh + i * (s + 4) + s // 2 - 6), label, fill="black")
    for j, f in enumerate(picks):
        G.paste(Image.fromarray(arr[f]).resize((s, s), Image.Resampling.BILINEAR), (lw + j * (s + 4), hh + i * (s + 4)))
G.save(OUT / "frames_grid.png")

# 영상: 2×2 (위 agentview, 아래 wrist / 왼쪽 Panda, 오른쪽 xArm7), 10fps = 데이터셋 fps
succ = {"Panda": S_pair["Panda"]["first_success_step"], "XArm7": variants[ADOPTED]["first_success"]}
frames = []
for t in range(T):
    F = Image.new("RGB", (512, 560), "white")
    d = ImageDraw.Draw(F)
    for c, (r, z) in enumerate((("Panda", P), ("XArm7", X))):
        F.paste(Image.fromarray(z["image"][t]), (c * 256, 24))
        F.paste(Image.fromarray(z["wrist_image"][t]), (c * 256, 24 + 256 + 24))
        ok = succ[r] is not None and t >= succ[r]
        d.text((c * 256 + 6, 6), f"{r}  agentview" + ("   SUCCESS" if ok else ""), fill=(0, 130, 0) if ok else "black")
        d.text((c * 256 + 6, 24 + 256 + 6), f"{r}  wrist", fill="black")
    d.text((420, 24 + 256 + 6), f"t={t:3d}/{T}", fill="black")
    frames.append(np.asarray(F))
imageio.mimwrite(OUT / "demo_pair.mp4", frames, fps=10, codec="libx264", quality=8, macro_block_size=8)
try:
    imageio.mimwrite(OUT / "demo_pair.webm", frames, fps=10, codec="libvpx-vp9", macro_block_size=8,
                     output_params=["-b:v", "0", "-crf", "32"])
except Exception as e:  # noqa: BLE001
    print("webm 실패:", e)
for f in sorted(OUT.iterdir()):
    print(f"{f.stat().st_size / 1e6:6.2f} MB  {f}")
