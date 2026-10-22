"""실험 6 단계 1 — 조건별 xArm7 타깃 데모 검증표 + 조건 간 / v0 데이터셋과의 동일성 확인.

  source ~/hayeon/dart_xarm7/env.sh && cd $DART_REPO && .venv/bin/python ~/hayeon/dart_xarm7/check_exp6_demos.py [cond ...]
입력: $DART_ROOT/demos/<cond>/libero_spatial_t0_demo_0_XArm7_retarget_rotq{.npz,_summary.json} (make_retarget_demo.py --rot-clip quantile)
정규화는 openpi _normalize_quantile 과 같은 식: (a − q01)/(q99 − q01 + 1e-6)·2 − 1, pi05_libero norm stats.
회전 클립 프레임 = ±1 로 자른 회전 액션이 q01~q99 밖이라 --rot-clip quantile 이 값을 바꾼 프레임.
"""
import io
import json
import os
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

D = pathlib.Path(os.path.expandvars("$DART_ROOT/demos"))
LR = pathlib.Path.home() / "hayeon/dart_xarm7/data/lerobot"
TAG = "libero_spatial_t0_demo_0"
NS = json.load(open(os.path.expandvars(
    "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json")))["norm_stats"]["actions"]
Q01, Q99 = np.array(NS["q01"][:7]), np.array(NS["q99"][:7])
conds = sys.argv[1:] or ["v0", "v1", "v2", "v3"]


def npz(c):
    return np.load(D / c / f"{TAG}_XArm7_retarget_rotq.npz")


def metrics(c):
    z, s = npz(c), json.load(open(D / c / f"{TAG}_XArm7_retarget_rotq_summary.json"))
    a = z["actions"].astype(np.float64)
    norm = (a - Q01) / (Q99 - Q01 + 1e-6) * 2 - 1
    rot1 = np.clip(z["rot_raw"], -1, 1)
    rot_clip = np.any((rot1 < Q01[3:6]) | (rot1 > Q99[3:6]), axis=1)
    return {"env": str(z["env_meta"]), "frames": len(a), "panda_success": s["panda_success"], "success": s["xarm_success_at_end"],
            "first_success": s["xarm_first_success_step"], "eef_vs_panda_cm": s["eef_vs_panda_cm"],
            "track_pos_cm": s["track_err_pos_cm"], "track_rot_deg": s["track_err_rot_deg"],
            "norm_over1_xyz": [float((np.abs(norm[:, i]) > 1).mean()) for i in range(3)],
            "norm_max_xyz": [float(np.abs(norm[:, i]).max()) for i in range(3)],
            "rot_clipped_frac": float(rot_clip.mean()), "pm1_clipped_frac": float(z["clipped"].mean()),
            "norm_rot_over1": float((np.abs(norm[:, 3:6]) > 1).any(1).mean())}


def load_ds(name):
    t = pq.read_table(next((LR / name / "data/chunk-000").glob("*.parquet"))).to_pydict()
    dec = lambda col: np.stack([np.array(Image.open(io.BytesIO(x["bytes"]))) for x in t[col]])
    return {"image": dec("image"), "wrist_image": dec("wrist_image"), "state": np.array(t["state"], np.float32),
            "actions": np.array(t["actions"], np.float32)}


def same(x, y):
    return x.shape == y.shape and np.array_equal(x, y)


def img_diff(x, y):
    return float((np.abs(x.astype(int) - y.astype(int)).sum(-1) > 0).mean())


res = {c: metrics(c) for c in conds if (D / c / f"{TAG}_XArm7_retarget_rotq.npz").exists()}
print("| 조건 | env (npz 기록) | 성공 / 첫 성공 | Panda 대비 위치차 평균/최대 cm | 추종오차 위치 평균/최대 cm | 추종오차 회전 평균/최대 ° | "
      "정규화 |a|>1 Δx / Δy / Δz | 회전 클립 프레임 |")
print("|---|---|---|---|---|---|---|---|")
print("| v0 (DEMO_PAIR.md, 10-08) | white, panda | 성공 / 73 | 1.8 / 3.3 | 4.0 / 6.0 | 5.9 / 17.1 | 22% / – / 21% | – |")
for c, m in res.items():
    o = m["norm_over1_xyz"]
    print(f"| {c} | {m['env'].replace('XARM7_COLOR=', '').replace('XARM7_DYNAMICS=', '')} | {'성공' if m['success'] else '**실패**'} / {m['first_success']} | "
          f"{m['eef_vs_panda_cm']['mean']:.1f} / {m['eef_vs_panda_cm']['max']:.1f} | {m['track_pos_cm']['mean']:.1f} / {m['track_pos_cm']['max']:.1f} | "
          f"{m['track_rot_deg']['mean']:.1f} / {m['track_rot_deg']['max']:.1f} | {100 * o[0]:.0f}% / {100 * o[1]:.0f}% / {100 * o[2]:.0f}% | "
          f"{100 * m['rot_clipped_frac']:.0f}% |")
print()
for c, m in res.items():
    print(f"{c}: frames {m['frames']}, Panda 재생 성공 {m['panda_success']}, ±1 클립 프레임 {100 * m['pm1_clipped_frac']:.0f}%, "
          f"정규화 최대 |Δx,Δy,Δz| {np.round(m['norm_max_xyz'], 2).tolist()}, 정규화 회전 |a|>1 {100 * m['norm_rot_over1']:.0f}%")

checks = {}
# 색만 다른 쌍: 물리가 같으므로 액션·궤적은 비트 단위로 같고 이미지만 달라야 한다
for a, b in (("v0", "v1"), ("v2", "v3")):
    if a in res and b in res:
        za, zb = npz(a), npz(b)
        checks[f"{a}_vs_{b}"] = {k: same(za[k], zb[k]) for k in ("actions", "state", "eef_pos", "track_err_pos")} | {
            "image_px_changed": img_diff(za["image"], zb["image"]), "wrist_px_changed": img_diff(za["wrist_image"], zb["wrist_image"])}
# v0 재현 vs 10-08 에 만든 v0 데이터셋 (학습에 쓴 것)
if "v0" in res:
    z, ds = npz("v0"), load_ds(f"{TAG}_tgt_xarm7_retarget_rotclip")
    checks["v0_repro_vs_v0_dataset"] = {"actions": same(z["actions"].astype(np.float32), ds["actions"]),
                                        "state": same(z["state"].astype(np.float32), ds["state"]),
                                        "image": same(z["image"], ds["image"]), "wrist_image": same(z["wrist_image"], ds["wrist_image"])}
    p, dp = np.load(D / "v0" / f"{TAG}_Panda.npz"), load_ds(f"{TAG}_src_panda")
    checks["panda_repro_vs_src_dataset"] = {"actions": same(p["actions"].astype(np.float32), dp["actions"]),
                                            "state": same(p["state"].astype(np.float32), dp["state"]),
                                            "image": same(p["image"], dp["image"]), "wrist_image": same(p["wrist_image"], dp["wrist_image"])}
print()
for k, v in checks.items():
    print(k, json.dumps(v, ensure_ascii=False))
(D / "stage1_check.json").write_text(json.dumps({"metrics": res, "checks": checks}, indent=1, ensure_ascii=False))
