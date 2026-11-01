"""실험 4-후속1 표: task × {Panda 재생, retarget rot-clip quantile, retarget 제한 없음} 의 성공 수/5, 첫 성공 스텝, 추종오차, ±1 클립 스텝.

입력: $DART_ROOT/demos/oracle_t<N>/demo<k>/ 의 summary.json (+ 회전 q01~q99 클립 스텝 수는 retarget npz 의 rot_raw 와 actions 로 계산).
출력: ~/hayeon/dart_xarm7/results/oracle_t45/summary.{json,md}. 성공 = 액션을 다 실행한 끝에서 _check_success() (summary 의 success_at_end).
retarget 이 하나도 없으면 STOP1 (Panda 재생) 표만 만든다.

  $LIBERO_PY ~/hayeon/dart_xarm7/oracle_t45_summary.py
"""
import json
import os
import pathlib

import numpy as np

D = pathlib.Path(os.path.expandvars("$DART_ROOT/demos"))
OUT = pathlib.Path(os.path.expanduser("~/hayeon/dart_xarm7/results/oracle_t45"))
OUT.mkdir(parents=True, exist_ok=True)
TASKS, DEMOS = (4, 5), range(5)
VARIANTS = {"quantile": "XArm7_retarget_rotq", "none": "XArm7_retarget"}
# DEMO_PAIR.md (2026-10-08, 잡 129138) 의 t0 demo_0 기준값
T0_REF = {"quantile": {"success": "성공", "first": 73, "pos": (4.0, 6.0), "rot": (5.9, 17.1), "clipped": 27},
          "none": {"success": "성공", "first": 74, "pos": None, "rot": None, "clipped": 27}}


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def retarget_row(folder, tag, suffix):
    s = load(folder / f"{tag}_{suffix}_summary.json")
    if s is None:
        return None
    z = np.load(folder / f"{tag}_{suffix}.npz")
    rot_q = int((np.abs(z["actions"][:, 3:6] - np.clip(z["rot_raw"], -1, 1)) > 1e-6).any(1).sum())
    return {"success": s["xarm_success_at_end"], "first": s["xarm_first_success_step"], "frames": s["frames"],
            "pos_mean": s["track_err_pos_cm"]["mean"], "pos_max": s["track_err_pos_cm"]["max"],
            "rot_mean": s["track_err_rot_deg"]["mean"], "rot_max": s["track_err_rot_deg"]["max"],
            "clipped": s["steps_with_clipped_action"], "rot_quantile_clipped": rot_q,
            "eef_vs_panda_mean": s["eef_vs_panda_cm"]["mean"], "eef_vs_panda_max": s["eef_vs_panda_cm"]["max"],
            "panda_success_inside": s["panda_success"], "env_meta": s["method"].rsplit("[", 1)[-1].rstrip("]"),
            "ik_pos_err": s["transfer"]["ik_pos_err"], "missing_joints": s["transfer"]["missing_joints"]}


rows = {}
for n in TASKS:
    for k in DEMOS:
        f, tag = D / f"oracle_t{n}" / f"demo{k}", f"libero_spatial_t{n}_demo_{k}"
        ps = load(f / f"{tag}_summary.json")
        r = {"panda": None if ps is None else {"success": ps["Panda"]["success_at_end"], "first": ps["Panda"]["first_success_step"],
                                                "frames": ps["Panda"]["frames"]}}
        for v, suf in VARIANTS.items():
            r[v] = retarget_row(f, tag, suf)
        rows[f"t{n}_demo{k}"] = r
t0 = {v: retarget_row(D / "oracle_t0_check" / "demo0", "libero_spatial_t0_demo_0", suf) for v, suf in VARIANTS.items()}


def agg(n, v):
    rs = [rows[f"t{n}_demo{k}"][v] for k in DEMOS]
    done = [r for r in rs if r is not None]
    if not done:
        return None
    a = {"n": len(done), "succ": sum(r["success"] for r in done), "first": [r["first"] for r in rs] if True else None}
    if v != "panda":
        for key in ("pos", "rot"):
            a[f"{key}_mean"] = float(np.mean([r[f"{key}_mean"] for r in done]))
            a[f"{key}_max"] = float(np.max([r[f"{key}_max"] for r in done]))
        a["clipped"] = [r["clipped"] for r in done]
        a["rot_quantile_clipped"] = [r["rot_quantile_clipped"] for r in done]
        a["frames"] = [r["frames"] for r in done]
    return a


summary = {"rows": rows, "t0_laptop": t0, "t0_ref_DEMO_PAIR": T0_REF,
           "agg": {f"t{n}": {v: agg(n, v) for v in ("panda", *VARIANTS)} for n in TASKS}}
gate = {f"t{n}": summary["agg"][f"t{n}"]["panda"] and summary["agg"][f"t{n}"]["panda"]["succ"] >= 4 for n in TASKS}
summary["stop1_gate"] = {"criterion": "task 마다 Panda 재생 성공 >= 4/5", **gate, "pass": all(gate.values())}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

fs = lambda xs: " ".join("–" if x is None else str(x) for x in xs)
md = ["# 실험 4-후속1 요약 (oracle_t45_summary.py 가 생성)", "",
      "성공 = 액션을 끝까지 실행한 뒤 성공 판정. 첫 성공 = demo_0~4 순서, – 는 성공 안 함. 추종오차 평균 = demo 별 평균의 평균, 최대 = 전체 최대.", "",
      "| task | 실행 | 성공/5 | 첫 성공 스텝 (demo0~4) | 액션 수 | 추종오차 위치 평균/최대 cm | 회전 평균/최대 ° | ±1 클립 스텝 | q01~q99 회전 클립 스텝 |",
      "|---|---|---|---|---|---|---|---|---|"]
for v in ("quantile", "none"):
    ref = T0_REF[v]
    pos = "–" if ref["pos"] is None else f"{ref['pos'][0]} / {ref['pos'][1]}"
    rot = "–" if ref["rot"] is None else f"{ref['rot'][0]} / {ref['rot'][1]}"
    md.append(f"| t0 기준 (DEMO_PAIR) | retarget {v} | {ref['success']} | {ref['first']} | 98 | {pos} | {rot} | {ref['clipped']} | – |")
    r = t0[v]
    if r:
        md.append(f"| t0 노트북 재현 | retarget {v} | {'성공' if r['success'] else '실패'} | {r['first']} | {r['frames']} | {r['pos_mean']:.1f} / {r['pos_max']:.1f} | "
                  f"{r['rot_mean']:.1f} / {r['rot_max']:.1f} | {r['clipped']} | {r['rot_quantile_clipped'] if v == 'quantile' else '–'} |")
for n in TASKS:
    A = summary["agg"][f"t{n}"]
    frames = [rows[f"t{n}_demo{k}"]["panda"]["frames"] if rows[f"t{n}_demo{k}"]["panda"] else None for k in DEMOS]
    if A["panda"]:
        md.append(f"| t{n} | Panda 재생 | {A['panda']['succ']}/{A['panda']['n']} | {fs(A['panda']['first'])} | {fs(frames)} | – | – | – | – |")
    for v in VARIANTS:
        a = A[v]
        if a:
            md.append(f"| t{n} | retarget {v} | {a['succ']}/{a['n']} | {fs(a['first'])} | {fs(a['frames'])} | {a['pos_mean']:.1f} / {a['pos_max']:.1f} | "
                      f"{a['rot_mean']:.1f} / {a['rot_max']:.1f} | {fs(a['clipped'])} | {fs(a['rot_quantile_clipped']) if v == 'quantile' else '–'} |")
md += ["", f"STOP1 기준 (task 마다 Panda 재생 성공 ≥ 4/5): " + ", ".join(f"{k} {'통과' if g else '미달'}" for k, g in gate.items()), ""]
md += ["## demo 별", "", "| demo | Panda | quantile 성공/첫 | quantile 추종 위치 평균/최대 cm, 회전 평균/최대 ° | none 성공/첫 | none 추종 위치, 회전 |", "|---|---|---|---|---|---|"]
for key, r in rows.items():
    cell = lambda x: "–" if x is None else f"{'성공' if x['success'] else '실패'} / {x['first']}"
    trk = lambda x: "–" if x is None else f"{x['pos_mean']:.1f}/{x['pos_max']:.1f}, {x['rot_mean']:.1f}/{x['rot_max']:.1f}"
    md.append(f"| {key} | {cell(r['panda'])} | {cell(r['quantile'])} | {trk(r['quantile'])} | {cell(r['none'])} | {trk(r['none'])} |")
(OUT / "summary.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
