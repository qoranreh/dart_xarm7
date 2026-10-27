"""실험 7 파일럿 집계 — θ0 의 섭동별 성공률, '없음' 대비 하락, 미리 정한 수준 선택 규칙 적용.

  /usr/bin/python3 ~/hayeon/dart_xarm7/exp7_pilot_summary.py
출력: results/exp7pilot/pilot_summary.{md,json}
선택 규칙 (노션 [7] §4, 파일럿 전에 기록): 축마다 θ0 하락 ≥ 15%p 이고 θ0 ≥ 15% 인 수준 중 하락이 25%p 에 가장 가까운 것.
  두 수준 모두 하락 < 15%p → 한 단계 키워 다시 (×2.0 / 45° / 0.3), 두 수준 모두 θ0 < 15% → 한 단계 줄여 다시 (×1.1 / 7.5° / 0.05).
성공률은 10 태스크 전체 (파일럿은 θ0 만이라 adapt/held-out 구분 없음). CI: Wilson, 차이는 Newcombe.
"""
import collections
import glob
import json
import math
import pathlib

R = pathlib.Path.home() / "hayeon/dart_xarm7/results/exp7pilot"
AXES = {"cam_dist": ("C1 카메라 거리", "×", (1.2, 1.5), "×2.0", "×1.1"), "cam_azim": ("C2 카메라 방위", "±", (15.0, 30.0), "45°", "7.5°"),
        "joint": ("R 초기 관절각", "", (0.1, 0.2, 0.3), "0.4 (규칙에 없음)", "0.05")}  # 0.3 = 규칙의 '한 단계 키움' (2026-10-27 재파일럿)


def wilson(s, n, z=1.96):
    p = s / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def newcombe(a, b):
    p1, p2 = a[0] / a[1], b[0] / b[1]
    (l1, u1), (l2, u2) = wilson(*a), wilson(*b)
    d = p1 - p2
    return 100 * d, 100 * (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), 100 * (d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def load(perturb, level):
    rows = [json.loads(l) for f in sorted(glob.glob(str(R / f"exp7pilot_theta0_{perturb}_{level:g}" / "client*.jsonl"))) for l in open(f)]
    if not rows:  # 러너 태그의 수준 표기 (예: 1.2, 15, 0.1) 와 맞춘다
        rows = [json.loads(l) for f in sorted(glob.glob(str(R / f"exp7pilot_theta0_{perturb}_*" / "client*.jsonl"))) for l in open(f)
                if abs(json.loads(l)["level"] - level) < 1e-9]
    return rows


base_rows = load("none", 0.0)
base = (sum(r["success"] for r in base_rows), len(base_rows))
out = {"none": {"succ": base, "errors": sum(1 for r in base_rows if r.get("error"))}}
md = ["| 섭동 | 수준 | θ0 성공 | 95% CI | '없음' 대비 (%p, 95% CI) | 태스크별 t0…t9 | 섭동 크기 (평균) | errors |", "|---|---|---|---|---|---|---|---|"]
pt = lambda rows: " ".join(str(sum(r["success"] for r in rows if r["task_id"] == t)) for t in range(10))
lo, hi = wilson(*base)
md.append(f"| 없음 | — | {base[0]}/{base[1]} = **{100 * base[0] / base[1]:.0f}%** | {100 * lo:.0f}–{100 * hi:.0f} | — | {pt(base_rows)} | — | {out['none']['errors']} |")
choice = {}
for ax, (name, pre, levels, up, down) in AXES.items():
    res = []
    for lv in levels:
        rows = load(ax, lv)
        if not rows:
            continue
        s = (sum(r["success"] for r in rows), len(rows))
        d, dl, dh = newcombe(s, base)
        lo, hi = wilson(*s)
        P = [r["pinfo"] for r in rows]
        if ax == "joint":
            size = f"손끝 {sum(p['eef_shift_cm'] for p in P) / len(P):.1f} cm, {sum(p['eef_rot_deg'] for p in P) / len(P):.1f}°, 재추첨 {sum(p['tries'] > 1 for p in P)}"
        else:
            size = f"중심까지 {sum(p['dist0'] for p in P) / len(P):.2f} → {sum(p['dist'] for p in P) / len(P):.2f} m"
        err = sum(1 for r in rows if r.get("error"))
        res.append({"level": lv, "succ": s, "rate": 100 * s[0] / s[1], "drop": -d, "diff_ci": [dl, dh], "errors": err})
        md.append(f"| {name} | {pre}{lv:g}{'°' if ax == 'cam_azim' else ''} | {s[0]}/{s[1]} = **{100 * s[0] / s[1]:.0f}%** | {100 * lo:.0f}–{100 * hi:.0f} | "
                  f"{d:+.0f} ({dl:+.0f}, {dh:+.0f}) | {pt(rows)} | {size} | {err} |")
    ok = [x for x in res if x["drop"] >= 15 and x["rate"] >= 15]
    if ok:
        dist = lambda x: round(abs(x["drop"] - 25), 6)  # 부동소수점 오차로 동점이 갈리지 않게
        m = min(dist(x) for x in ok)
        best = [x for x in ok if dist(x) == m]
        if len(best) > 1:
            choice[ax] = {"decision": "동점 — 규칙에 동점 처리 없음, 사용자 결정 필요",
                          "tied": [{"level": x["level"], "drop": x["drop"], "rate": x["rate"]} for x in best]}
        else:
            choice[ax] = {"decision": "선택", "level": best[0]["level"], "drop": best[0]["drop"], "rate": best[0]["rate"]}
    elif res and all(x["drop"] < 15 for x in res):
        choice[ax] = {"decision": f"키워서 다시 ({up})"}
    elif res and all(x["rate"] < 15 for x in res):
        choice[ax] = {"decision": f"줄여서 다시 ({down})"}
    elif res:
        choice[ax] = {"decision": "규칙 밖 (한 수준은 하락 부족, 다른 수준은 바닥) — 중간 수준 필요"}
    out[ax] = res
out["choice"] = choice
md.append("")
md.append("**수준 선택 (미리 정한 규칙)**")
for ax, c in choice.items():
    name, pre = AXES[ax][0], AXES[ax][1]
    if c["decision"] == "선택":
        md.append(f"- {name}: **{pre}{c['level']:g}{'°' if ax == 'cam_azim' else ''}** (θ0 {c['rate']:.0f}%, 하락 {c['drop']:.0f}%p)")
    elif "tied" in c:
        md.append(f"- {name}: **동점** " + " / ".join(f"{pre}{t['level']:g}{'°' if ax == 'cam_azim' else ''} (θ0 {t['rate']:.0f}%, 하락 {t['drop']:.0f}%p)" for t in c["tied"]) + " — 규칙에 동점 처리 없음, 사용자 결정 필요")
    else:
        md.append(f"- {name}: {c['decision']}")
(R / "pilot_summary.md").write_text("\n".join(md) + "\n")
(R / "pilot_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("\n".join(md))
