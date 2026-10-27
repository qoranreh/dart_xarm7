"""실험 6 단계 4 — v0~v3 결과 집계·가설 대조·그림 (새 실행 없음, 기존 결과만 읽음).

  /usr/bin/python3 ~/hayeon/dart_xarm7/exp6_results.py
출력: results/exp6_master.json, results/exp6_master_tables.md, figs/exp6_fig1_mask_minus_theta0.png, figs/exp6_fig2_theta0_tgt_star.png
- 성공률은 held-out (t1–9, 90 에피소드) 기준, 95% CI 는 Wilson (summarize_exp6.py 와 같은 식).
- 차이의 95% CI 는 Newcombe (Wilson 구간 결합, 독립 표본으로 취급 — 단계 0~3 의 Fisher 와 같은 가정).
- 회복률·부트스트랩 구간은 단계 0·3 에서 summarize_exp6.py 가 만든 results/exp6_summary.json 값을 그대로 쓴다.
- 조건별 판정은 사전 규칙: |차이| < 15%p 이면 "차이 없음/모름".
- 합산 검정 (CMH, 4 조건 층화) 은 결과를 본 뒤 추가한 사후 분석 (단계 3 과 같은 식: 연속성 보정, MH 공통 OR).
"""
import glob
import json
import math
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2, fisher_exact

ROOT = pathlib.Path.home() / "hayeon/dart_xarm7"
R = ROOT / "results"
FIGS = ROOT / "figs"
CONDS = ["v0", "v1", "v2", "v3"]
COND_LABEL = {"v0": "v0 white · panda", "v1": "v1 dark · panda", "v2": "v2 white · robosuite", "v3": "v3 dark · robosuite"}
COND_TICK = {"v0": "v0\nwhite · panda dyn.", "v1": "v1\ndark · panda dyn.", "v2": "v2\nwhite · robosuite dyn.", "v3": "v3\ndark · robosuite dyn."}
MODELS = ["theta0", "theta_tgt", "dart_a0.5", "mask_vision", "mask_vlm", "mask_expert", "mask_visionvlm"]
MASKS = [("mask_vision", "vision", "vision only"), ("mask_vlm", "vlm", "VLM only"),
         ("mask_expert", "expert", "expert only"), ("mask_visionvlm", "visionvlm", "vision+VLM")]
RULE = 15.0  # %p, 조건당 90회 held-out 의 사전 검출 한계


def tag(c, m):
    # v0 θ_tgt 는 같은 잡 실행이 없음 → 10-10 실행 (잡 129459, 같은 가중치) results/theta_tgt
    return "theta_tgt" if (c, m) == ("v0", "theta_tgt") else f"{c}_{m}"


def heldout(t):
    rows = [json.loads(l) for f in sorted(glob.glob(str(R / t / "client*.jsonl"))) for l in open(f)]
    h = [r["success"] for r in rows if r["task_id"] != 0]
    assert len(h) == 90 and not any(r.get("error") for r in rows), t
    return sum(h), len(h)


def wilson(s, n, z=1.96):
    p = s / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def newcombe(a, b):
    """(a 성공률 − b 성공률) 의 95% CI, %p. a, b = (성공, n)."""
    p1, p2 = a[0] / a[1], b[0] / b[1]
    l1, u1 = wilson(*a); l2, u2 = wilson(*b)
    d = p1 - p2
    return 100 * d, 100 * (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), 100 * (d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def fisher(a, b):
    return float(fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])[1])


def cmh(ma, mb):
    num = var = orn = ord_ = 0.0
    for c in CONDS:
        (A, n1), (Cc, n2) = H[c][ma], H[c][mb]
        B, D = n1 - A, n2 - Cc; n = n1 + n2
        num += A - (A + B) * (A + Cc) / n
        var += (A + B) * (Cc + D) * (A + Cc) * (B + D) / (n * n * (n - 1))
        orn += A * D / n; ord_ += B * Cc / n
    p = float(chi2.sf((abs(num) - 0.5) ** 2 / var, 1))
    sa, sb = sum(H[c][ma][0] for c in CONDS), sum(H[c][mb][0] for c in CONDS)
    signs = " ".join("+" if H[c][ma][0] > H[c][mb][0] else "−" if H[c][ma][0] < H[c][mb][0] else "=" for c in CONDS)
    return {"diff_pp": 100 * (sa - sb) / 360, "mh_or": orn / ord_, "p": p, "signs": signs}


def step0_loss(c):
    log = ROOT / ("logs/oneshot_ft/tgt_mem0.9_bs64.log" if c == "v0" else f"logs/oneshot_ft_exp6/{c}_tgt_mem0.9_bs64.log")
    return float(re.search(r"Step 0: grad_norm=[\d.]+, loss=([\d.]+)", log.read_text()).group(1))


def judge(diff, pred):
    """사전 규칙으로 한 예측 판정. pred: down / up / none (거의 그대로) / similar."""
    if abs(diff) < RULE:
        return "?", "차이 없음 (모순 없음)" if pred in ("none", "similar") else "모름 (15%p 미만)"
    if pred == "down":
        return ("○", "하락") if diff < 0 else ("✕", "반대로 상승")
    if pred == "up":
        return ("○", "상승") if diff > 0 else ("✕", "반대로 하락")
    return "✕", "15%p 이상 차이"


def pct(s, n):
    lo, hi = wilson(s, n)
    return f"{100 * s / n:.0f} ({100 * lo:.0f}–{100 * hi:.0f})"


# ---------------------------------------------------------------- 데이터
H = {c: {m: heldout(tag(c, m)) for m in MODELS} for c in CONDS}
S6 = json.loads((R / "exp6_summary.json").read_text())
STAGE1 = json.loads((R / "exp6_demos/stage1_check.json").read_text())["metrics"]
LOSS = {c: step0_loss(c) for c in CONDS}
out = {"heldout": {c: {m: list(H[c][m]) for m in MODELS} for c in CONDS}, "step0_loss": LOSS, "rule_pp": RULE,
       "note": "v0 theta_tgt = results/theta_tgt (10-10 run, job 129459); others same job 130582"}
md = []

# 마스터 표
md.append("| 조건 | θ0 | θ_tgt | θ* α=0.5 | δ→vision | δ→VLM | δ→expert | δ→vision+VLM | θ*−θ0 (%p) | 회복률 % (vision / VLM / expert / vision+VLM) | 데모 클립 ±1 / 회전 | FT step-0 loss |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
out["master"] = {}
for c in CONDS:
    t2 = S6[c]["table2"]
    rec = [100 * t2[k]["recovery"] for _, k, _ in MASKS]
    gap = newcombe(H[c]["dart_a0.5"], H[c]["theta0"])
    m1 = STAGE1[c]
    row = {"gap_pp": gap, "recovery_pct": rec, "recovery_boot95": [t2[k]["recovery_boot95"] for _, k, _ in MASKS],
           "clip_pm1": m1["pm1_clipped_frac"], "clip_rot": m1["rot_clipped_frac"], "step0_loss": LOSS[c]}
    out["master"][c] = row
    cells = [pct(*H[c][m]) for m in MODELS]
    if c == "v0":
        cells[1] += "†"
    md.append(f"| {COND_LABEL[c]} | " + " | ".join(cells) + f" | {gap[0]:+.1f} | " + " / ".join(f"{x:.0f}" for x in rec)
              + f" | {100 * m1['pm1_clipped_frac']:.0f}% / {100 * m1['rot_clipped_frac']:.0f}% | {LOSS[c]:.4f} |")
pool = {m: (sum(H[c][m][0] for c in CONDS), 360) for m in MODELS}
md.append("| **합산 (360회)** | " + " | ".join(pct(*pool[m]) for m in MODELS)
          + f" | {newcombe(pool['dart_a0.5'], pool['theta0'])[0]:+.1f} | — | — | — |")
md.append("")
md.append("† v0 θ_tgt 는 같은 잡 실행이 없어 10-10 실행 (잡 129459, 같은 가중치) 값.")

# 모듈 × 조건 (사전 규칙)
md.append("\n### 마스크 − θ0 (%p, 95% CI, Fisher p) — 조건별 판정 (|차이| ≥ 15%p 만 차이로 씀)\n")
md.append("| δ 를 넣은 모듈 | " + " | ".join(COND_LABEL[c] for c in CONDS) + " |")
md.append("|---|---|---|---|---|")
out["mask_minus_theta0"] = {}
for m, k, lab in MASKS + [("dart_a0.5", "all", "θ* (전부)")]:
    cells = []
    for c in CONDS:
        d, lo, hi = newcombe(H[c][m], H[c]["theta0"])
        p = fisher(H[c][m], H[c]["theta0"])
        out["mask_minus_theta0"].setdefault(k, {})[c] = {"diff": d, "ci": [lo, hi], "p": p}
        mark = "**↑**" if d >= RULE else "**↓**" if d <= -RULE else "–"
        cells.append(f"{d:+.1f} ({lo:+.0f}, {hi:+.0f}) p={p:.2g} {mark}")
    md.append(f"| {lab} | " + " | ".join(cells) + " |")
md.append("\n↑/↓ = 15%p 이상 차이, – = 차이 없음/모름.")

# 가설 대조
d_t0 = {c: newcombe(H[c]["theta0"], H["v0"]["theta0"]) for c in CONDS}
eff = {(c, m): newcombe(H[c][m], H[c]["theta0"])[0] for c in CONDS for m in MODELS}
claims = {
    ("V1", "H1"): [("θ0 하락 (v1 θ0 − v0 θ0)", d_t0["v1"][0], "down"),
                   ("회복은 vision+VLM (v1 vision+VLM − θ0; 격차 θ*−θ0 = %+.1f)" % eff[("v1", "dart_a0.5")], eff[("v1", "mask_visionvlm")], "up")],
    ("V1", "H2"): [("θ0 거의 유지 (v1 θ0 − v0 θ0)", d_t0["v1"][0], "none")],
    ("V2", "H1"): [("θ0 거의 유지 (v2 θ0 − v0 θ0)", d_t0["v2"][0], "none"),
                   ("expert 효과 없음 (v2 expert − θ0)", eff[("v2", "mask_expert")], "none")],
    ("V2", "H2"): [("θ0 하락 (v2 θ0 − v0 θ0)", d_t0["v2"][0], "down"),
                   ("expert 회복 ↑ (v2 expert − θ0)", eff[("v2", "mask_expert")], "up")],
    ("V3", "H1"): [("V1 과 비슷: θ0 하락 v3 − v1", d_t0["v3"][0] - d_t0["v1"][0], "similar"),
                   ("V1 과 비슷: vision+VLM 효과 v3 − v1", eff[("v3", "mask_visionvlm")] - eff[("v1", "mask_visionvlm")], "similar")],
    ("V3", "H2"): [("V1+V2 합 근처: θ0 하락 v3 − (v1 + v2)", d_t0["v3"][0] - d_t0["v1"][0] - d_t0["v2"][0], "similar"),
                   ("(H2 함의) expert 효과 v3", eff[("v3", "mask_expert")], "up")],
}
md.append("\n### 가설 대조 (조건별, 사전 규칙)\n")
md.append("| 셀 | 가설 | 예측 항목 (관찰값 %p) | 판정 |")
md.append("|---|---|---|---|")
out["hypothesis"] = {}
for (v, hname), items in claims.items():
    marks = []
    for desc, val, pred in items:
        s, txt = judge(val, pred)
        marks.append(s)
        md.append(f"| {v} | {hname} | {desc}: **{val:+.1f}** | {s} {txt} |")
        out["hypothesis"].setdefault(f"{v}_{hname}", []).append({"claim": desc, "value_pp": val, "pred": pred, "mark": s})
md.append("\n○ = 예측 방향으로 15%p 이상, ✕ = 반대 방향(또는 '그대로' 예측인데 15%p 이상), ? = 15%p 미만 → 차이 없음/모름.")

# 교란 점검 (c) 가산성
add = {"theta0_drop": [d_t0[c][0] for c in CONDS], "loss_rise": [LOSS[c] - LOSS["v0"] for c in CONDS]}
out["additivity"] = add

# 합산 (사후)
pairs = [("dart_a0.5", "theta0"), ("mask_vision", "theta0"), ("mask_visionvlm", "theta0"), ("mask_vlm", "theta0"),
         ("mask_expert", "theta0"), ("mask_vision", "dart_a0.5"), ("mask_expert", "dart_a0.5"), ("mask_vlm", "dart_a0.5")]
NAME = {"dart_a0.5": "θ*", "theta0": "θ0", "mask_vision": "vision", "mask_visionvlm": "vision+VLM", "mask_vlm": "VLM", "mask_expert": "expert"}
md.append("\n### 합산 검정 — 사후 분석 (결과를 본 뒤 추가, 4 조건 층화 CMH, 각 360회)\n")
md.append("| 비교 | 차이 (합산 %p) | MH 공통 OR | CMH p | 조건별 부호 (v0 v1 v2 v3) | Bonferroni (0.05/8) |")
md.append("|---|---|---|---|---|---|")
out["cmh_posthoc"] = {}
for a, b in pairs:
    r = cmh(a, b)
    out["cmh_posthoc"][f"{a}_vs_{b}"] = r
    md.append(f"| {NAME[a]} vs {NAME[b]} | {r['diff_pp']:+.1f} | {r['mh_or']:.2f} | {r['p']:.2g} | {r['signs']} | {'통과' if r['p'] < 0.05 / 8 else '—'} |")
vp = (H["v0"]["mask_vlm"][0] + H["v1"]["mask_vlm"][0], 180); tp = (H["v0"]["theta0"][0] + H["v1"]["theta0"][0], 180)
vr = (H["v2"]["mask_vlm"][0] + H["v3"]["mask_vlm"][0], 180); tr = (H["v2"]["theta0"][0] + H["v3"]["theta0"][0], 180)
out["vlm_by_dynamics_posthoc"] = {"panda": [newcombe(vp, tp)[0], fisher(vp, tp)], "robosuite": [newcombe(vr, tr)[0], fisher(vr, tr)]}

(R / "exp6_master.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
(R / "exp6_master_tables.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
print("\n가산성 θ0 하락 (v0..v3):", [f"{x:+.1f}" for x in add["theta0_drop"]], " step-0 loss 상승:", [f"{x:+.4f}" for x in add["loss_rise"]])
print("VLM 동역학별 (사후):", out["vlm_by_dynamics_posthoc"])

# ---------------------------------------------------------------- 그림 (dart_exp2/plots.py 스타일)
SURFACE, INK, INK2, MUTED, GRID, AXIS, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
SER1 = {"mask_vision": "#2a78d6", "mask_vlm": "#eb6834", "mask_expert": "#1baf7a", "mask_visionvlm": "#eda100"}  # 검증: 인접 CVD ΔE ≥ 9.1
SER2 = {"theta0": "#e87ba4", "theta_tgt": "#008300", "dart_a0.5": "#4a3aa7"}  # 검증: 인접 CVD ΔE ≥ 17.6
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
FIGS.mkdir(exist_ok=True)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", length=0)


# 그림 1: 마스크 − θ0 (%p) + 95% CI, 점선 = θ* − θ0
fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=300, facecolor=SURFACE)
style(ax)
ax.axhspan(-RULE, RULE, color=BAND, zorder=0.5, lw=0)
ax.axhline(0, color=AXIS, lw=1, zorder=1)
w, gapw = 0.17, 0.02
for i, c in enumerate(CONDS):
    for j, (m, k, lab) in enumerate(MASKS):
        x = i + (j - 1.5) * (w + gapw)
        d, lo, hi = newcombe(H[c][m], H[c]["theta0"])
        ax.bar(x, d, width=w, color=SER1[m], zorder=2, label=lab if i == 0 else None)
        ax.plot([x, x], [lo, hi], color=INK2, lw=1, zorder=3, solid_capstyle="butt")
        if abs(d) >= RULE:
            ax.text(x, hi + 1.2, f"{d:+.0f}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
    g = newcombe(H[c]["dart_a0.5"], H[c]["theta0"])[0]
    ax.plot([i - 0.44, i + 0.44], [g, g], color=INK, lw=1.3, ls=(0, (3, 2)), zorder=4, label="θ* − θ₀ (all δ)" if i == 0 else None)
ax.set_xticks(range(len(CONDS)), [COND_TICK[c] for c in CONDS])
ax.set_xlim(-0.55, len(CONDS) - 0.45); ax.set_ylim(-35, 42)
ax.set_ylabel("held-out success − θ₀ (pp)")
from matplotlib.patches import Patch
hs, ls = ax.get_legend_handles_labels(); order = [ls.index(l) for l in [m[2] for m in MASKS] + ["θ* − θ₀ (all δ)"]]
hs = [hs[i] for i in order] + [Patch(color=BAND)]; ls = [ls[i] for i in order] + ["|Δ| < 15 pp → no difference / unknown (pre-set rule)"]
ax.legend(hs, ls, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.17), ncol=3, fontsize=8.3, handlelength=1.4, columnspacing=1.2)
ax.set_title("δ in one module only · difference from θ₀ · 90 held-out episodes each · 95% CI (Newcombe)",
             fontsize=9.2, color=INK2, loc="left", pad=40)
fig.tight_layout(); fig.savefig(FIGS / "exp6_fig1_mask_minus_theta0.png", facecolor=SURFACE); plt.close(fig)

# 그림 2: θ0 / θ_tgt / θ* held-out 성공률 + Wilson 95% CI
fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=300, facecolor=SURFACE)
style(ax)
ax.axhline(0, color=AXIS, lw=1, zorder=1)
w = 0.22
LAB2 = {"theta0": "θ₀", "theta_tgt": "θ_tgt (one-shot FT)", "dart_a0.5": "θ* (DART α=0.5)"}
for i, c in enumerate(CONDS):
    for j, m in enumerate(SER2):
        x = i + (j - 1) * (w + 0.03)
        s, n = H[c][m]; p = 100 * s / n; lo, hi = wilson(s, n)
        ax.bar(x, p, width=w, color=SER2[m], zorder=2, label=LAB2[m] if i == 0 else None)
        if s == 0:  # 0% 막대는 높이가 없어 기준선 위에 색 표시만 (값 = 0)
            ax.plot([x - w / 2, x + w / 2], [0, 0], color=SER2[m], lw=3, solid_capstyle="butt", zorder=2.5)
        ax.plot([x, x], [100 * lo, 100 * hi], color=INK2, lw=1, zorder=3)
        ax.text(x, 100 * hi + 1.5, f"{p:.0f}" + ("†" if (c, m) == ("v0", "theta_tgt") else ""), ha="center", va="bottom", fontsize=8.3, color=INK)
ax.set_xticks(range(len(CONDS)), [COND_TICK[c] for c in CONDS])
ax.set_xlim(-0.55, len(CONDS) - 0.45); ax.set_ylim(0, 100)
ax.set_ylabel("held-out success (%)")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.12), ncol=3, fontsize=8.5)
ax.set_title("θ₀ / θ_tgt / θ* per condition · 90 held-out episodes each · 95% Wilson CI", fontsize=9.2, color=INK2, loc="left", pad=24)
fig.text(0.01, 0.01, "† v0 θ_tgt from the 10-10 run (job 129459, same weights); all others job 130582.", fontsize=7.3, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(FIGS / "exp6_fig2_theta0_tgt_star.png", facecolor=SURFACE); plt.close(fig)
print("saved figs:", sorted(p.name for p in FIGS.glob("exp6_*.png")))
