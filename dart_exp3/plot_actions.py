"""실험 3-2 그림: 바뀐 시점에서 기준(θ₀·original) action 과의 차이 (점 + 95% CI), 아래에 실험 2 성공률."""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
root = pathlib.Path(__file__).parent
S = json.load(open(root / "results/actions_summary.json"))
SURFACE, INK, INK2, GRID, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#a8a7a1"
M = [("base", "θ₀", GRAY), ("vision", "vision δ only", "#2a78d6"), ("vlm", "VLM δ only", "#eb6834"), ("all", "θ* (all δ)", "#1baf7a")]
SR = {"medium": {"base": 71.4, "all": 85.0, "vision": 77.6, "vlm": 78.4}, "large": {"base": 12.6, "all": 66.0, "vision": 18.2, "vlm": 69.4}}
FLOOR = 0.31
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), dpi=200, facecolor=SURFACE, sharey=True)
for ax, view in zip(axes, ("medium", "large")):
    ax.set_facecolor(SURFACE)
    ax.axhline(FLOOR, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.text(3.45, FLOOR + 0.02, "sampling noise floor", fontsize=8, color=INK2, ha="right", va="bottom")
    for i, (k, lab, col) in enumerate(M):
        s = S[f"{view}/{k}"]
        ax.plot([i, i], [s["lo"], s["hi"]], color=col, lw=2.2, solid_capstyle="round")
        ax.plot(i, s["dev"], "o", ms=9, color=col, mec=SURFACE, mew=2)
        ax.text(i + 0.13, s["dev"], f"{s['dev']:.2f}", va="center", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none"))
    ax.set_xticks(range(4), [f"{lab}\nsuccess {SR[view][k]:.1f}%" for k, lab, _ in M], fontsize=8.5)
    ax.set_xlim(-0.4, 3.5); ax.set_ylim(0, 1.45)
    ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.set_title(f"{view} view", fontsize=10, color=INK, loc="left")
axes[0].set_ylabel("action deviation from θ₀ · original view\n(first 5 steps, arm dims, std-normalized)")
fig.suptitle("Held-out LIBERO-spatial states (546 snapshots, 50 episodes): lower = acts more like the source policy",
             fontsize=10, color=INK2, x=0.01, ha="left")
fig.tight_layout(); (root / "figs").mkdir(exist_ok=True)
fig.savefig(root / "figs/fig3_action_deviation.png", facecolor=SURFACE); print("saved")
