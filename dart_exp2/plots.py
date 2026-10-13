"""실험 2 그림: (1) 조건별 성공률 + 95% Wilson CI, (2) 조건 × 태스크 성공 수 heatmap."""
import collections, json, math, os, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

root = pathlib.Path(__file__).parent
VIEW = os.environ.get("VIEW", "medium")
CONDS = ["none", "vision", "vlm", "vision,vlm", "expert", "all"]
LABEL = {"none": "θ₀ (no δ)", "vision": "vision only", "vlm": "VLM only", "vision,vlm": "vision+VLM", "expert": "expert only", "all": "θ* (all δ)"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GRAY = "#2a78d6", "#a8a7a1"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

def load(c, view=None):
    rows = [json.loads(l) for f in sorted((root / f"results/{view or VIEW}_libero_spatial_{c}").glob("client*.jsonl")) for l in open(f)]
    by = collections.defaultdict(int)
    for r in rows: by[r["task_id"]] += r["success"]
    return sum(r["success"] for r in rows), len(rows), [by[t] for t in range(10)]

def wilson(s, n, z=1.96):
    p = s / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

CONDS = [c for c in CONDS if (root / f"results/{VIEW}_libero_spatial_{c}/summary.txt").exists()]
data = {c: load(c) for c in CONDS}
SUF = "" if VIEW == "medium" else f"_{VIEW}"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

# (1) 성공률 점 + 95% CI (축을 0에서 시작하지 않으므로 막대 대신 점)
fig, ax = plt.subplots(figsize=(max(6.4, 1.2 * len(CONDS) + 0.8), 3.4), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for c in ("none", "all"):
    ax.axhline(100 * data[c][0] / 500, color=GRAY, lw=1, ls=(0, (3, 3)), zorder=1)
for i, c in enumerate(CONDS):
    s, n, _ = data[c]; p = 100 * s / n; lo, hi = wilson(s, n)
    color = GRAY if c in ("none", "all") else BLUE
    ax.plot([i, i], [100 * lo, 100 * hi], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.plot(i, p, "o", ms=9, color=color, mec=SURFACE, mew=2, zorder=3)
    ax.text(i + 0.12, p, f"{p:.1f}%", ha="left", va="center", fontsize=9.5, color=INK, fontweight="bold", bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"), zorder=4)
ax.set_xticks(range(len(CONDS)), [LABEL[c] for c in CONDS]); ax.set_xlim(-0.5, len(CONDS) - 0.3)
lo_all = min(100 * wilson(*data[c][:2])[0] for c in CONDS); hi_all = max(100 * wilson(*data[c][:2])[1] for c in CONDS)
ax.set_ylim(5 * math.floor(lo_all / 5 - 0.4), 5 * math.ceil(hi_all / 5 + 0.2)); ax.set_ylabel("success rate (%)")
ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
ax.set_title(f"LIBERO-view {VIEW} · libero_spatial · 500 episodes each (95% CI)", fontsize=10, color=INK2, loc="left")
fig.tight_layout(); fig.savefig(root / f"figs/fig1_success_rate{SUF}.png", facecolor=SURFACE); plt.close(fig)

# (2) 태스크별 heatmap
order = CONDS
M = [data[c][2] for c in order]
cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ)
fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(CONDS) + 0.8), dpi=200, facecolor=SURFACE)
ax.imshow(M, cmap=cmap, vmin=0, vmax=50, aspect="auto")
for i, row in enumerate(M):
    for j, v in enumerate(row):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=9, color="#ffffff" if v >= 25 else INK)
ax.set_xticks(range(10), [f"T{t}" for t in range(10)]); ax.set_yticks(range(len(order)), [LABEL[c] for c in order])
ax.set_xticks([x - 0.5 for x in range(1, 10)], minor=True); ax.set_yticks([y - 0.5 for y in range(1, len(order))], minor=True)
ax.grid(which="minor", color=SURFACE, lw=2); ax.tick_params(which="both", length=0)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("successes out of 50 per task", fontsize=10, color=INK2, loc="left")
fig.tight_layout(); fig.savefig(root / f"figs/fig2_per_task{SUF}.png", facecolor=SURFACE); plt.close(fig)

# (3) medium vs large: 격차 회복 비율 (vision만 / VLM만)
if VIEW == "medium" and (root / "results/large_libero_spatial_vlm/summary.txt").exists():
    rec = {}
    for v in ("medium", "large"):
        S = {c: load(c, v)[0] for c in ("none", "all", "vision", "vlm")}
        rec[v] = {c: 100 * (S[c] - S["none"]) / (S["all"] - S["none"]) for c in ("vision", "vlm")}
    fig, ax = plt.subplots(figsize=(5.6, 3.3), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    SER = {"vision": ("vision only", "#2a78d6"), "vlm": ("VLM only", "#eb6834")}
    w = 0.34
    for k, (c, (lab, col)) in enumerate(SER.items()):
        xs = [i + (k - 0.5) * (w + 0.03) for i in range(2)]
        ys = [rec[v][c] for v in ("medium", "large")]
        ax.bar(xs, ys, width=w, color=col, label=lab, zorder=2)
        for x, y in zip(xs, ys):
            ax.text(x, y + 2, f"{y:.0f}%", ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
    ax.axhline(100, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.text(0.5, 101.5, "θ* = 100%", fontsize=8.5, color=INK2, ha="center", va="bottom")
    ax.set_xticks([0, 1], ["medium view\n(θ₀ 71.4 → θ* 85.0)", "large view\n(θ₀ 12.6 → θ* 66.0)"])
    ax.set_ylim(0, 122); ax.set_ylabel("share of θ₀→θ* gap recovered (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("δ on one module only · libero_spatial · 500 episodes each", fontsize=10, color=INK2, loc="left")
    fig.tight_layout(); fig.savefig(root / "figs/fig3_recovery_medium_vs_large.png", facecolor=SURFACE); plt.close(fig)
print("saved", VIEW, CONDS)
