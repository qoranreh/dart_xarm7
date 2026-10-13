"""실험 3 그림: 레이어별 CKA 곡선 (results/cka_layers.csv 에서)."""

import collections
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).parent
SURFACE, INK, INK2, GRID, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#a8a7a1"
SERIES = {  # 비교 → (라벨, 색, 선 모양). 색은 실험 2 그림과 같은 뜻 (vision=파랑, VLM=주황)
    "A:base": ("θ₀", GRAY, (0, (3, 2))),
    "A:vision": ("vision δ only", "#2a78d6", "-"),
    "A:vlm": ("VLM δ only", "#eb6834", "-"),
    "A:all": ("θ* (all δ)", "#1baf7a", "-"),
}
STREAMS = [("vision", "vision encoder\n(3rd-person image)"), ("llm_img", "VLM · image tokens"),
           ("llm_txt", "VLM · prompt tokens"), ("expert", "action expert")]

d = collections.defaultdict(list)
with open(root / "results/cka_layers.csv") as f:
    for r in csv.DictReader(f):
        d[(r["mode"], r["view"], r["comparison"], r["stream"])].append((int(r["layer"]), float(r["cka"])))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def fig_A(mode):
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4), dpi=200, facecolor=SURFACE, sharey=True)
    for r, view in enumerate(("medium", "large")):
        for c, (stream, title) in enumerate(STREAMS):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            for comp, (lab, col, ls) in SERIES.items():
                pts = sorted(d[(mode, view, comp, stream)])
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=col, ls=ls, lw=2 if comp != "A:base" else 1.6, label=lab, zorder=3 if comp == "A:all" else 2)
            ax.set_ylim(0, 1.02)
            ax.yaxis.grid(True, color=GRID, lw=0.7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=9.5, color=INK)
            if c == 0:
                ax.set_ylabel(f"{view} view\nCKA vs θ₀ · original view")
            if r == 1:
                ax.set_xlabel("layer (0 = input)" if stream != "vision" else "SigLIP block")
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle(f"How close is each model's representation of the shifted view to θ₀'s original-view representation?  ({mode} CKA)",
                 fontsize=10, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(root / f"figs/fig1_cka_to_source_{mode}.png", facecolor=SURFACE)
    plt.close(fig)


def fig_B(mode):
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), dpi=200, facecolor=SURFACE, sharey=True)
    styles = {("medium", "B:base_vs_all@target"): ("medium δ · shifted input", "#2a78d6", "-"),
              ("large", "B:base_vs_all@target"): ("large δ · shifted input", "#eb6834", "-"),
              ("medium", "C:base_vs_all@original"): ("medium δ · original input", "#2a78d6", (0, (3, 2))),
              ("large", "C:base_vs_all@original"): ("large δ · original input", "#eb6834", (0, (3, 2)))}
    for c, (stream, title) in enumerate(STREAMS):
        ax = axes[c]
        ax.set_facecolor(SURFACE)
        for (view, comp), (lab, col, ls) in styles.items():
            xs, ys = zip(*sorted(d[(mode, view, comp, stream)]))
            ax.plot(xs, ys, color=col, ls=ls, lw=1.8, label=lab)
        ax.set_ylim(0, 1.02)
        ax.yaxis.grid(True, color=GRID, lw=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_xlabel("layer (0 = input)" if stream != "vision" else "SigLIP block")
    axes[0].set_ylabel("CKA(θ₀, θ*) on the same input")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"Where does δ change the representation?  (1 = unchanged, {mode} CKA)", fontsize=10, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(root / f"figs/fig2_delta_change_{mode}.png", facecolor=SURFACE)
    plt.close(fig)


(root / "figs").mkdir(exist_ok=True)
for mode in ("global", "within"):
    fig_A(mode)
    fig_B(mode)
print("saved")
