"""실험 6 집계 — 표 1 (모델 × adapt t0 / held-out t1–9 + 95% Wilson CI), 표 2 (마스크별 격차 회복률, held-out), α 선택 분리.

  /usr/bin/python3 ~/hayeon/dart_xarm7/summarize_exp6.py --cond v0 [--alpha-split]
결과는 results/exp6_summary.json 의 조건 키에 쓴다. 기존 summarize_eval.py 의 results/summary.json 은 건드리지 않는다.
회복률 = (mask − θ0)/(θ* − θ0), 분자·분모 모두 같은 조건·같은 잡의 <cond>_theta0 / <cond>_dart_a0.5 를 쓴다.
회복률 95% 구간은 모델별 held-out 에피소드를 독립 재표집한 부트스트랩 (2000회, 백분위).
"""
import argparse
import collections
import glob
import json
import math
import pathlib

import numpy as np

R = pathlib.Path.home() / "hayeon/dart_xarm7/results"
MASKS = [("vision", "vision"), ("vlm", "VLM"), ("expert", "expert"), ("visionvlm", "vision+VLM")]
# 실험 2 (시점 변화 medium, libero_spatial 500 에피소드) 격차 회복률 % — ~/hayeon/dart_exp2/RESULTS.md
EXP2 = {"vision": 46, "vlm": 51, "expert": 3, "visionvlm": 79}


def wilson(s, n, z=1.96):  # summarize_eval.py 와 같은 식
    if n == 0:
        return (0.0, 0.0)
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def load(tag):
    return [json.loads(l) for f in sorted(glob.glob(str(R / tag / "client*.jsonl"))) for l in open(f)]


def fisher_p(a, b):
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        return float("nan")
    return float(fisher_exact([[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]])[1])


def heldout(rows, trials=None):
    return [r["success"] for r in rows if r["task_id"] != 0 and (trials is None or r["episode"] in trials)]


def table1(tags):
    print("| 모델 | 태그 | t0 t1 … t9 (성공/10) | adapt t0 | held-out t1–9 (95% CI) | 전체 | errors |")
    print("|---|---|---|---|---|---|---|")
    out = {}
    for name, tag in tags:
        rows = load(tag)
        if not rows:
            continue
        a = [r["success"] for r in rows if r["task_id"] == 0]
        h = heldout(rows)
        lo, hi = wilson(sum(h), len(h))
        e = sum(1 for r in rows if r.get("error"))
        out[tag] = {"adapt": [sum(a), len(a)], "heldout": [sum(h), len(h)], "heldout_ci95": [lo, hi], "errors": e,
                    "per_task": {t: [sum(r["success"] for r in rows if r["task_id"] == t), sum(1 for r in rows if r["task_id"] == t)]
                                 for t in range(10)}}
        pt = " ".join(str(out[tag]["per_task"][t][0]) for t in range(10))
        print(f"| {name} | `{tag}` | {pt} | {sum(a)}/{len(a)} ({100 * sum(a) / max(1, len(a)):.0f}%) | "
              f"{sum(h)}/{len(h)} = **{100 * sum(h) / max(1, len(h)):.0f}%** ({100 * lo:.0f}–{100 * hi:.0f}) | "
              f"{100 * sum(r['success'] for r in rows) / len(rows):.0f}% | {e} |")
    return out


def table2(cond, n_boot=2000, seed=0):
    b, f = np.array(heldout(load(f"{cond}_theta0")), float), np.array(heldout(load(f"{cond}_dart_a0.5")), float)
    if not len(b) or not len(f):
        print("(θ0 / θ* 결과 없음 — 표 2 생략)")
        return {}
    rng = np.random.default_rng(seed)
    bs = lambda x: rng.choice(x, (n_boot, len(x))).mean(1)
    Bb, Bf = bs(b), bs(f)
    gap = f.mean() - b.mean()
    print(f"분모 θ* − θ0 (held-out, `{cond}_dart_a0.5` − `{cond}_theta0`) = {100 * f.mean():.1f} − {100 * b.mean():.1f} = "
          f"**{100 * gap:+.1f}%p** (Fisher p={fisher_p(f.tolist(), b.tolist()):.3g})")
    tg = heldout(load(f"{cond}_theta_tgt"))
    if tg:
        print(f"θ* vs θ_tgt (held-out): {int(f.sum())}/{len(f)} vs {sum(tg)}/{len(tg)}, Fisher p={fisher_p(f.tolist(), tg):.3g}")
    print()
    print(f"| 마스크 (δ 를 넣은 모듈) | held-out | − θ0 | 회복률 {cond} (부트스트랩 95%) | p vs θ0 | p vs θ* | 실험 2 회복률 (시점 medium) |")
    print("|---|---|---|---|---|---|---|")
    out = {"gap": gap, "theta0": b.mean(), "theta_star": f.mean()}
    for key, name in MASKS:
        m = np.array(heldout(load(f"{cond}_mask_{key}")), float)
        if not len(m):
            continue
        rec = (m.mean() - b.mean()) / gap if gap else float("nan")
        Bm = bs(m)
        with np.errstate(divide="ignore", invalid="ignore"):
            boot = (Bm - Bb) / (Bf - Bb)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        out[key] = {"heldout": m.mean(), "recovery": rec, "recovery_boot95": [float(lo), float(hi)],
                    "p_vs_theta0": fisher_p(m.tolist(), b.tolist()), "p_vs_theta_star": fisher_p(m.tolist(), f.tolist())}
        print(f"| {name} | {int(m.sum())}/{len(m)} = {100 * m.mean():.0f}% | {100 * (m.mean() - b.mean()):+.1f}%p | "
              f"**{100 * rec:.0f}%** ({100 * lo:.0f}–{100 * hi:.0f}) | {out[key]['p_vs_theta0']:.3g} | {out[key]['p_vs_theta_star']:.3g} | "
              f"{EXP2[key]}% |")
    return out


def alpha_split():
    """trial(=초기 상태 인덱스) 0~4 의 held-out 으로 α 를 고르고 5~9 로 보고 (10/10 실행 results/dart_a*)."""
    sel, rep = range(0, 5), range(5, 10)
    print("| 모델 | 선택용 held-out (trial 0–4) | 보고용 held-out (trial 5–9) |")
    print("|---|---|---|")
    out, best = {}, None
    for tag in ("theta0", "dart_a0.5", "dart_a0.8", "dart_a1.0"):
        rows = load(tag)
        if not rows or "episode" not in rows[0]:
            print(f"(`{tag}` 에 episode 인덱스 없음 — 분리 불가)")
            return None
        a, b = heldout(rows, sel), heldout(rows, rep)
        out[tag] = {"select": [sum(a), len(a)], "report": [sum(b), len(b)]}
        print(f"| `{tag}` | {sum(a)}/{len(a)} = {100 * sum(a) / len(a):.0f}% | {sum(b)}/{len(b)} = {100 * sum(b) / len(b):.0f}% |")
        if tag.startswith("dart_a") and (best is None or sum(a) > out[best]["select"][0]):
            best = tag
    t0r, br = out["theta0"]["report"], out[best]["report"]
    print(f"\n선택: **{best}** (trial 0–4 최고). 보고(trial 5–9): {best} {br[0]}/{br[1]} vs θ0 {t0r[0]}/{t0r[1]}, "
          f"Fisher p={fisher_p([1] * br[0] + [0] * (br[1] - br[0]), [1] * t0r[0] + [0] * (t0r[1] - t0r[0])):.3g}")
    out["selected"] = best
    return out


p = argparse.ArgumentParser()
p.add_argument("--cond", default="v0")
p.add_argument("--alpha-split", action="store_true")
args = p.parse_args()
c = args.cond
rows = [(f"θ0 ({c})", f"{c}_theta0"), (f"θ_tgt ({c})", f"{c}_theta_tgt"), (f"θ* α=0.5 ({c})", f"{c}_dart_a0.5")]
rows += [(f"mask {n} ({c})", f"{c}_mask_{k}") for k, n in MASKS]
if c == "v0":
    rows += [("θ0 (10/10 실행, 참고)", "theta0"), ("θ_tgt (10/10 실행, 참고)", "theta_tgt"), ("θ* α=0.5 (10/10 실행, 참고)", "dart_a0.5")]
print(f"## 표 1 — {c}\n")
res = {"table1": table1(rows)}
print(f"\n## 표 2 — {c} 격차 회복률 (held-out t1–9)\n")
res["table2"] = table2(c)
if args.alpha_split:
    print("\n## α 선택 분리 (10/10 실행 결과, 새 실행 없음)\n")
    res["alpha_split"] = alpha_split()
js = R / "exp6_summary.json"
allres = json.loads(js.read_text()) if js.exists() else {}
allres[c] = res
js.write_text(json.dumps(allres, indent=1, ensure_ascii=False, default=float))
