"""실험 3-2 분석: 바뀐 시점에서 각 모델의 action 이 기준(θ₀·original)에서 얼마나 벗어나나.

dev = 처음 5스텝(실행되는 구간)의 팔 6차원 action 차이 L2 평균. 차원마다 기준 action 의 표준편차로 나눠 스케일을 맞춤.
grip = 처음 5스텝 중 그리퍼 열기/닫기 부호가 기준과 다른 비율.
floor = θ₀·original 을 다른 노이즈로 → 샘플링 흔들림. 회복률 = (dev_θ₀ − dev_모델) / (dev_θ₀ − dev_floor).
95% CI: 에피소드(50개) 단위 부트스트랩 2000회.
"""
import json, pathlib
import numpy as np
R = pathlib.Path(__file__).parent / "results"
A = dict(np.load(R / "actions.npz"))
ref = A["base:original:0"]
sd = ref[:, :5, :6].reshape(-1, 6).std(0)
ep_key = A["task_id"] * 100 + A["episode"]
eps = np.unique(ep_key)
rng = np.random.default_rng(0)
boot = [rng.choice(eps, len(eps)) for _ in range(2000)]

def dev(run):
    d = (A[run][:, :5, :6] - ref[:, :5, :6]) / sd
    return np.linalg.norm(d, axis=-1).mean(1)
def grip(run):
    return (np.sign(A[run][:, :5, 6]) != np.sign(ref[:, :5, 6])).mean(1)
def boot_mean(x):
    per_ep = {e: x[ep_key == e] for e in eps}
    m = [np.concatenate([per_ep[e] for e in b]).mean() for b in boot]
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

D = {r: dev(r) for r in A if ":" in r}
G = {r: grip(r) for r in A if ":" in r}
floor = D["base:original:1"]
SR = {"medium": {"base": 71.4, "all": 85.0, "vision": 77.6, "vlm": 78.4},
      "large": {"base": 12.6, "all": 66.0, "vision": 18.2, "vlm": 69.4}}
lines = ["| 시점 | 모델 | action 차이 (95% CI) | 그리퍼 불일치 | action 회복률 | 실험 2 성공률 (회복) |", "|---|---|---|---|---|---|"]
m, lo, hi = boot_mean(floor)
lines.append(f"| original | θ₀ (노이즈만 다름, 바닥값) | {m:.2f} ({lo:.2f}–{hi:.2f}) | {G['base:original:1'].mean():.1%} | — | — |")
summary = {}
for view in ("medium", "large"):
    b = D[f"base:{view}:0"].mean()
    for name, run in (("θ₀", f"base:{view}:0"), ("θ*", f"{view}-all:{view}:0"), ("vision만", f"{view}-vision:{view}:0"), ("VLM만", f"{view}-vlm:{view}:0")):
        m, lo, hi = boot_mean(D[run])
        rec = (b - m) / (b - floor.mean())
        key = {"θ₀": "base", "θ*": "all", "vision만": "vision", "VLM만": "vlm"}[name]
        sr = SR[view][key]; srr = (sr - SR[view]["base"]) / (SR[view]["all"] - SR[view]["base"])
        summary[f"{view}/{key}"] = dict(dev=m, lo=lo, hi=hi, grip=G[run].mean(), rec=rec)
        lines.append(f"| {view} | {name} | {m:.2f} ({lo:.2f}–{hi:.2f}) | {G[run].mean():.1%} | {'—' if key=='base' else f'{rec:+.0%}'} | {sr:.1f}% ({'—' if key=='base' else f'{srr:+.0%}'}) |")
out = "\n".join(lines); print(out)

# 태스크별 (large): action 차이 vs 실험 2 태스크별 성공 수
succ_large = {"base": [0,11,23,24,0,1,0,1,3,0], "all": [50,47,50,48,16,33,15,12,43,16],
              "vision": [11,11,22,31,0,1,0,0,15,0], "vlm": [43,48,50,50,42,25,15,12,29,33]}
tl = ["", "| 태스크 (large) | θ₀ | θ* | vision만 | VLM만 |  ← action 차이 (괄호: 실험 2 성공 /50)", "|---|---|---|---|---|"]
xs, ys = [], []
for t in range(10):
    mk = A["task_id"] == t
    row = []
    for key, run in (("base", "base:large:0"), ("all", "large-all:large:0"), ("vision", "large-vision:large:0"), ("vlm", "large-vlm:large:0")):
        v = D[run][mk].mean(); row.append(f"{v:.2f} ({succ_large[key][t]})"); xs.append(v); ys.append(succ_large[key][t])
    tl.append(f"| T{t} | " + " | ".join(row) + " |")
r_s = np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0, 1]
tl.append(f"\n태스크×모델 40개 칸에서 action 차이와 성공 수의 Spearman 상관: {r_s:.2f}")
print("\n".join(tl))
(R / "actions_summary.md").write_text(out + "\n" + "\n".join(tl) + "\n")
json.dump(summary, open(R / "actions_summary.json", "w"), indent=1)
