"""실험 2 결과 집계: 태스크별 성공 수, 95% Wilson CI, θ₀/θ* 대비 차이 검정, 회복 비율."""
import collections, json, math, os, pathlib, sys

CONDS = ["none", "all", "vision", "vlm", "vision,vlm", "expert"]
NAMES = {"none": "θ₀ (none)", "all": "θ* (all)", "vision": "vision만", "vlm": "VLM만", "expert": "expert만", "vision,vlm": "vision+VLM"}
root = pathlib.Path(__file__).parent / "results"
VIEW = os.environ.get("VIEW", "medium")
CONDS = [c for c in CONDS if (root / f"{VIEW}_libero_spatial_{c}" / "summary.txt").exists()]

def load(c):
    rows = [json.loads(l) for f in sorted((root / f"{VIEW}_libero_spatial_{c}").glob("client*.jsonl")) for l in open(f)]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task_id"]].append(r)
    return rows, by

def wilson(s, n, z=1.96):
    p = s / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def two_prop_p(s1, n1, s2, n2):
    p = (s1 + s2) / (n1 + n2); se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (s1 / n1 - s2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))

data = {c: load(c) for c in CONDS}
S = {c: sum(r["success"] for r in data[c][0]) for c in CONDS}
N = {c: len(data[c][0]) for c in CONDS}
assert all(n == 500 for n in N.values()), N
gap = S["all"] - S["none"]

print("| 조건 | " + " | ".join(f"T{t}" for t in range(10)) + " | 성공률 | 95% CI | θ₀ 대비 | θ* 대비 | 격차 회복 |")
print("|---" * 16 + "|")
for c in CONDS:
    rows, by = data[c]
    lo, hi = wilson(S[c], N[c])
    d0 = 100 * (S[c] - S["none"]) / 500; p0 = two_prop_p(S[c], 500, S["none"], 500)
    d1 = 100 * (S[c] - S["all"]) / 500; p1 = two_prop_p(S[c], 500, S["all"], 500)
    rec = (S[c] - S["none"]) / gap
    print(f"| {NAMES[c]} | " + " | ".join(str(sum(r['success'] for r in by[t])) for t in range(10))
          + f" | {100*S[c]/N[c]:.1f}% | {100*lo:.1f}–{100*hi:.1f} | {d0:+.1f}%p (p={p0:.3f}) | {d1:+.1f}%p (p={p1:.3f}) | {100*rec:.0f}% |")

print("\n성공 에피소드 평균 스텝:", {c: round(sum(r['steps'] for r in data[c][0] if r['success']) / S[c], 1) for c in CONDS})
print("태스크 설명:")
for t, rs in sorted(data["none"][1].items()):
    print(f"  T{t}: {rs[0]['task']}")
