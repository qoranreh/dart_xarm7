"""평가 결과(results/<tag>/client*.jsonl)를 모델 × 태스크 성공률 표로 정리한다.

  /usr/bin/python3 ~/hayeon/dart_xarm7/summarize_eval.py [tag ...]
  SUMMARY_JSON=<경로> 로 json 출력 위치를 바꿀 수 있다 (기본 results/summary.json — 실험 6 은 조건별 파일로 써서 기존 것을 덮어쓰지 않음).
"""
import os
import collections
import glob
import json
import math
import pathlib
import sys

R = pathlib.Path.home() / "hayeon/dart_xarm7/results"
ORDER = ["theta0_panda", "theta0", "theta_tgt", "dart_a0.5", "dart_a0.8", "dart_a1.0"]
NAMES = {"theta0_panda": "θ0 (Panda, 참고)", "theta0": "(a) θ0", "theta_tgt": "(b) θ_tgt one-shot FT",
         "dart_a0.5": "(c) θ* α=0.5", "dart_a0.8": "(c) θ* α=0.8", "dart_a1.0": "(c) θ* α=1.0"}


def wilson(s, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def load(tag):
    rows = [json.loads(l) for f in glob.glob(str(R / tag / "client*.jsonl")) for l in open(f)]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task_id"]].append(r)
    return rows, by


tags = sys.argv[1:] or [t for t in ORDER if (R / t).exists()]
out = {}
tasks = list(range(10))
hdr = "| 모델 | " + " | ".join(f"t{t}" for t in tasks) + " | **adapt t0** | **held-out t1–9** | 전체 |"
print(hdr)
print("|" + "---|" * (len(tasks) + 4))
for tag in tags:
    rows, by = load(tag)
    if not rows:
        continue
    cell = []
    for t in tasks:
        v = by.get(t, [])
        cell.append(f"{sum(r['success'] for r in v)}/{len(v)}" if v else "–")
    a = by.get(0, [])
    ho = [r for t in tasks[1:] for r in by.get(t, [])]
    sa, sh, s = sum(r["success"] for r in a), sum(r["success"] for r in ho), sum(r["success"] for r in rows)
    lo, hi = wilson(sh, len(ho))
    out[tag] = {"adapt": [sa, len(a)], "heldout": [sh, len(ho)], "all": [s, len(rows)], "heldout_ci95": [lo, hi],
                "per_task": {t: [sum(r["success"] for r in by.get(t, [])), len(by.get(t, []))] for t in tasks},
                "errors": sum(1 for r in rows if r.get("error")),
                "mean_steps_success": round(sum(r["steps"] for r in rows if r["success"]) / max(1, s), 1),
                "fail_timeouts": sum(1 for r in rows if not r["success"] and not r.get("error") and r["steps"] >= 219)}
    print(f"| {NAMES.get(tag, tag)} | " + " | ".join(cell) +
          f" | **{100 * sa / max(1, len(a)):.0f}%** | **{100 * sh / max(1, len(ho)):.0f}%** ({100 * lo:.0f}–{100 * hi:.0f}) | {100 * s / len(rows):.0f}% |")
print()
for tag, v in out.items():
    print(f"{tag}: errors={v['errors']} 성공 에피소드 평균 스텝={v['mean_steps_success']} 실패 중 시간초과={v['fail_timeouts']}")
pathlib.Path(os.environ.get("SUMMARY_JSON", R / "summary.json")).write_text(json.dumps(out, indent=1, ensure_ascii=False))
