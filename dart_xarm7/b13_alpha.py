"""[13] Baseline v2 — 2단계 α 선택 (결과만 읽음, 새 실행 없음).  /usr/bin/python3 ~/hayeon/dart_xarm7/b13_alpha.py

보정 task t0 t2 t4 t6 × 초기 상태 0–19 (태그마다 80행). 규칙 (사전 등록): 격자 0.2 0.4 0.6 0.8 1.0 중 평균 성공률이 가장 높은 α = α*,
동률이면 작은 α. θ0 (b13_cal_theta0) 는 α=0 기준 행이고 후보가 아니다. 참고로 t0 열만 볼 때 최고인 α 도 적는다 (선택에는 안 씀).
태그마다 80행·errors 0·(task_id, episode) 중복 없음·집합이 정확히 t0 t2 t4 t6 × 0–19 인지 assert. <태그>_r2 가 있으면 그것을 쓴다.
출력: results/b13_table2.md, results/b13_alpha_choice.txt (첫 줄 alpha_star=<α*> — run_b13.sh 가 읽음)
"""
import collections
import datetime
import json
import pathlib

from scipy.stats import binomtest

R = pathlib.Path.home() / "hayeon/dart_xarm7/results"
ALPHAS = ["0.2", "0.4", "0.6", "0.8", "1.0"]
TASKS = [0, 2, 4, 6]
EPS = range(20)
N = len(TASKS) * len(EPS)


def load(tag):
    used = f"{tag}_r2" if (R / f"{tag}_r2" / "summary.txt").exists() else tag
    rows = [json.loads(l) for f in sorted((R / used).glob("client*.jsonl")) for l in open(f) if l.strip()]
    keys = collections.Counter((x["task_id"], x["episode"]) for x in rows)
    assert len(rows) == N, f"{used}: {len(rows)}행 (기대 {N})"
    assert not any(x.get("error") for x in rows), f"{used}: errors {sum(1 for x in rows if x.get('error'))}"
    assert max(keys.values()) == 1, f"{used}: (task_id, episode) 중복"
    assert set(keys) == {(t, e) for t in TASKS for e in EPS}, f"{used}: (task_id, episode) 집합이 t0 t2 t4 t6 × 0–19 가 아님"
    assert {x["robot"] for x in rows} == {"XArm7"}, f"{used}: robot {sorted({x['robot'] for x in rows})}"
    return used, {t: sum(x["success"] for x in rows if x["task_id"] == t) for t in TASKS}


def cp(k, n):
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return 100 * ci.low, 100 * ci.high


rows = [("θ0 (α=0, 기준)", "b13_cal_theta0")] + [(f"α={a}", f"b13_cal_a{a}") for a in ALPHAS]
res = {}
for name, tag in rows:
    used, by = load(tag)
    res[tag] = {"name": name, "used": used, "by_task": by, "total": sum(by.values())}

cand = [(a, res[f"b13_cal_a{a}"]) for a in ALPHAS]
best = max(cand, key=lambda c: (c[1]["total"], -float(c[0])))[0]          # 평균 최고, 동률이면 작은 α
t0_best = max(cand, key=lambda c: (c[1]["by_task"][0], -float(c[0])))[0]  # 참고: t0 열만
ties = [a for a, r in cand if r["total"] == res[f"b13_cal_a{best}"]["total"]]
t0_ties = [a for a, r in cand if r["by_task"][0] == res[f"b13_cal_a{t0_best}"]["by_task"][0]]

lines = ["| 행 | 태그 | " + " | ".join(f"t{t} /20" for t in TASKS) + f" | 합계 /{N} | 평균 % (95% CI) |",
         "|---|---|" + "---|" * len(TASKS) + "---|---|"]
for name, tag in rows:
    r = res[tag]; lo, hi = cp(r["total"], N)
    mark = " **← α\\***" if tag == f"b13_cal_a{best}" else ""
    lines.append(f"| {name}{mark} | {r['used']} | " + " | ".join(str(r["by_task"][t]) for t in TASKS)
                 + f" | {r['total']} | {100 * r['total'] / N:.1f} ({lo:.0f}–{hi:.0f}) |")
table = "\n".join(lines)
now = datetime.datetime.now().strftime("%F %T")
means = " ".join(f"{a}:{100 * r['total'] / N:.1f}" for a, r in cand) + f" (θ0 {100 * res['b13_cal_theta0']['total'] / N:.1f})"
rule = "보정 task (t0 t2 t4 t6 × 초기 상태 0–19) 평균 성공률 최고, 동률이면 작은 α (격자 0.2 0.4 0.6 0.8 1.0 고정)"

(R / "b13_table2.md").write_text(
    f"# [13] Baseline v2 — 표 2: α 선택 (보정 task t0 t2 t4 t6 × 초기 상태 0–19, xArm7, 조건 v0, seed 7)\n\n"
    f"{table}\n\n- 규칙: {rule}\n- α\\* = **{best}** (최고 평균 동률 α: {' '.join(ties)})\n"
    f"- 참고: t0 열만 볼 때 최고 α = {t0_best} (동률: {' '.join(t0_ties)}) — 선택에는 안 씀\n- 기록 {now} (보고 task 평가 전)\n")
(R / "b13_alpha_choice.txt").write_text(
    f"alpha_star={best}\nrule={rule}\nmeans={means}\nties_at_best={' '.join(ties)}\n"
    f"t0_only_best={t0_best} (동률: {' '.join(t0_ties)}; 참고만, 선택에는 안 씀)\n"
    f"tags={' '.join(res[t]['used'] for _, t in rows)}\nrecorded={now} (보고 task t1 t3 t5 t7 평가 전)\n\n{table}\n")
print(table)
print(f"alpha_star={best} | means {means} | t0_only_best={t0_best}")
