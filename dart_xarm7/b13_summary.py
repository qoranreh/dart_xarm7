"""[13] Baseline v2 — 3단계 본 평가 집계 (결과만 읽음, 새 실행 없음).  /usr/bin/python3 ~/hayeon/dart_xarm7/b13_summary.py

보고 task t1 t3 t5 t7 × 초기 상태 0–49 (태그마다 200행). α* 는 results/b13_alpha_choice.txt 의 alpha_star.
- 태그마다 200행·errors 0·(task_id, episode) 중복 없음·집합이 정확히 t1 t3 t5 t7 × 0–49 인지 assert. <태그>_r2 가 있으면 그것을 쓴다.
- 태그별: task 성공/50, 합계/200, Clopper–Pearson 95% CI.
- 짝 비교 ((task_id, episode) 로 맞춤): DART(α*) − zero-shot (주 비교), DART(α*) − 원샷 FT, DART(α*) − DART(0.4) (α*≠0.4 일 때, 참고)
  차이 %p · 짝 부트스트랩 95% CI: task 별로 묶어 각 task 안에서 50쌍을 복원추출 (task 구성 고정), 1만 번,
  비교마다 np.random.default_rng(0), task 순서 1 3 5 7 로 (10000, 50) 씩 뽑음, 백분위 2.5·97.5
  · McNemar 정확 검정 양측 p (엇갈린 쌍에 이항 검정, p=0.5).
- 판정 (프롬프트 STOP2, 미리 정한 기준 그대로):
  대조군 통과 = θ0·Panda ≥ 90% 이고 원샷 FT ≤ 10%
  지지 = 차이 ≥ +5%p 이고 짝 CI 하한 > 0 / 역효과 = 차이 ≤ −5%p 이고 CI 상한 < 0 / 효과 없음 = |차이| < 1.5%p / 그 밖 = 판단 불가
  H1 = DART(α*) vs zero-shot, H2 = DART(α*) vs 원샷 FT (STOP2 의 이름. 카드의 H2 '안 옮겨 줌' 과 다름 — 카드 판정은 card_hypothesis)
  [12] 진행 조건 = H1 지지 + DART(α*) 평균 30% 이상 90% 미만
출력: results/b13_table1.md, results/b13_summary.json
"""
import collections
import datetime
import json
import pathlib

import numpy as np
from scipy.stats import binomtest

R = pathlib.Path.home() / "hayeon/dart_xarm7/results"
TASKS = [1, 3, 5, 7]
EPS = range(50)
N = len(TASKS) * len(EPS)
B = 10000

choice = dict(l.split("=", 1) for l in (R / "b13_alpha_choice.txt").read_text().splitlines() if "=" in l and not l.startswith("|"))
AS = choice["alpha_star"]
ROWS = [("panda", "b13_panda", "θ0 · Panda (양성 대조)", "Panda"),
        ("theta0", "b13_theta0", "zero-shot θ0 · xArm7", "XArm7"),
        ("tgt", "b13_tgt", "원샷 FT θ_tgt · xArm7", "XArm7"),
        ("dart", "b13_dart_main", f"DART θ*(α*={AS}) · xArm7", "XArm7")]
COMP = [("H1", "DART(α*) − zero-shot", "dart", "theta0"),
        ("H2", "DART(α*) − 원샷 FT", "dart", "tgt")]
if AS != "0.4":
    ROWS.append(("dart04", "b13_dart_a0.4", "DART θ*(α=0.4, 참고·논문 값) · xArm7", "XArm7"))
    COMP.append(("ref", "DART(α*) − DART(0.4)", "dart", "dart04"))


def load(tag, robot):
    used = f"{tag}_r2" if (R / f"{tag}_r2" / "summary.txt").exists() else tag
    rows = [json.loads(l) for f in sorted((R / used).glob("client*.jsonl")) for l in open(f) if l.strip()]
    keys = collections.Counter((x["task_id"], x["episode"]) for x in rows)
    assert len(rows) == N, f"{used}: {len(rows)}행 (기대 {N})"
    assert not any(x.get("error") for x in rows), f"{used}: errors {sum(1 for x in rows if x.get('error'))}"
    assert max(keys.values()) == 1, f"{used}: (task_id, episode) 중복"
    assert set(keys) == {(t, e) for t in TASKS for e in EPS}, f"{used}: (task_id, episode) 집합이 t1 t3 t5 t7 × 0–49 가 아님"
    assert {x["robot"] for x in rows} == {robot}, f"{used}: robot {sorted({x['robot'] for x in rows})} (기대 {robot})"
    return used, {(x["task_id"], x["episode"]): int(x["success"]) for x in rows}


def cp(k, n):
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return 100 * ci.low, 100 * ci.high


def paired(a, b):
    assert set(a) == set(b)
    d = {t: np.array([a[(t, e)] - b[(t, e)] for e in EPS]) for t in TASKS}
    rng = np.random.default_rng(0)
    boot = np.zeros(B)
    for t in TASKS:
        boot += d[t][rng.integers(0, len(EPS), size=(B, len(EPS)))].sum(axis=1)
    lo, hi = np.percentile(100 * boot / N, [2.5, 97.5])
    nb = sum(1 for k in a if a[k] == 1 and b[k] == 0)  # A 만 성공
    nc = sum(1 for k in a if a[k] == 0 and b[k] == 1)  # B 만 성공
    p = binomtest(nb, nb + nc, 0.5).pvalue if nb + nc else 1.0
    return {"diff_pp": 100 * sum(v.sum() for v in d.values()) / N, "ci_lo": float(lo), "ci_hi": float(hi),
            "only_a": nb, "only_b": nc, "mcnemar_p": float(p),
            "by_task_pp": {f"t{t}": 100 * float(d[t].mean()) for t in TASKS}}


def verdict(c):
    if c["diff_pp"] >= 5 and c["ci_lo"] > 0:
        return "지지"
    if c["diff_pp"] <= -5 and c["ci_hi"] < 0:
        return "역효과"
    if abs(c["diff_pp"]) < 1.5:
        return "효과 없음"
    return "판단 불가"


meta_main = (R / ("b13_dart_main_r2" if (R / "b13_dart_main_r2/summary.txt").exists() else "b13_dart_main") / "meta.txt").read_text()
assert f"/dart_a{AS}\n" in meta_main and f"alpha_star={AS}" in meta_main, "b13_dart_main meta.txt 의 체크포인트·α* 가 b13_alpha_choice.txt 와 다름"

out = {"alpha_star": AS, "alpha_choice": {k: choice[k] for k in ("means", "t0_only_best", "recorded") if k in choice},
       "rows": {}, "comparisons": {}}
for key, tag, name, robot in ROWS:
    used, s = load(tag, robot)
    by = {f"t{t}": sum(s[(t, e)] for e in EPS) for t in TASKS}
    k = sum(by.values()); lo, hi = cp(k, N)
    out["rows"][key] = {"tag": used, "name": name, "by_task": by, "success": k, "n": N,
                        "rate": 100 * k / N, "cp_lo": lo, "cp_hi": hi, "_s": s}
for cid, name, a, b in COMP:
    c = paired(out["rows"][a]["_s"], out["rows"][b]["_s"])
    c.update(name=name, a=out["rows"][a]["tag"], b=out["rows"][b]["tag"], verdict=verdict(c) if cid != "ref" else "참고 (판정 규칙 없음)")
    out["comparisons"][cid] = c
for r in out["rows"].values():
    r.pop("_s")

rate = {k: r["rate"] for k, r in out["rows"].items()}
ctrl = rate["panda"] >= 90 and rate["tgt"] <= 10
h1, h2 = out["comparisons"]["H1"]["verdict"], out["comparisons"]["H2"]["verdict"]
go12 = h1 == "지지" and 30 <= rate["dart"] < 90
out["judgment"] = {"controls_pass": ctrl, "H1_dart_vs_zeroshot": h1, "H2_dart_vs_oneshot": h2, "go_exp12": go12,
                   "card_hypothesis": "H1" if h1 == "지지" else "H2"}
out["method"] = {"bootstrap": f"task 층화 짝 부트스트랩 (각 task 안에서 50쌍 복원추출), {B}회, 비교마다 default_rng(0), 백분위 2.5·97.5",
                 "mcnemar": "엇갈린 쌍 이항 검정 (p=0.5) 양측 정확", "ci_rate": "Clopper–Pearson 95%",
                 "generated": datetime.datetime.now().strftime("%F %T")}

L = [f"# [13] Baseline v2 — 표 1: 본 평가 (보고 task t1 t3 t5 t7 × 초기 상태 0–49, 조건 v0, seed 7)", "",
     f"α* = **{AS}** (results/b13_alpha_choice.txt, 기록 {choice.get('recorded', '?')}; 보정 평균 {choice.get('means', '?')})", "",
     "| 방법 · 로봇 | 태그 | " + " | ".join(f"t{t} /50" for t in TASKS) + f" | 합계 /{N} | 평균 % (95% CI) |",
     "|---|---|" + "---|" * len(TASKS) + "---|---|"]
for r in out["rows"].values():
    L.append(f"| {r['name']} | {r['tag']} | " + " | ".join(str(r["by_task"][f"t{t}"]) for t in TASKS)
             + f" | {r['success']} | {r['rate']:.1f} ({r['cp_lo']:.1f}–{r['cp_hi']:.1f}) |")
L += ["", f"## 짝 비교 ((task_id, episode) {N}쌍)", "",
      "| 비교 | 차이 %p | 짝 부트스트랩 95% CI | A만 성공 / B만 성공 | McNemar 정확 p | task별 차이 %p (t1 t3 t5 t7) | 판정 |",
      "|---|---|---|---|---|---|---|"]
for cid, c in out["comparisons"].items():
    lab = {"H1": "H1 ", "H2": "H2 ", "ref": ""}[cid]
    L.append(f"| {lab}{c['name']} | {c['diff_pp']:+.1f} | {c['ci_lo']:+.1f} ~ {c['ci_hi']:+.1f} | {c['only_a']} / {c['only_b']} | "
             f"{c['mcnemar_p']:.3g} | " + " ".join(f"{v:+.0f}" for v in c["by_task_pp"].values()) + f" | {c['verdict']} |")
J = [f"대조군 통과: {'예' if ctrl else '아니오'} (θ0·Panda {rate['panda']:.1f}% ≥ 90, 원샷 FT {rate['tgt']:.1f}% ≤ 10)",
     f"H1 (DART(α*) vs zero-shot): {h1}",
     f"H2 (DART(α*) vs 원샷 FT): {h2}",
     f"[12] 진행 조건: {'충족' if go12 else '미충족'} (H1 지지 + DART(α*) 평균 {rate['dart']:.1f}% 가 30 이상 90 미만)",
     f"카드 가설 판정: {out['judgment']['card_hypothesis']} (H1 = DART − zero-shot 이 +5%p 이상이고 짝 CI 하한 > 0, 아니면 H2)"]
L += ["", "## 판정 (미리 정한 기준 그대로)", ""] + [f"- {j}" for j in J] + [
      "", f"방법: {out['method']['bootstrap']} · McNemar {out['method']['mcnemar']} · 평균 CI {out['method']['ci_rate']}. "
      "STOP2 의 'H2 (DART vs 원샷 FT)' 는 카드의 H2 ('안 옮겨 줌') 와 이름만 같다.", ""]
(R / "b13_table1.md").write_text("\n".join(L))
(R / "b13_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
print("\n".join(L[:len(out["rows"]) + 6]))
print("\n".join(J))
