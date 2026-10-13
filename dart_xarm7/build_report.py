"""report/stats.json + 영상·프레임 그리드를 내장한 단일 HTML 검증 리포트를 만든다 (Artifact 로 게시).

  /usr/bin/python3 ~/hayeon/dart_xarm7/build_report.py
"""
import base64
import html
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from report_log_section import TOC, log_section_html  # noqa: E402

R = pathlib.Path.home() / "hayeon/dart_xarm7/report"
s = json.loads((R / "stats.json").read_text())
V = s["variants"]
TASK = s["pair_summary"]["task"]
b64 = lambda f: base64.b64encode((R / f).read_bytes()).decode()

ADIM = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "grip"]
SDIM = ["x", "y", "z", "rx", "ry", "rz", "g1", "g2"]
# make_demo_pair 의 Panda 녹화 vs 공개 SNUMPR/dart-libero-viewpoints episode 4 (같은 태스크) 대조 결과 (2026-10-08 측정)
OFFICIAL = {"state_max": "1e-4", "actions_max": "0", "img_mean": "2.3", "img_px20": "0.57%"}
# make_retarget_demo 직후 측정한 Panda 자신의 추종 오차 (명령 목표 vs 다음 프레임 실제 포즈)
PANDA_TRACK = {"pos_mean": 3.0, "pos_max": 5.6, "rot_mean": 2.2, "rot_max": 5.3}
gap, ga = s["pad_gap_cm"], s["grip_action"]
ci = next(i for i, a in enumerate(ga) if a > 0)
oi = next(i for i in range(ci, len(ga)) if ga[i] < 0)
ret = V[s["adopted"]]
rs = s["adopted_summary"]
raw = V["retarget"]


def f(x, d=3):
    return f"{x:.{d}f}".replace("-", "−")


def cls(n):
    return ' class="warn"' if abs(n) > 1.01 else ""


def action_rows():
    P, X = s["actions"]["Panda"], s["actions"]["XArm7"]
    q01, q99 = s["norm_stats_q01"]["actions"], s["norm_stats_q99"]["actions"]
    out = []
    for i, n in enumerate(ADIM):
        out.append(
            f"<tr><th>{n}</th><td>{f(P['mean'][i])}</td><td>{f(X['mean'][i])}</td>"
            f"<td>{f(P['min'][i], 2)} … {f(P['max'][i], 2)}</td><td>{f(X['min'][i], 2)} … {f(X['max'][i], 2)}</td>"
            f"<td class='muted'>{f(q01[i], 2)} … {f(q99[i], 2)}</td>"
            f"<td><span{cls(P['norm_min'][i])}>{f(P['norm_min'][i], 2)}</span> … <span{cls(P['norm_max'][i])}>{f(P['norm_max'][i], 2)}</span></td>"
            f"<td><span{cls(X['norm_min'][i])}>{f(X['norm_min'][i], 2)}</span> … <span{cls(X['norm_max'][i])}>{f(X['norm_max'][i], 2)}</span></td>"
            f"<td>{P['out_of_q01_q99_pct'][i] * 100:.0f}% / <span{' class=warn' if X['out_of_q01_q99_pct'][i] > 0.1 else ''}>{X['out_of_q01_q99_pct'][i] * 100:.0f}%</span></td></tr>")
    return "\n".join(out)


def state_rows():
    P, X = s["state"]["Panda"], s["state"]["XArm7"]
    out = []
    for i, n in enumerate(SDIM):
        out.append(
            f"<tr><th>{n}</th><td>{f(P['mean'][i])}</td><td>{f(X['mean'][i])}</td>"
            f"<td>{f(P['min'][i])} … {f(P['max'][i])}</td><td>{f(X['min'][i])} … {f(X['max'][i])}</td>"
            f"<td>{f(P['norm_min'][i], 2)} … {f(P['norm_max'][i], 2)}</td><td>{f(X['norm_min'][i], 2)} … {f(X['norm_max'][i], 2)}</td></tr>")
    return "\n".join(out)


def variant_rows():
    names = {"replay": "액션 재생", "retarget": "MimicGen 방식 (제한 없음)", "retarget_rotq": "MimicGen + 회전 보정 제한 (채택)"}
    out = []
    for k in ("replay", "retarget", "retarget_rotq"):
        if k not in V:
            continue
        v = V[k]
        rot = "—" if v["track_rot_deg_mean"] is None else f"{v['track_rot_deg_mean']:.1f} / {v['track_rot_deg_max']:.1f}"
        out.append(
            f"<tr{' class=sel' if k == s['adopted'] else ''}><th>{names[k]}</th>"
            f"<td>{v['first_success']} ✓</td><td>{v['eef_diff_cm_mean']:.1f} / {v['eef_diff_cm_max']:.1f}</td>"
            f"<td>{rot}</td>"
            f"<td>{f(v['action_absdiff_mean'][0], 2)} / {f(v['action_absdiff_mean'][2], 2)} / {f(v['action_absdiff_mean'][4], 2)}</td>"
            f"<td><span{cls(v['norm_max'][4])}>{f(v['norm_max'][4], 2)}</span></td>"
            f"<td>{v['out_pct'][0] * 100:.0f} / {v['out_pct'][2] * 100:.0f} / {v['out_pct'][3] * 100:.0f} / {v['out_pct'][4] * 100:.0f}%</td></tr>")
    return "\n".join(out)


def grip_rows():
    picks = [ci - 1, ci + 2, ci + 4, ci + 6, ci + 8, ci + 10, ci + 12, oi - 1, oi + 4, oi + 8]
    head = "".join(f"<th>{t}</th>" for t in picks)
    rp = "".join(f"<td>{gap['Panda'][t]:.2f}</td>" for t in picks)
    rx = "".join(f"<td>{gap['XArm7'][t]:.2f}</td>" for t in picks)
    rc = "".join(f"<td>{'+1 닫기' if ga[t] > 0 else '−1 열기'}</td>" for t in picks)
    return f"<tr><th>프레임</th>{head}</tr><tr><th>그리퍼 명령</th>{rc}</tr><tr><th class='panda'>Panda</th>{rp}</tr><tr><th class='xarm'>xArm7</th>{rx}</tr>"


LOSS = json.loads((R / "loss_curves.json").read_text()) if (R / "loss_curves.json").exists() else None
CHART = ""
if LOSS:
    sl, tl = LOSS["src"], LOSS["tgt"]
    rows = "".join(f"<tr><th>{a[0]}</th><td>{a[1]:.4f}</td><td>{b[1]:.4f}</td></tr>" for a, b in zip(sl, tl) if a[0] % 100 == 0 or a[0] >= 950)
    avg = lambda c: sum(v for k, v in c if k >= 900) / len([1 for k, v in c if k >= 900])
    CHART = """
<section id="loss">
  <h2>one-shot 학습 loss (θ0 → θ_src / θ_tgt)</h2>
  <div class="legend"><span><i class="sw panda-bg"></i>source · Panda 데모 → θ_src</span><span><i class="sw xarm-bg"></i>target · xArm7 데모 → θ_tgt</span></div>
  <div class="chartbox"><svg id="losschart" role="img" aria-label="학습 스텝에 따른 loss, 로그 스케일. source 0.0041에서 0.0017로, target 0.1218에서 0.0026으로 감소"></svg>
    <div id="losstip" class="tip" hidden></div></div>
  <p class="note">y축 로그 스케일. 10스텝마다 기록된 값 (각 지점은 직전 10스텝 평균, step 0 은 첫 스텝). batch 64, 1000스텝, pi05_libero_oneshotft, A100 1장.
  source 0.0041 → 0.0017 (마지막 100스텝 평균 __SA__), target 0.1218 → 0.0026 (__TA__). 학습 시간 source __SW__, target __TW__.
  target 은 θ0 가 xArm7 데이터를 처음 봐서 시작 loss 가 약 30배 높고 50~100스텝 안에 source 수준으로 내려온다. 잡 __JOB__ 실행분 (10-08 실행과 끝값 거의 같음).</p>
  <details><summary>숫자 표</summary><div class="tbl"><table><thead><tr><th>step</th><th class="panda">source</th><th class="xarm">target</th></tr></thead><tbody>__ROWS__</tbody></table></div></details>
</section>
<script>
(() => {
  const S = __S__, T = __T__;
  const svg = document.getElementById("losschart"), tip = document.getElementById("losstip");
  const W = 900, H = 340, m = {l: 58, r: 120, t: 14, b: 40};
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const lo = Math.log10(0.001), hi = Math.log10(0.15);
  const x = s => m.l + (s / 1000) * (W - m.l - m.r);
  const y = v => m.t + (hi - Math.log10(v)) / (hi - lo) * (H - m.t - m.b);
  const ns = "http://www.w3.org/2000/svg", el = (n, a, p = svg) => { const e = document.createElementNS(ns, n); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
  for (const v of [0.001, 0.003, 0.01, 0.03, 0.1]) {
    el("line", {x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "grid"});
    el("text", {x: m.l - 8, y: y(v) + 4, class: "tick", "text-anchor": "end"}).textContent = v;
  }
  for (const s of [0, 200, 400, 600, 800, 1000]) el("text", {x: x(s), y: H - m.b + 18, class: "tick", "text-anchor": "middle"}).textContent = s;
  el("text", {x: (m.l + W - m.r) / 2, y: H - 4, class: "axis", "text-anchor": "middle"}).textContent = "학습 스텝";
  el("text", {x: 14, y: (m.t + H - m.b) / 2, class: "axis", "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})`}).textContent = "loss";
  const path = d => d.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
  el("path", {d: path(S), class: "ln panda-st"}); el("path", {d: path(T), class: "ln xarm-st"});
  const end = (d, cls, label) => { const p = d[d.length - 1]; el("circle", {cx: x(p[0]), cy: y(p[1]), r: 4, class: cls + " dot"});
    el("text", {x: x(p[0]) + 9, y: y(p[1]) + 4, class: "lab"}).textContent = `${label} ${p[1].toFixed(4)}`; };
  end(S, "panda-fill", "source"); end(T, "xarm-fill", "target");
  // 시작점 라벨 (target 만 크게 다름)
  el("text", {x: x(T[0][0]) + 8, y: y(T[0][1]) + 4, class: "lab"}).textContent = `target 시작 ${T[0][1].toFixed(4)}`;
  const cross = el("line", {y1: m.t, y2: H - m.b, class: "cross", visibility: "hidden"});
  const ds = el("circle", {r: 5, class: "panda-fill dot", visibility: "hidden"}), dt = el("circle", {r: 5, class: "xarm-fill dot", visibility: "hidden"});
  const hit = el("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent"});
  const show = ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) / r.width * W;
    const i = Math.max(0, Math.min(S.length - 1, Math.round(((px - m.l) / (W - m.l - m.r)) * 1000 / 10)));
    const a = S[i], b = T[i], cx = x(a[0]);
    for (const e of [cross, ds, dt]) e.setAttribute("visibility", "visible");
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx);
    ds.setAttribute("cx", cx); ds.setAttribute("cy", y(a[1])); dt.setAttribute("cx", cx); dt.setAttribute("cy", y(b[1]));
    tip.hidden = false; tip.innerHTML = `<b>step ${a[0]}</b><span><i class="sw panda-bg"></i>source ${a[1].toFixed(4)}</span><span><i class="sw xarm-bg"></i>target ${b[1].toFixed(4)}</span>`;
    const left = cx / W * r.width; tip.style.left = Math.min(left + 12, r.width - 170) + "px"; tip.style.top = "8px";
  };
  hit.addEventListener("pointermove", show); hit.addEventListener("pointerdown", show);
  hit.addEventListener("pointerleave", () => { tip.hidden = true; for (const e of [cross, ds, dt]) e.setAttribute("visibility", "hidden"); });
})();
</script>
""".replace("__S__", json.dumps(sl)).replace("__T__", json.dumps(tl)).replace("__ROWS__", rows) \
       .replace("__SA__", f"{avg(sl):.4f}").replace("__TA__", f"{avg(tl):.4f}") \
       .replace("__SW__", f"{LOSS['wall']['src'] // 3600}시간 {LOSS['wall']['src'] % 3600 // 60}분").replace("__TW__", f"{LOSS['wall']['tgt'] // 3600}시간 {LOSS['wall']['tgt'] % 3600 // 60}분") \
       .replace("__JOB__", LOSS["job"])


# 평가 성공률 (summarize_eval.py 가 만든 summary.json; 오늘 실행 + 10-09 실행 비교)
def _load(d):
    f = pathlib.Path.home() / "hayeon/dart_xarm7" / d / "summary.json"
    return json.loads(f.read_text()) if f.exists() else {}
EV_NOW, EV_OLD = _load("results"), _load("results_20261009_job129138")
EVAL_HTML = ""
if EV_NOW:
    order = [("theta0", "(a) θ0"), ("theta_tgt", "(b) θ_tgt one-shot FT"), ("dart_a0.5", "(c) θ* α=0.5"), ("dart_a0.8", "(c) θ* α=0.8"), ("dart_a1.0", "(c) θ* α=1.0")]
    pct = lambda v: 100 * v[0] / max(1, v[1])
    best = max(pct(EV_NOW[k]["heldout"]) for k, _ in order if k in EV_NOW)
    trs = []
    for k, name in order:
        if k not in EV_NOW: continue
        v = EV_NOW[k]; pt = v["per_task"]
        cells = "".join(f"<td{' class=zero' if pt[str(t)][0] == 0 else ''}>{pt[str(t)][0]}</td>" for t in range(10))
        lo, hi = v["heldout_ci95"]; h = pct(v["heldout"])
        trs.append(f"<tr{' class=sel' if h == best else ''}><th>{name}</th>{cells}<td>{pct(v['adapt']):.0f}%</td>"
                   f"<td class='hd'>{h:.0f}% <span class='muted'>({100*lo:.0f}–{100*hi:.0f})</span></td><td>{pct(v['all']):.0f}%</td></tr>")
    comp = []
    for k, name in order:
        if k in EV_NOW and k in EV_OLD:
            a, b = EV_OLD[k]["heldout"], EV_NOW[k]["heldout"]
            comp.append(f"<tr><th>{name}</th><td>{a[0]}/{a[1]} = {pct(a):.0f}%</td><td>{b[0]}/{b[1]} = {pct(b):.0f}%</td><td>{(a[0]+b[0])}/{(a[1]+b[1])} = {100*(a[0]+b[0])/(a[1]+b[1]):.0f}%</td></tr>")
    EVAL_HTML = f"""
<section id="eval">
  <h2>cross-embodiment 평가 성공률 (xArm7, libero_spatial)</h2>
  <p>θ0 · θ_tgt · DART θ* 를 xArm7 로 평가. 태스크당 10 에피소드 (공식 초기 상태 0~9). <b>t0 = adaptation 태스크</b> (one-shot 데모를 만든 태스크), <b>t1–t9 = held-out</b>.
  DART 의 핵심 주장은 held-out 에서 (c) θ* 가 (b) θ_tgt 보다 좋다는 것.</p>
  <div class="tbl"><table class="ev">
    <thead><tr><th>모델</th>{"".join(f"<th>t{t}</th>" for t in range(10))}<th>adapt t0</th><th>held-out t1–9 (95% CI)</th><th>전체</th></tr></thead>
    <tbody>{"".join(trs)}</tbody></table></div>
  <p class="note">칸의 숫자 = 10 에피소드 중 성공 수. 2026-10-10 실행 (잡 129459, θ_src/θ_tgt 재학습). 강조 행 = held-out 최고.
  held-out Fisher exact: θ*0.8 vs θ_tgt p=3e-22, θ*0.5 vs θ0 p=0.013, θ*0.8 vs θ0 p=0.45, θ*1.0 vs θ0 p=1.0.
  실패는 전부 시간초과(220 스텝), 프로그램 에러 0건. θ* 는 공식 CPU 합성. 참고: 같은 클라이언트로 θ0 를 Panda 에서 돌리면 100/100 (10-09).</p>
  <h3>두 번 실행한 held-out 결과 (각 90 에피소드)</h3>
  <div class="tbl"><table class="ev2"><thead><tr><th>모델</th><th>10-09 실행</th><th>10-10 실행</th><th>합계</th></tr></thead><tbody>{"".join(comp)}</tbody></table></div>
  <ul class="findings">
    <li><b>θ* &gt; θ_tgt 는 세 α 모두, 두 번 다 압도적으로 재현.</b> θ_tgt 는 adaptation 태스크만 성공하고 held-out 은 0% — 장면을 무시하고 데모 궤적을 따라감.</li>
    <li><b>α=0.5 가 가장 좋고 θ0 보다 유의하게 높음 (두 번 모두).</b> α=0.8 은 θ0 와 비슷, α=1.0 은 θ0 이하~비슷. α 는 이 평가 결과로 고른 것이라 낙관적일 수 있음 (논문은 별도 seed 로 튜닝, MimicGen cross-embodiment 에서 α=0.4).</li>
    <li><b>adapt t0 한 태스크 10 에피소드는 노이즈가 큼</b> — 같은 θ0 가 3/10 (10-09) vs 7/10 (10-10). 비교는 held-out 합계로.</li>
    <li><b>계속 어려운 태스크:</b> t4 (서랍 속 그릇, 서랍 안에서 재파지 반복), t5 (ramekin 위, 기운 채 잡아 떨어뜨림) — 모든 모델 0~30%.</li>
  </ul>
</section>"""


# 대표 평가 영상 (results/videos, 태스크별 episode 0; agentview | 손목, 10fps)
VID = pathlib.Path.home() / "hayeon/dart_xarm7/results/videos"
PICKS = [
    ("held-out t1 비교", "t1 “pick up the black bowl next to the ramekin and place it on the plate”", [
        ("theta0_t1_ep0_failure", "(a) θ0 — 실패", "왼쪽 위 그릇으로 가지만 못 잡고, 빈손으로 접시까지 가서 시간초과까지 맴돈다."),
        ("theta_tgt_t1_ep0_failure", "(b) θ_tgt — 실패", "그릇은 보지 않고 adaptation 데모에서 그릇이 있던 접시 쪽으로 곧장 가서 머문다 (one-shot 과적합)."),
        ("dart_a0.5_t1_ep0_success", "(c) θ* α=0.5 — 성공 (94스텝)", "왼쪽 위 그릇을 잡아 접시에 놓는다."),
    ]),
    ("adaptation 태스크와 계속 어려운 태스크", "", [
        ("theta_tgt_t0_ep0_success", "(b) θ_tgt, t0 (adaptation) — 성공 (67스텝)", "데모를 만든 태스크에서는 θ_tgt 도 잘한다. 이 궤적을 다른 태스크에서도 그대로 반복하는 것이 위의 실패."),
        ("dart_a0.5_t4_ep0_failure", "(c) θ* α=0.5, t4 서랍 속 그릇 — 실패", "서랍 안 그릇을 잡으려다 팔이 서랍·캐비닛에 걸린 채 재파지만 반복한다."),
        ("dart_a0.5_t5_ep0_failure", "(c) θ* α=0.5, t5 ramekin 위 그릇 — 실패", "그릇을 들어 옮기지만 기운 채 잡혀서 접시 옆에 뒤집혀 떨어진다."),
    ]),
]
VIDEO_HTML = ""
if VID.exists():
    groups = []
    for title, sub, items in PICKS:
        cards = []
        for vf, cap, desc in items:
            fp = VID / f"{vf}.mp4"
            if not fp.exists():
                continue
            b = base64.b64encode(fp.read_bytes()).decode()
            cards.append(f'<figure class="vcard"><video controls muted loop playsinline preload="metadata" src="data:video/mp4;base64,{b}"></video>'
                         f'<figcaption><b>{cap}</b><span>{desc}</span></figcaption></figure>')
        groups.append(f'<h3>{title}</h3>' + (f'<p class="note">{html.escape(sub)}</p>' if sub else "") + f'<div class="vgrid">{"".join(cards)}</div>')
    VIDEO_HTML = f"""
<section id="videos">
  <h2>대표 평가 영상 (xArm7)</h2>
  <p class="note">각 영상 왼쪽 = agentview, 오른쪽 = 손목 카메라 (모델 입력 그대로, 224×224 씩), 10fps. 태스크별 에피소드 0 만 저장됨. 2026-10-10 실행.
  전체 50개: <code>~/hayeon/dart_xarm7/results/videos/</code></p>
  {"".join(groups)}
</section>"""

LOG_HTML = log_section_html()


page = f"""<title>Panda·xArm7 데모 쌍</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* 레이아웃: 실험 노트 한 장 — 위에서 아래로 읽는 단일 컬럼, 영상·표만 폭을 넓게 */
:root {{
  --bg: #f6f7f8; --surface: #ffffff; --fg: #1c2328; --muted: #5d6a73; --line: #d9dee2;
  --panda: #c4622d; --xarm: #00819b; --ok: #2f7d4a; --warn: #b5541a; --warn-bg: #fbeee4; --sel: #e8f3f5;
  --sans: "IBM Plex Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #12171a; --surface: #1a2125; --fg: #e3e8eb; --muted: #94a2ab; --line: #2c363c;
  --panda: #d9733a; --xarm: #1aa0bd; --ok: #6cc28a; --warn: #f0a070; --warn-bg: #3a2519; --sel: #17323a; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #12171a; --surface: #1a2125; --fg: #e3e8eb; --muted: #94a2ab; --line: #2c363c;
  --panda: #d9733a; --xarm: #1aa0bd; --ok: #6cc28a; --warn: #f0a070; --warn-bg: #3a2519; --sel: #17323a; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--sans); font-size: 15px; line-height: 1.6; }}
main {{ max-width: 1000px; margin: 0 auto; padding-inline: 20px; padding-block: 32px 64px; display: grid; gap: 40px; }}
h1 {{ font-size: 1.9rem; line-height: 1.25; margin: 0; text-wrap: balance; font-weight: 700; }}
h2 {{ font-size: 1.15rem; margin: 0 0 10px; text-wrap: balance; }}
p {{ margin: 0; max-width: 70ch; }}
.eyebrow {{ font-family: var(--mono); font-size: .75rem; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
header {{ display: grid; gap: 10px; }}
.panda {{ color: var(--panda); }} .xarm {{ color: var(--xarm); }}
.facts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); }}
.fact {{ background: var(--surface); padding: 12px 14px; display: grid; gap: 2px; min-width: 0; }}
.fact b {{ font-family: var(--mono); font-weight: 500; font-size: 1.05rem; font-variant-numeric: tabular-nums; }}
.fact b.warn {{ color: var(--warn); }}
.fact span {{ color: var(--muted); font-size: .82rem; }}
video, img {{ width: 100%; max-width: 100%; height: auto; display: block; border: 1px solid var(--line); background: #000; }}
img {{ background: var(--surface); }}
.note {{ color: var(--muted); font-size: .85rem; margin-top: 6px; }}
section {{ display: grid; gap: 10px; min-width: 0; }}
.tbl {{ overflow-x: auto; border: 1px solid var(--line); background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; font-size: .85rem; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }}
td {{ font-family: var(--mono); }}
th {{ text-align: left; font-weight: 600; }}
thead th {{ color: var(--muted); font-weight: 500; font-size: .78rem; text-align: right; }}
thead th:first-child {{ text-align: left; }}
tr:last-child th, tr:last-child td {{ border-bottom: 0; }}
tr.sel {{ background: var(--sel); }}
.muted {{ color: var(--muted); }}
.warn {{ color: var(--warn); font-weight: 600; }}
.findings {{ display: grid; gap: 12px; margin: 0; padding: 0; list-style: none; }}
.findings li {{ background: var(--surface); border: 1px solid var(--line); padding: 12px 14px; display: grid; gap: 4px; }}
.findings li.attn {{ background: var(--warn-bg); }}
.findings b {{ font-weight: 600; }}
code {{ font-family: var(--mono); font-size: .85em; word-break: break-all; }}
.paths {{ display: grid; gap: 6px; }}
.paths div {{ background: var(--surface); border: 1px solid var(--line); padding: 8px 12px; min-width: 0; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: .85rem; color: var(--fg); }}
.legend span, .tip span {{ display: inline-flex; align-items: center; gap: 6px; }}
.sw {{ display: inline-block; width: 14px; height: 3px; border-radius: 2px; }}
.panda-bg {{ background: var(--panda); }} .xarm-bg {{ background: var(--xarm); }}
.chartbox {{ position: relative; border: 1px solid var(--line); background: var(--surface); }}
#losschart {{ display: block; width: 100%; height: auto; }}
#losschart .grid {{ stroke: var(--line); stroke-width: 1; }}
#losschart .tick {{ fill: var(--muted); font: 12px var(--mono); }}
#losschart .axis {{ fill: var(--muted); font: 12px var(--sans); }}
#losschart .lab {{ fill: var(--fg); font: 12px var(--mono); }}
#losschart .ln {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
#losschart .panda-st {{ stroke: var(--panda); }} #losschart .xarm-st {{ stroke: var(--xarm); }}
#losschart .panda-fill {{ fill: var(--panda); }} #losschart .xarm-fill {{ fill: var(--xarm); }}
#losschart .dot {{ stroke: var(--surface); stroke-width: 2; }}
#losschart .cross {{ stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }}
.tip {{ position: absolute; display: grid; gap: 2px; padding: 6px 10px; background: var(--surface); border: 1px solid var(--line); font: 12px var(--mono); pointer-events: none; box-shadow: 0 2px 8px rgba(0,0,0,.12); }}
details summary {{ cursor: pointer; color: var(--muted); font-size: .85rem; margin-top: 6px; }}
table.ev td, table.ev th {{ padding: 6px 8px; }}
table.ev td.zero {{ color: var(--muted); }}
table.ev td.hd {{ font-weight: 600; }}
h3 {{ font-size: 1rem; margin: 10px 0 0; }}
.vgrid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }}
.vcard {{ margin: 0; display: grid; gap: 6px; min-width: 0; }}
.vcard figcaption {{ display: grid; gap: 2px; font-size: .85rem; }}
.vcard figcaption span {{ color: var(--muted); }}
.toc {{ display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: .85rem; padding: 10px 12px; border: 1px solid var(--line); background: var(--surface); }}
.toc a {{ color: var(--xarm); text-decoration: none; }} .toc a:hover {{ text-decoration: underline; }}
.tag {{ display: inline-block; font-size: .72rem; padding: 1px 6px; border-radius: 3px; white-space: nowrap; }}
.tag-perm {{ background: var(--sel); color: var(--fg); }} .tag-vol {{ background: var(--warn-bg); color: var(--warn); }}
table.log td, table.log th {{ white-space: normal; text-align: left; vertical-align: top; min-width: 90px; }}
table.log td {{ font-family: var(--sans); }}
pre.patch {{ overflow-x: auto; font: 12px/1.45 var(--mono); background: var(--surface); border: 1px solid var(--line); padding: 10px; margin: 6px 0 0; }}
h4 {{ font-size: .95rem; margin: 14px 0 0; }}
html {{ scroll-behavior: smooth; }}
</style>

<main>
<header>
  <div class="eyebrow">DART · Panda → xArm7 · one-shot 데모 쌍 · 2026-10-08</div>
  <h1><span class="panda">Panda</span> source와 MimicGen 방식 <span class="xarm">xArm7</span> target (회전 보정 제한)</h1>
  <p>LIBERO libero_spatial task 0 “{html.escape(TASK)}”. source는 원본 데모(demo_0, 98스텝) 재생이고,
  target은 DART 논문(부록 C.3)이 쓴 MimicGen 방식으로 만들었다. xArm7이 Panda 데모의 목표 그리퍼 포즈 궤적을 따라가며 매 스텝 액션을 새로 계산한다.
  회전 액션은 학습 정규화 범위(q01~q99) 안으로 제한했다. 두 데모 모두 태스크에 성공했다.</p>
</header>
{TOC}

<div class="facts">
  <div class="fact"><b>98 / 98</b><span>프레임 수 (Panda / xArm7), fps 10</span></div>
  <div class="fact"><b>{s['panda_first_success']} / {ret['first_success']}</b><span>첫 성공 판정 스텝, 마지막 스텝도 성공</span></div>
  <div class="fact"><b>{ret['eef_diff_cm_mean']:.1f} / {ret['eef_diff_cm_max']:.1f} cm</b><span>Panda 궤적과의 그리퍼 위치 차이 평균 / 최대</span></div>
  <div class="fact"><b>{f(ret['action_absdiff_mean'][0], 2)} · {f(ret['action_absdiff_mean'][2], 2)} · {f(ret['action_absdiff_mean'][4], 2)}</b><span>source 대비 평균 액션 차이 Δx · Δz · Δpitch</span></div>
  <div class="fact"><b>{max(ret['out_pct'][3:6]) * 100:.0f}%</b><span>회전 액션의 학습 범위 밖 비율 (제한 전 Δpitch {raw['out_pct'][4] * 100:.0f}%)</span></div>
</div>

<section id="demo">
  <h2>데모 영상</h2>
  <video controls muted loop playsinline preload="metadata" src="data:video/mp4;base64,{b64('demo_pair.mp4')}"></video>
  <p class="note">왼쪽 Panda(source), 오른쪽 xArm7(target, MimicGen 방식 + 회전 보정 제한). 위는 agentview, 아래는 손목 카메라. 데이터셋에 저장된 이미지 그대로(180° 회전된 모델 입력 방향), 10fps. 성공 판정 시점부터 라벨이 초록색으로 바뀐다.</p>
</section>

<section>
  <h2>프레임 샘플</h2>
  <img alt="프레임 0, 25, 50, 75, 97에서 Panda와 xArm7의 agentview와 손목 이미지" src="data:image/png;base64,{b64('frames_grid.png')}">
</section>

<section>
  <h2>점검 결과</h2>
  <ul class="findings">
    <li><b>만든 방법: MimicGen 규칙 그대로.</b> 목표 포즈는 Panda 데모의 명령 목표 포즈(현재 포즈 + 액션×0.05m/0.5rad),
      액션은 (목표 − xArm7 현재 포즈)를 0.05m·0.5rad로 나눠 ±1로 자른 값, 그리퍼 명령은 원본 복사, 성공한 궤적만 채택.
      시작 장면이 같아 물체 기준 변환은 생략했고, MimicGen 기본 액션 노이즈(0.05)는 넣지 않았다.
      그리퍼 좌표계 차이(Panda grip_site가 손목 대비 90° 회전)는 xArm7에 Panda 손목 목표 자세를 주는 것으로 보정했다.
      여기에 회전 액션(Δroll·Δpitch·Δyaw)만 pi05_libero 정규화 통계의 q01~q99로 잘랐다 (MimicGen 원래 규칙에는 없는 단계).</li>
    <li><b>액션 스케일: 회전은 범위 안, Δx·Δz는 경계를 살짝 넘음.</b> FT 정규화는 pi05_libero 체크포인트의 Panda LIBERO 통계(q01/q99)를 그대로 쓴다.
      회전 3차원은 제한 덕분에 모두 ±1 안이다 (제한 없이는 Δpitch 정규화 최대 {f(raw['norm_max'][4], 2)}, 프레임 {raw['out_pct'][4] * 100:.0f}%가 범위 밖).
      Δx·Δz는 프레임 {ret['out_pct'][0] * 100:.0f}%·{ret['out_pct'][2] * 100:.0f}%가 정규화 1.07인데, 액션 상한 ±1이 q99(0.94)보다 조금 큰 탓이다. 원본 Panda 데모도 Δx가 −1.23까지 간다.</li>
    <li><b>제한이 필요했던 이유: xArm7 손목이 기울고, 그 보정이 회전 액션으로 나옴.</b> x 방향으로 움직일 때 손목이 ±2~5° 앞뒤로 기울고,
      그릇을 내려놓는 구간(프레임 72~88)에서는 최대 13°까지 기운다. 제한 없는 MimicGen 방식은 이걸 매 스텝 되돌리려 해서 Δpitch가 최대 0.44가 된다.
      제한한 뒤에는 기울기를 덜 되돌리는 만큼 회전 추종 오차가 조금 더 남는다.
      추종 오차는 위치 {rs['track_err_pos_cm']['mean']:.1f}/{rs['track_err_pos_cm']['max']:.1f} cm, 회전 {rs['track_err_rot_deg']['mean']:.1f}/{rs['track_err_rot_deg']['max']:.1f}°로
      Panda 자신(위치 {PANDA_TRACK['pos_mean']}/{PANDA_TRACK['pos_max']} cm, 회전 {PANDA_TRACK['rot_mean']}/{PANDA_TRACK['rot_max']}°)보다 크다.
      빠른 구간 {ret['clipped_steps']}스텝에서는 계산된 액션이 ±1을 넘어 잘렸다 (원본 액션이 이미 0.94 근처).</li>
    <li><b>논문과의 차이.</b> 논문은 norm stats를 UR5e가 포함된 사전학습 데이터로 계산해 이런 보정값이 범위 안이었다. 우리는 Panda 통계라 범위 밖이 된다.
      그래서 회전 보정만 통계 범위로 제한한 변형을 채택했다. 위치 추종과 성공 여부는 제한 없는 버전과 거의 같다 (아래 비교표).</li>
    <li><b>state는 이 설정에서 모델 입력이 아님.</b> pi05_libero·pi05_libero_oneshotft는 <code>discrete_state_input=False</code>이고 pi05에는 state 투영층이 없어서
      state(그리퍼 환산값 포함)는 학습·추론에 쓰이지 않는다. 표는 참고용.</li>
    <li><b>source는 공식 DART 데이터를 재현.</b> 공개 데이터셋의 같은 태스크 에피소드와 state 차이 최대 {OFFICIAL['state_max']}, 액션 차이 {OFFICIAL['actions_max']},
      이미지 평균 {OFFICIAL['img_mean']}/255 차이 (공식은 GPU, 여기는 CPU 렌더러).</li>
  </ul>
</section>

<section>
  <h2>target 만드는 방식 비교</h2>
  <div class="tbl"><table>
    <thead><tr><th>방식</th><th>첫 성공</th><th>Panda와 위치차 평균/최대 cm</th><th>회전 추종오차 평균/최대 °</th><th>액션차 Δx/Δz/Δpitch</th><th>Δpitch 정규화 최대</th><th>범위 밖 Δx/Δz/Δroll/Δpitch</th></tr></thead>
    <tbody>{variant_rows()}</tbody>
  </table></div>
  <p class="note">세 버전 모두 데이터셋으로 보관한다. 액션 재생은 source 액션을 그대로 넣은 버전, 제한 없음은 MimicGen 원래 규칙 그대로인 버전이다.</p>
</section>

<section>
  <h2>액션 통계</h2>
  <div class="tbl"><table>
    <thead><tr><th>차원</th><th class="panda">평균 Panda</th><th class="xarm">평균 xArm7</th><th class="panda">범위 Panda</th><th class="xarm">범위 xArm7</th><th>학습 통계 q01 … q99</th><th class="panda">정규화 Panda</th><th class="xarm">정규화 xArm7</th><th>범위 밖 P / X</th></tr></thead>
    <tbody>{action_rows()}</tbody>
  </table></div>
  <p class="note">정규화 = openpi 분위수 정규화 (x − q01)/(q99 − q01)·2 − 1. ±1을 1% 넘으면 범위 밖으로 셌다 (그리퍼 ±1이 q99 = 0.9996 경계에 걸리는 것은 제외).</p>
</section>

<section>
  <h2>state 통계 (참고)</h2>
  <div class="tbl"><table>
    <thead><tr><th>차원</th><th class="panda">평균 Panda</th><th class="xarm">평균 xArm7</th><th class="panda">범위 Panda</th><th class="xarm">범위 xArm7</th><th class="panda">정규화 Panda</th><th class="xarm">정규화 xArm7</th></tr></thead>
    <tbody>{state_rows()}</tbody>
  </table></div>
  <p class="note">state = 그리퍼 위치(x, y, z, m) + 자세 axis-angle(rx, ry, rz, rad) + 그리퍼(g1, g2, m). xArm7의 6관절 그리퍼는 패드 안쪽 면 간격/2 + 0.0004로 Panda 형식으로 환산. 이 설정에서 모델은 state를 읽지 않는다.</p>
</section>

<section>
  <h2>그리퍼 타이밍 (패드 간격, cm)</h2>
  <div class="tbl"><table><tbody>{grip_rows()}</tbody></table></div>
  <p class="note">닫기 명령은 프레임 {ci}, 열기 명령은 프레임 {oi}. 그리퍼 명령은 두 데모가 같다. 동역학은 Panda 규약에 맞춘 설정(관절 마찰 0, 그리퍼 speed 0.01).</p>
</section>

<section>
  <h2>데이터셋 위치</h2>
  <div class="paths">
    <div><span class="panda">source</span> &nbsp;<code>~/hayeon/dart_xarm7/data/lerobot/libero_spatial_t0_demo_0_src_panda</code></div>
    <div><span class="xarm">target (채택: MimicGen + 회전 보정 제한)</span> &nbsp;<code>~/hayeon/dart_xarm7/data/lerobot/libero_spatial_t0_demo_0_tgt_xarm7_retarget_rotclip</code></div>
    <div><span class="muted">target (MimicGen 제한 없음)</span> &nbsp;<code>~/hayeon/dart_xarm7/data/lerobot/libero_spatial_t0_demo_0_tgt_xarm7_retarget</code></div>
    <div><span class="muted">target (액션 재생)</span> &nbsp;<code>~/hayeon/dart_xarm7/data/lerobot/libero_spatial_t0_demo_0_tgt_xarm7_replay</code></div>
  </div>
  <p class="note">LeRobot v2.1, 에피소드 1개씩. 키 image / wrist_image (256×256), state (8), actions (7). pi05_libero_oneshotft 데이터 파이프라인으로 읽히는 것을 확인했다.
  학습할 때는 <code>DATA_REPO_ID=&lt;위 절대경로&gt;</code>.</p>
</section>
{CHART}
{EVAL_HTML}
{VIDEO_HTML}
{LOG_HTML}
</main>
"""
(R / "demo_pair_report.html").write_text(page)
print(R / "demo_pair_report.html", f"{len(page) / 1e6:.2f} MB")
