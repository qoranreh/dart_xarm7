"""DART 실험 3: 레이어별 linear CKA.

기준 = θ₀ 가 원래 시점(original)을 볼 때의 표현. 시점이 바뀌면(medium/large) θ₀ 표현이 기준에서 멀어지는데,
δ(전체 / vision만 / VLM만)를 넣은 모델이 그 거리를 얼마나 되돌리는지를 레이어마다 본다.

  A. CKA(θ₀·original, 모델·target)          — 원래 시점 표현과의 유사도 (높을수록 '원래처럼' 봄)
  B. CKA(θ₀·target, θ*·target)              — 같은 입력에서 δ 가 표현을 얼마나 바꾸는가
  C. CKA(θ₀·original, θ*·original)          — δ 가 원래 시점 표현을 망가뜨리는가
  회복률 = (A_모델 − A_θ₀) / (1 − A_θ₀)    — θ₀ 에서 잃은 유사도 중 되찾은 비율

global: 1,085 프레임 전체로 CKA. within: 에피소드(5개)별 평균을 빼고 CKA (장면 차이 대신 궤적 안의 변화만 봄).
"""

import os
import pathlib

import numpy as np

ACTS = pathlib.Path(os.environ.get("ACTS", "/var/tmp/pbs.128706.ECE-util1/dart/exp3_acts_v2"))
OUT = pathlib.Path(__file__).parent / "results"
STREAMS = ["vision", "llm_img", "llm_wrist", "llm_txt", "expert"]


def cka(x, y):
    x = x - x.mean(0)
    y = y - y.mean(0)
    hsic = np.linalg.norm(y.T @ x) ** 2
    return hsic / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y))


def load(name):
    return dict(np.load(ACTS / f"{name}.npz"))


def main():
    OUT.mkdir(exist_ok=True)
    episode = np.load(ACTS / "meta.npz")["episode"]
    cache = {}

    def get(name, stream, mode):
        if name not in cache:
            cache[name] = {k: v.astype(np.float64) for k, v in load(name).items()}
        a = cache[name][stream]
        if mode == "within":
            a = a.copy()
            for e in np.unique(episode):
                a[:, episode == e] -= a[:, episode == e].mean(1, keepdims=True)
        return a

    rows = []
    for mode in ("global", "within"):
        for view in ("medium", "large"):
            comps = {
                "A:base": ("base__original", f"base__{view}"),
                "A:all": ("base__original", f"{view}-all__{view}"),
                "A:vision": ("base__original", f"{view}-vision__{view}"),
                "A:vlm": ("base__original", f"{view}-vlm__{view}"),
                "B:base_vs_all@target": (f"base__{view}", f"{view}-all__{view}"),
                "C:base_vs_all@original": ("base__original", f"{view}-all__original"),
            }
            for comp, (a, b) in comps.items():
                for stream in STREAMS:
                    xa, xb = get(a, stream, mode), get(b, stream, mode)
                    for layer in range(xa.shape[0]):
                        rows.append((mode, view, comp, stream, layer, cka(xa[layer], xb[layer])))
        # 참고: θ₀ 의 시점 민감도 (small 포함)
        for view in ("small", "medium", "large"):
            for stream in STREAMS:
                xa, xb = get("base__original", stream, mode), get(f"base__{view}", stream, mode)
                for layer in range(xa.shape[0]):
                    rows.append((mode, view, "A:base_sens", stream, layer, cka(xa[layer], xb[layer])))

    with open(OUT / "cka_layers.csv", "w") as f:
        f.write("mode,view,comparison,stream,layer,cka\n")
        for r in rows:
            f.write(",".join(map(str, r[:-1])) + f",{r[-1]:.6f}\n")
    print(f"wrote {OUT / 'cka_layers.csv'} ({len(rows)} rows)")

    # 요약: 스트림별 레이어 평균과 마지막 레이어, 회복률
    import collections

    tab = collections.defaultdict(dict)
    for mode, view, comp, stream, layer, v in rows:
        tab[(mode, view, stream)].setdefault(comp, []).append(v)
    lines = ["| mode | view | stream | A θ₀ (mean / last) | A θ* | A vision만 | A VLM만 | 회복률 θ* / vision / VLM (last) | B θ₀↔θ* @target (last) | C θ₀↔θ* @original (last) |", "|" + "---|" * 10]
    for (mode, view, stream), d in sorted(tab.items()):
        if "A:all" not in d:
            continue
        f = lambda k: f"{np.mean(d[k]):.3f} / {d[k][-1]:.3f}"  # noqa: E731
        rec = lambda k: (d[k][-1] - d["A:base"][-1]) / (1 - d["A:base"][-1])  # noqa: E731
        lines.append(
            f"| {mode} | {view} | {stream} | {f('A:base')} | {f('A:all')} | {f('A:vision')} | {f('A:vlm')} | "
            f"{rec('A:all'):+.0%} / {rec('A:vision'):+.0%} / {rec('A:vlm'):+.0%} | "
            f"{d['B:base_vs_all@target'][-1]:.3f} | {d['C:base_vs_all@original'][-1]:.3f} |"
        )
    (OUT / "cka_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
