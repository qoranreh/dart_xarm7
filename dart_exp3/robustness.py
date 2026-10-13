"""핵심 수치의 안정성: 에피소드 하나씩 빼고(leave-one-episode-out) CKA 회복률 재계산.
VLM 스트림은 17층(=expert 가 key/value 로 읽는 마지막 prefix 표현), expert 는 마지막 18층."""
import numpy as np, pathlib
from cka import ACTS, cka
ep = np.load(ACTS / "meta.npz")["episode"]
SEL = {"vision": 26, "llm_img": 17, "llm_wrist": 17, "llm_txt": 17, "expert": 18}
lines = ["| view | stream (layer) | θ₀ CKA | 회복률 θ* | 회복률 vision만 | 회복률 VLM만 | (에피소드 1개씩 뺀 5회 최소~최대) |", "|---|---|---|---|---|---|---|"]
for view in ("medium", "large"):
    names = ["base__original", f"base__{view}", f"{view}-all__{view}", f"{view}-vision__{view}", f"{view}-vlm__{view}"]
    for stream, L in SEL.items():
        X = {nm: np.load(ACTS / f"{nm}.npz")[stream][L].astype(np.float64) for nm in names}
        def rec(mask):
            ref = X["base__original"][mask]
            a0 = cka(ref, X[f"base__{view}"][mask])
            r = [(cka(ref, X[n][mask]) - a0) / (1 - a0) for n in names[2:]]
            return a0, r
        a0, full = rec(np.ones(len(ep), bool))
        jk = np.array([rec(ep != e)[1] for e in np.unique(ep)])
        cell = lambda i: f"{full[i]:+.0%} ({jk[:, i].min():+.0%}~{jk[:, i].max():+.0%})"
        lines.append(f"| {view} | {stream} ({L}) | {a0:.3f} | {cell(0)} | {cell(1)} | {cell(2)} |  |")
out = "\n".join(lines); print(out)
pathlib.Path("results/robustness.md").write_text(out + "\n")
