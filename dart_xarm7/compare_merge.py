"""두 DART 출력(θ*)을 원소 단위로 비교한다 (예: CPU 합성 vs GPU 합성). 기준 θ0 대비 δ 의 차이도 본다.
  cd $DART_REPO && JAX_PLATFORMS=cpu .venv/bin/python ~/hayeon/dart_xarm7/compare_merge.py <θ*_A> <θ*_B> <θ0>
"""
import sys

import jax
import ml_dtypes
import numpy as np
from flax.traverse_util import flatten_dict

from openpi.models import model as M

a, b, base = (flatten_dict(M.restore_params(f"{p}/params", restore_type=np.ndarray, dtype=ml_dtypes.bfloat16)) for p in sys.argv[1:4])
n = same = 0
worst = []
for k in a:
    x, y, w = a[k], b[k], base[k]
    eq = x.view(np.uint16) == y.view(np.uint16)
    n += x.size
    same += int(eq.sum())
    dx, dy = x.astype(np.float32) - w.astype(np.float32), y.astype(np.float32) - w.astype(np.float32)
    rel = float(np.linalg.norm(dx - dy) / (np.linalg.norm(dx) + 1e-12))
    ulps = int(np.abs(x.view(np.int16).astype(np.int32) - y.view(np.int16).astype(np.int32)).max())
    worst.append((rel, "/".join(map(str, k)), x.size, float(eq.mean()), ulps, float(np.linalg.norm(dx))))
worst.sort(reverse=True)
print(f"전체 원소 {n / 1e9:.2f}B 중 비트 단위 동일 {100 * same / n:.4f}%")
print("δ 상대차 ‖δ_A−δ_B‖/‖δ_A‖ 상위 5개 키:")
for rel, k, sz, eqf, ulps, nd in worst[:5]:
    print(f"  {rel:.2e}  동일 {100 * eqf:.2f}%  최대 {ulps} ulp  ‖δ‖={nd:.3g}  {k} ({sz / 1e6:.1f}M)")
tot = np.sqrt(sum((r * d) ** 2 for r, _, _, _, _, d in worst)) / np.sqrt(sum(d ** 2 for *_, d in worst))
print(f"전체 δ 상대차: {tot:.3e}")
