"""θ0(fp32) 를 bf16 체크포인트로 바꿔 저장한 뒤, DART 로더(load_policy_params)로 두 쪽을 읽어 비트 단위로 비교한다."""
import os
import shutil
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from domain_arithmetic.utils import Checkpoint, flatten_dict, load_policy_params, save_params
from openpi.models import model as _model

src = Path(sys.argv[1])  # fp32 체크포인트 디렉터리 (params/, assets/ 포함)
dst = Path(sys.argv[2])  # bf16 변환본을 쓸 디렉터리

# 1) 변환: DART/평가 서버와 똑같은 호출(restore_params(dtype=bf16))로 읽고, DART 의 save_params 로 저장
t = time.time()
p = _model.restore_params(src / "params", dtype=jnp.bfloat16)
save_params(p, dst)
shutil.copytree(src / "assets", dst / "assets")
del p
print(f"[convert] {time.time() - t:.0f}s")
os.system(f"du -sh {src}/params {dst}/params")

# 2) DART 로더로 양쪽을 읽어 비교
a = flatten_dict(load_policy_params(Checkpoint(config="pi05_libero", dir=str(src))))
b = flatten_dict(load_policy_params(Checkpoint(config="pi05_libero", dir=str(dst))))
assert a.keys() == b.keys(), "키 불일치"
n_el = n_diff = 0
dtypes = set()
for k in a:
    x, y = np.asarray(a[k]), np.asarray(b[k])
    dtypes.add((str(x.dtype), str(y.dtype)))
    assert x.shape == y.shape, k
    same = x.view(np.uint8).tobytes() == y.view(np.uint8).tobytes()
    n_el += x.size
    if not same:
        n_diff += 1
        print("DIFF", k)
print(f"[compare] leaves={len(a)} elements={n_el / 1e9:.2f}B dtypes={dtypes} differing_leaves={n_diff}")
print("RESULT:", "BIT-IDENTICAL" if n_diff == 0 else "DIFFERENT")
