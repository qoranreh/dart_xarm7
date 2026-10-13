"""학습 체크포인트 step 폴더의 params(fp32)를 bf16 으로 바꿔 같은 자리에 저장한다.

DART(domain_arithmetic.utils.load_policy_params)와 서빙(create_trained_policy)은 params 를
restore_params(dtype=bfloat16) 로 읽으므로, 같은 호출로 변환해 두면 결과가 비트 단위로 같다 (2026-10-07 θ0 로 검증).
디스크를 아끼려고 bf16 을 메모리에 올린 뒤 fp32 폴더를 먼저 지우고 저장한다. 저장 후 다시 읽어 비트 단위로 비교.

  cd $DART_REPO && .venv/bin/python ~/hayeon/dart_xarm7/convert_ckpt_bf16.py <ckpt>/pi05_libero_oneshotft/original/999
  시험: ... convert_ckpt_bf16.py <θ0 체크포인트 폴더> --test-out <임시 폴더>   (원본은 건드리지 않음)
"""
import argparse
import datetime
import os
import pathlib
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.models import model as _model

p = argparse.ArgumentParser()
p.add_argument("step_dir")
p.add_argument("--test-out", default=None, help="지정하면 원본을 지우지 않고 이 폴더에 bf16 params 를 쓴 뒤 검증만 함")
args = p.parse_args()

step = pathlib.Path(args.step_dir).resolve()
src = step / "params"
dst = (pathlib.Path(args.test_out).resolve() / "params") if args.test_out else src
du = lambda d: sum(f.stat().st_size for f in pathlib.Path(d).rglob("*") if f.is_file()) / 2**30

meta = ocp.PyTreeCheckpointer().metadata(src)
dtypes = {str(x.dtype) for x in jax.tree_util.tree_leaves(meta["params"]) if hasattr(x, "dtype")}
print(f"[bf16] {src}  저장 dtype={sorted(dtypes)}  크기 {du(src):.2f} GiB")
if dtypes == {"bfloat16"}:
    print("[bf16] 이미 bf16 — 건너뜀")
    raise SystemExit(0)

A = _model.restore_params(src, dtype=jnp.bfloat16)  # DART·서빙과 같은 로딩 (GPU 메모리에 bf16 으로)
jax.block_until_ready(A)
fp32_gib = du(src)
if not args.test_out:
    shutil.rmtree(src)  # 디스크 확보: A 가 메모리에 있음
try:
    ocp.PyTreeCheckpointer().save(dst, {"params": A})
except Exception:
    fallback = pathlib.Path("/dev/shm") / f"rescue_{step.parent.parent.parent.name}_params_bf16"
    print(f"[bf16] 저장 실패 → 메모리의 bf16 을 {fallback} 에 비상 저장 시도")
    ocp.PyTreeCheckpointer().save(fallback, {"params": A})
    raise

B = _model.restore_params(dst, dtype=jnp.bfloat16)
la, lb = jax.tree_util.tree_leaves(A), jax.tree_util.tree_leaves(B)
assert len(la) == len(lb)
diff = sum(int(not np.array_equal(np.asarray(x).view(np.uint16), np.asarray(y).view(np.uint16))) for x, y in zip(la, lb))
n = sum(x.size for x in la)
print(f"[bf16] {dst}  {fp32_gib:.2f} → {du(dst):.2f} GiB, 배열 {len(la)}개 / 값 {n / 1e9:.2f}B, 다른 배열 {diff}개")
assert diff == 0, "bf16 저장본이 메모리 값과 다름"
if not args.test_out:
    (step / "params_dtype.txt").write_text(
        f"params: bfloat16 (fp32 학습 결과를 {datetime.datetime.now():%Y-%m-%d %H:%M} 에 변환, convert_ckpt_bf16.py).\n"
        "DART/서빙은 dtype=bfloat16 으로 읽으므로 결과 동일. train_state 는 저장하지 않음 (OPENPI_SAVE_TRAIN_STATE=0).\n")
print("[bf16] OK")
