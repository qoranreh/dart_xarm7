"""serve_masked.py 의 modules=none / all 이 serve_ckpt.py(create_trained_policy) 와 같은 정책인지 확인한다.

두 경로로 정책을 만들고 (같은 config, 같은 웜업 3회 → openpi Policy 의 rng 가 key(0) 에서 같은 횟수만큼 진행),
고정된 임의 관측 N 개에 대한 행동이 비트 단위로 같은지 본다. 같으면 파라미터 키 구조·norm_stats·config 가 모두 같다는 뜻.
serve_masked.main 은 그대로 호출하고, 마지막의 websocket 서버만 가짜로 바꿔 정책 객체를 받아온다 (serve_masked.py 무수정).

  cd $DART_REPO && .venv/bin/python ~/hayeon/dart_xarm7/check_masked_equiv.py --modules none --base-dir <θ0> --tuned-dir <θ*>
  기준 정책: none → base_dir, all → tuned_dir 를 create_trained_policy 로 로드
"""
import argparse
import os
import sys

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

sys.path.insert(0, os.path.expanduser("~/hayeon/dart_exp2"))
import serve_masked as SM  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--modules", choices=["none", "all"], required=True)
p.add_argument("--base-dir", required=True)
p.add_argument("--tuned-dir", required=True)
p.add_argument("--config", default="pi05_libero")
p.add_argument("--n", type=int, default=4)
a = p.parse_args()

DUMMY = {"observation/image": np.zeros((224, 224, 3), np.uint8), "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
         "observation/state": np.zeros(8, np.float32), "prompt": "warmup"}
rs = np.random.default_rng(0)
OBS = [{"observation/image": rs.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rs.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/state": rs.normal(size=8).astype(np.float32),
        "prompt": "pick up the black bowl between the plate and the ramekin and place it on the plate"} for _ in range(a.n)]

# 1) 기준: serve_ckpt.py 와 같은 경로 (serve_masked.main 이 restore_params 를 바꿔치기하기 전에 만든다)
ref_dir = a.base_dir if a.modules == "none" else a.tuned_dir
ref = _policy_config.create_trained_policy(_config.get_config(a.config), ref_dir)
for _ in range(3):
    ref.infer(DUMMY)
ref_out = [np.asarray(ref.infer(o)["actions"]) for o in OBS]
del ref

# 2) serve_masked.main 그대로 (웜업 3회 포함), 서버만 가짜
captured = {}


class _FakeServer:
    def __init__(self, policy, **kw):
        captured["policy"] = policy

    def serve_forever(self):
        pass


SM.websocket_policy_server.WebsocketPolicyServer = _FakeServer
SM.main(SM.Args(base_dir=a.base_dir, tuned_dir=a.tuned_dir, modules=a.modules, config=a.config, port=0))
masked_out = [np.asarray(captured["policy"].infer(o)["actions"]) for o in OBS]

ok = True
for i, (x, y) in enumerate(zip(ref_out, masked_out)):
    same = x.shape == y.shape and np.array_equal(x, y)
    ok &= same
    print(f"obs {i}: shape {x.shape} 비트 동일 {same}, max|Δ| {np.abs(x - y).max():.3g}, |a| 평균 {np.abs(x).mean():.3g}")
print(f"modules={a.modules}: serve_masked == create_trained_policy({os.path.basename(ref_dir)}) → {'일치' if ok else '불일치'}")
sys.exit(0 if ok else 1)
