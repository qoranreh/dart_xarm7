"""DART 실험 2 정책 서버: θ = θ₀ + mask·δ (δ = θ* − θ₀) 를 메모리에서 합성해 openpi 서버로 띄운다.

mask 는 모듈 단위 0/1 이므로 "선택한 모듈은 θ* leaf, 나머지는 θ₀ leaf" 와 같다 (디스크에 저장하지 않음).
모듈 구분 (실험 1과 동일한 3분할):
  vision = PaliGemma/img/*
  vlm    = PaliGemma/llm/* 중 이름이 *_1 이 아닌 것 (Gemma 2B, 토큰 임베딩 포함)
  expert = PaliGemma/llm/* 중 *_1 (action expert 300M) + action_in/out_proj, time_mlp_in/out
"""

import dataclasses
import logging
import pathlib
import socket
import time

import jax.numpy as jnp
import numpy as np
import tyro
from flax import traverse_util

import openpi.models.model as _model
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

MODULES = ("vision", "vlm", "expert")


def module_of(path: tuple[str, ...]) -> str:
    if path[:2] == ("PaliGemma", "img"):
        return "vision"
    if path[:2] == ("PaliGemma", "llm"):
        return "expert" if any(p.endswith("_1") for p in path[2:]) else "vlm"
    return "expert"


@dataclasses.dataclass
class Args:
    base_dir: str  # θ₀ (pi05_libero)
    tuned_dir: str  # θ* (dart-pi05-viewpoint-*)
    modules: str = "all"  # "none" | "all" | 쉼표 목록 (vision,vlm,expert)
    config: str = "pi05_libero"
    port: int = 8000


def load_flat(ckpt_dir: str) -> dict:
    params = _model.restore_params(pathlib.Path(ckpt_dir) / "params", restore_type=np.ndarray)
    return traverse_util.flatten_dict(params)


def main(args: Args) -> None:
    if args.modules == "all":
        selected = set(MODULES)
    elif args.modules == "none":
        selected = set()
    else:
        selected = set(args.modules.split(","))
        assert selected <= set(MODULES), f"unknown modules: {selected - set(MODULES)}"

    base = load_flat(args.base_dir)
    tuned = load_flat(args.tuned_dir) if selected else {}
    if tuned:
        assert tuned.keys() == base.keys(), "θ₀ 와 θ* 의 키 구조가 다름"

    merged, counts = {}, {m: [0, 0] for m in MODULES}
    for k, w0 in base.items():
        m = module_of(k)
        use_tuned = m in selected
        counts[m][0 if use_tuned else 1] += w0.size
        # create_trained_policy 와 같은 bf16 로 올린다
        merged[k] = jnp.asarray(tuned[k] if use_tuned else w0, dtype=jnp.bfloat16)
    del base, tuned
    for m, (t, b) in counts.items():
        logging.info("module %-6s: θ* %.3fB params, θ₀ %.3fB params", m, t / 1e9, b / 1e9)
    params = traverse_util.unflatten_dict(merged)

    # create_trained_policy 의 transform/norm_stats 구성을 그대로 쓰고 파라미터만 바꿔 끼운다.
    # norm_stats 는 base_dir/assets 에서 읽음 (θ₀ 와 θ* 의 norm_stats.json 은 바이트 단위로 동일).
    _model.restore_params = lambda *a, **kw: params
    policy = _policy_config.create_trained_policy(_config.get_config(args.config), args.base_dir)

    # 첫 추론의 JIT 컴파일 동안 서버가 멈추면 클라이언트들이 keepalive 타임아웃으로 끊기므로 미리 컴파일한다.
    dummy = {
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/state": np.zeros(8, np.float32),
        "prompt": "warmup",
    }
    for i in range(3):
        t0 = time.time()
        policy.infer(dummy)
        logging.info("warmup infer %d: %.2fs", i, time.time() - t0)

    logging.info("Serving modules=%s on port %d (host %s)", args.modules, args.port, socket.gethostname())
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
