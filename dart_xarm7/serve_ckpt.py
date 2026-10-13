"""openpi 체크포인트 하나를 websocket 정책 서버로 띄운다 (공식 scripts/serve_policy.py 와 같은 create_trained_policy 경로).

Docker 없이 같은 컨테이너에서 LIBERO 클라이언트와 같이 쓰기 위한 것. 서빙 전에 JIT 웜업을 해서
첫 추론(~40초) 동안 클라이언트 keepalive 가 끊기지 않게 한다 (~/hayeon/dart_exp2/serve_masked.py 와 같은 방식).

  cd $DART_REPO && .venv/bin/python ~/hayeon/dart_xarm7/serve_ckpt.py --config pi05_libero --dir <ckpt> --port 9100
"""
import dataclasses
import logging
import socket
import time

import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config: str
    dir: str
    port: int = 9100


def main(args: Args) -> None:
    policy = _policy_config.create_trained_policy(_config.get_config(args.config), args.dir)
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
    logging.info("Serving config=%s dir=%s on port %d (host %s)", args.config, args.dir, args.port, socket.gethostname())
    websocket_policy_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
