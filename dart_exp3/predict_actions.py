"""실험 3-2: 수집한 장면에서 각 모델·시점의 action chunk 예측 (서버와 같은 transform, 고정 노이즈).

기준 = θ₀ 가 original 시점을 볼 때의 action. 시점이 바뀌었을 때 각 모델의 action 이 기준에서 얼마나 벗어나는지 본다.
'floor' = θ₀·original 을 다른 노이즈로 한 번 더 → flow matching 샘플링 자체의 흔들림.
"""

import dataclasses
import logging
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from extract import build_params, load_flat  # noqa: E402

W = pathlib.Path("/var/tmp/pbs.128706.ECE-util1/dart")


@dataclasses.dataclass
class Args:
    states_dir: str = str(W / "exp3_states")
    ckpt_dir: str = str(W / "ckpts")
    out: str = str(pathlib.Path(__file__).parent / "results/actions.npz")
    batch_size: int = 32
    runs: tuple[str, ...] = (
        "base:original:0", "base:original:1", "base:medium:0", "base:large:0",
        "medium-all:medium:0", "medium-vision:medium:0", "medium-vlm:medium:0",
        "large-all:large:0", "large-vision:large:0", "large-vlm:large:0",
    )  # 모델:시점:노이즈 시드


def main(args: Args):
    files = sorted(pathlib.Path(args.states_dir).glob("states_*.npz"))
    S = [dict(np.load(f)) for f in files]
    S = {k: np.concatenate([s[k] for s in S]) for k in S[0]}
    n = len(S["t"])
    logging.info("snapshots: %d from %d files", n, len(files))

    cfg = _config.get_config("pi05_libero")
    ah, ad = cfg.model.action_horizon, cfg.model.action_dim
    base = load_flat(pathlib.Path(args.ckpt_dir) / "pi05_libero")
    tuned = {v: load_flat(pathlib.Path(args.ckpt_dir) / f"dart-pi05-viewpoint-{v}") for v in ("medium", "large")}
    from serve_masked import MODULES  # noqa: E402

    results = {k: S[k] for k in ("task_id", "episode", "t", "ep_success")}
    current = None
    for run in args.runs:
        model_name, view, seed = run.split(":")
        if model_name != current:
            if model_name == "base":
                params = build_params(base, base, set())
            else:
                tv, mods = model_name.split("-")
                params = build_params(base, tuned[tv], set(MODULES) if mods == "all" else {mods})
            _model.restore_params = lambda *a, p=params, **kw: p
            policy = _policy_config.create_trained_policy(cfg, pathlib.Path(args.ckpt_dir) / "pi05_libero")
            current = model_name
        noise1 = np.asarray(jax.random.normal(jax.random.key(int(seed)), (1, ah, ad)))
        acts = []
        for s in range(0, n, args.batch_size):
            idx = list(range(s, min(s + args.batch_size, n)))
            idx_p = idx + [idx[-1]] * (args.batch_size - len(idx))
            items = [policy._input_transform({
                "observation/image": S[view][i], "observation/wrist_image": S["wrist"][i],
                "observation/state": S["state"][i], "prompt": str(S["prompt"][i]),
            }) for i in idx_p]
            batch = jax.tree.map(lambda *x: jnp.asarray(np.stack(x)), *items)
            obs = _model.Observation.from_dict(batch)
            noise = jnp.asarray(np.repeat(noise1, len(idx_p), axis=0))
            out = np.asarray(policy._sample_actions(jax.random.key(0), obs, noise=noise))
            for j in range(len(idx)):
                acts.append(policy._output_transform({"state": np.asarray(batch["state"][j]), "actions": out[j]})["actions"])
        results[run] = np.stack(acts)
        logging.info("%s: %s", run, results[run].shape)
    np.savez(args.out, **results)
    logging.info("saved %s", args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
