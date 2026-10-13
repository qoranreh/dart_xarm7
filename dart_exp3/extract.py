"""DART 실험 3: 레이어별 활성값 추출 (θ₀ / θ* / 모듈별 δ 모델 × 시점).

dart-libero-viewpoints 의 같은 프레임을 original/small/medium/large 시점으로 넣고,
모듈별 레이어 출력을 토큰 평균해서 (레이어, 프레임, 차원) 으로 저장한다.

  vision : SigLIP 27 블록 출력, 3인칭 이미지 패치 256개 평균              (27, N, 1152)
  llm_img / llm_wrist / llm_txt : Gemma 2B (VLM) 입력 + 18 레이어 출력,
           3인칭 이미지 / 손목 이미지 / 프롬프트 토큰 평균                 (19, N, 2048)
  expert : action expert 입력 + 18 레이어 출력, action 토큰 10개 평균   (19, N, 1024)
           (flow matching 첫 스텝: t=1, 고정 노이즈)

모델 구성은 serve_masked.py 와 같다 (모듈 단위로 θ* leaf 를 바꿔 끼우고 bf16).
Gemma 레이어는 scan 대신 레이어별 파라미터를 잘라 Block 을 하나씩 적용해서 중간 출력을 얻는다.
"""

import dataclasses
import io
import json
import logging
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
import tyro
from flax import traverse_util
from PIL import Image

import openpi.models.gemma as _gemma
import openpi.models.model as _model
import openpi.models.siglip as _siglip
from openpi.models.pi0 import make_attn_mask, posemb_sincos
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

sys.path.insert(0, str(pathlib.Path.home() / "hayeon/dart_exp2"))
from serve_masked import MODULES, module_of  # noqa: E402

W = pathlib.Path("/var/tmp/pbs.128706.ECE-util1/dart")
VIEWS = {"original": "image", "small": "small_image", "medium": "medium_image", "large": "large_image"}
N_IMG = 256  # SigLIP So400m/14 @224 → 16×16 패치


@dataclasses.dataclass
class Args:
    data_dir: str = str(W / "data/dart-libero-viewpoints")
    ckpt_dir: str = str(W / "ckpts")
    out_dir: str = str(W / "exp3_acts_v2")
    batch_size: int = 32
    # "모델:시점" 목록. 모델 = base | <view>-all | <view>-vision | <view>-vlm | <view>-expert
    runs: tuple[str, ...] = (
        "base:original", "base:small", "base:medium", "base:large",
        "medium-all:original", "medium-all:medium", "medium-vision:medium", "medium-vlm:medium",
        "large-all:original", "large-all:large", "large-vision:large", "large-vlm:large",
    )
    check: bool = True  # 첫 배치에서 수동 forward 와 openpi 모델 forward 가 같은지 확인


def load_frames(data_dir):
    rows = []
    for f in sorted(pathlib.Path(data_dir, "data/chunk-000").glob("episode_*.parquet")):
        rows += pq.read_table(f).to_pylist()
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(pathlib.Path(data_dir, "meta/tasks.jsonl"))}
    dec = lambda v: np.asarray(Image.open(io.BytesIO(v["bytes"])).convert("RGB"))  # noqa: E731
    frames = {
        "images": {view: np.stack([dec(r[col]) for r in rows]) for view, col in VIEWS.items()},
        "wrist": np.stack([dec(r["wrist_image"]) for r in rows]),
        "state": np.stack([np.asarray(r["state"], np.float32) for r in rows]),
        # 데이터셋 task 문자열 끝의 " demo d" 는 평가 프롬프트에 없으므로 뗀다
        "prompt": [tasks[r["task_index"]].removesuffix(" demo d") for r in rows],
        "episode": np.array([r["episode_index"] for r in rows]),
        "frame": np.array([r["frame_index"] for r in rows]),
    }
    return frames


def load_flat(path):
    return traverse_util.flatten_dict(_model.restore_params(pathlib.Path(path) / "params", restore_type=np.ndarray))


def build_params(base, tuned, selected):
    merged = {k: jnp.asarray(tuned[k] if module_of(k) in selected else v, dtype=jnp.bfloat16) for k, v in base.items()}
    return traverse_util.unflatten_dict(merged)


def make_forward(model_cfg):
    pg, ex = _gemma.get_config(model_cfg.paligemma_variant), _gemma.get_config(model_cfg.action_expert_variant)
    img_mod = _siglip.Module(num_classes=pg.width, variant="So400m/14", pool_type="none", scan=True, dtype_mm=model_cfg.dtype)
    llm_mod = _gemma.Module(configs=[pg, ex], embed_dtype=model_cfg.dtype, adarms=True)
    block = _gemma.Block(configs=(pg, ex))
    norm = _gemma.RMSNorm()
    depth, ah, ad = pg.depth, model_cfg.action_horizon, model_cfg.action_dim

    def linear(p, x):
        return x @ p["kernel"] + p["bias"]

    def forward(P, obs: _model.Observation, noise):
        b = noise.shape[0]
        llm = P["PaliGemma"]["llm"]
        # --- prefix: 이미지(SigLIP) + 프롬프트 (Pi0.embed_prefix 와 같은 순서) ---
        tokens, input_mask, vision = [], [], None
        for name in obs.images:
            x, out = img_mod.apply({"params": P["PaliGemma"]["img"]}, obs.images[name], train=False)
            if name == "base_0_rgb":
                n_blk = sum(k.startswith("block") for k in out["encoder"])
                vision = jnp.stack([out["encoder"][f"block{l:02d}"]["+mlp"].mean(1) for l in range(n_blk)])
            tokens.append(x)
            input_mask.append(jnp.repeat(obs.image_masks[name][:, None], x.shape[1], axis=1))
        txt = llm_mod.apply({"params": llm}, obs.tokenized_prompt, method="embed")
        tokens.append(txt)
        input_mask.append(obs.tokenized_prompt_mask)
        prefix = jnp.concatenate(tokens, axis=1)
        prefix_mask = jnp.concatenate(input_mask, axis=1)
        # --- suffix: action expert (Pi0.embed_suffix, pi05, t=1 = 순수 노이즈) ---
        t = jnp.ones((b,))
        act = linear(P["action_in_proj"], noise)
        temb = posemb_sincos(t, act.shape[-1], min_period=4e-3, max_period=4.0)
        temb = jax.nn.swish(linear(P["time_mlp_in"], temb))
        adarms = jax.nn.swish(linear(P["time_mlp_out"], temb))
        input_mask_all = jnp.concatenate([prefix_mask, jnp.ones((b, ah), bool)], axis=1)
        ar = jnp.array([False] * prefix.shape[1] + [True] + [False] * (ah - 1))
        mask = make_attn_mask(input_mask_all, ar)[:, None]
        positions = jnp.cumsum(input_mask_all, axis=1) - 1

        n_pre = prefix.shape[1]
        txt_mask = obs.tokenized_prompt_mask.astype(jnp.float32)

        def pools(xs):
            p, s = xs[0].astype(jnp.float32), xs[1].astype(jnp.float32)
            return (
                p[:, :N_IMG].mean(1),
                p[:, N_IMG : 2 * N_IMG].mean(1),
                (p[:, n_pre - txt.shape[1] :] * txt_mask[..., None]).sum(1) / txt_mask.sum(1, keepdims=True),
                s.mean(1),
            )

        xs = [prefix.astype(jnp.bfloat16), act.astype(jnp.bfloat16)]
        acts = [pools(xs)]
        for l in range(depth):
            lp = jax.tree.map(lambda a, l=l: a[l], llm["layers"])
            xs, _ = block.apply({"params": lp}, xs, None, positions, mask, [None, adarms], True)
            acts.append(pools(xs))
        llm_img, llm_wrist, llm_txt, expert = (jnp.stack(a) for a in zip(*acts))
        # 확인용: 최종 norm + action_out_proj → 첫 스텝 속도장 v_t
        suf, _ = norm.apply({"params": llm["final_norm_1"]}, xs[1], adarms)
        v_t = linear(P["action_out_proj"], suf[:, -ah:])
        return {"vision": vision, "llm_img": llm_img, "llm_wrist": llm_wrist, "llm_txt": llm_txt, "expert": expert}, v_t

    return jax.jit(forward)


def reference_vt(model, obs, noise):
    """openpi nnx 모델로 같은 입력의 v_t 를 계산 (compute_loss 와 같은 경로)."""
    b = noise.shape[0]
    pt, pm, par = model.embed_prefix(obs)
    st, sm, sar, ad = model.embed_suffix(obs, noise, jnp.ones((b,)))
    mask = make_attn_mask(jnp.concatenate([pm, sm], 1), jnp.concatenate([par, sar], 0))
    positions = jnp.cumsum(jnp.concatenate([pm, sm], 1), axis=1) - 1
    (_, so), _ = model.PaliGemma.llm([pt, st], mask=mask, positions=positions, adarms_cond=[None, ad])
    return model.action_out_proj(so[:, -model.action_horizon :])


def main(args: Args):
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frames = load_frames(args.data_dir)
    n = len(frames["prompt"])
    logging.info("frames: %d (%.0fs)", n, time.time() - t0)
    np.savez(out_dir / "meta.npz", episode=frames["episode"], frame=frames["frame"], prompt=np.array(frames["prompt"]))

    train_cfg = _config.get_config("pi05_libero")
    base = load_flat(pathlib.Path(args.ckpt_dir) / "pi05_libero")
    tuned = {v: load_flat(pathlib.Path(args.ckpt_dir) / f"dart-pi05-viewpoint-{v}") for v in ("medium", "large")}

    # 입력 transform(리사이즈·토크나이즈)은 서버와 똑같이 create_trained_policy 에서 가져온다
    params0 = build_params(base, base, set())
    _model.restore_params = lambda *a, **kw: params0
    policy = _policy_config.create_trained_policy(train_cfg, pathlib.Path(args.ckpt_dir) / "pi05_libero")

    input_transform = policy._input_transform

    def make_obs(view, idx):
        items = [
            input_transform({
                "observation/image": frames["images"][view][i],
                "observation/wrist_image": frames["wrist"][i],
                "observation/state": frames["state"][i],
                "prompt": frames["prompt"][i],
            })
            for i in idx
        ]
        batch = jax.tree.map(lambda *x: jnp.asarray(np.stack(x)), *items)
        return _model.preprocess_observation(None, _model.Observation.from_dict(batch), train=False)

    forward = make_forward(train_cfg.model)
    # 모든 프레임에 같은 노이즈 하나를 쓴다. 프레임마다 다르면 expert 표현의 프레임 간 차이가 관측이 아니라
    # 노이즈 차이에 지배된다 (v1 에서 expert CKA 가 전부 0.9 이상으로 나온 원인).
    noise1 = np.asarray(jax.random.normal(jax.random.key(0), (1, train_cfg.model.action_horizon, train_cfg.model.action_dim)))
    noise_all = np.repeat(noise1, n, axis=0)

    if args.check:
        idx = list(range(4))
        obs = make_obs("medium", idx)
        _, vt = forward(params0, obs, jnp.asarray(noise_all[idx]))
        vt_ref = reference_vt(policy._model, obs, jnp.asarray(noise_all[idx]))
        err = float(jnp.abs(vt - vt_ref).max() / (jnp.abs(vt_ref).max() + 1e-6))
        logging.info("check: manual forward vs openpi model, max rel err of v_t = %.2e", err)
        assert err < 5e-2, err
    del policy

    cache = {}
    for run in args.runs:
        model_name, view = run.split(":")
        path = out_dir / f"{model_name}__{view}.npz"
        if path.exists():
            logging.info("skip %s", run)
            continue
        if model_name not in cache:
            cache.clear()
            if model_name == "base":
                cache[model_name] = params0
            else:
                tv, mods = model_name.split("-")
                sel = set(MODULES) if mods == "all" else {mods}
                cache[model_name] = build_params(base, tuned[tv], sel)
        P = cache[model_name]
        t1 = time.time()
        chunks = []
        for s in range(0, n, args.batch_size):
            idx = list(range(s, min(s + args.batch_size, n)))
            pad = args.batch_size - len(idx)
            idx_p = idx + [idx[-1]] * pad  # 마지막 배치는 채워서 같은 shape 로 (재컴파일 방지)
            acts, _ = forward(P, make_obs(view, idx_p), jnp.asarray(noise_all[idx_p]))
            chunks.append({k: np.asarray(v[:, : len(idx)], np.float16) for k, v in acts.items()})
        out = {k: np.concatenate([c[k] for c in chunks], axis=1) for k in chunks[0]}
        np.savez(path, **out)
        logging.info("%s: %s (%.0fs)", run, {k: v.shape for k, v in out.items()}, time.time() - t1)
    logging.info("done (%.0fs)", time.time() - t0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
