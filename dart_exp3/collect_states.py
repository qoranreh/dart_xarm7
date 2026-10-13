"""실험 3-2 데이터 수집: θ₀ 가 original 시점으로 libero_spatial 을 수행하는 동안, 같은 시뮬레이터 상태를
original / medium / large 세 시점 + 손목 카메라로 동시에 찍어 둔다 (δ 학습에 안 쓰인 장면).

평가 클라이언트(dart_exp2/run_eval.py)와 같은 전처리: 180도 회전 → 224 resize_with_pad → uint8.
렌더링 카메라만 4개로 늘림. num_steps_wait 이후 snapshot_every 스텝마다 저장.
"""

import collections
import dataclasses
import logging
import math
import pathlib

import numpy as np
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
VIEWS = {"original": "agentview", "medium": "agentview_medium", "large": "agentview_large"}


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 9013
    task_suite_name: str = "libero_spatial"
    task_ids: tuple[int, ...] = (0,)
    episodes: tuple[int, ...] = (0, 1, 2, 3, 4)  # init state index
    snapshot_every: int = 10
    max_steps: int = 220
    num_steps_wait: int = 10
    replan_steps: int = 5
    seed: int = 7
    out: str = "states.npz"


def prep(img):
    img = np.ascontiguousarray(img[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main(args: Args):
    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    snaps = collections.defaultdict(list)
    for task_id in args.task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                 camera_names=[*VIEWS.values(), "robot0_eye_in_hand"])
        env.seed(args.seed)
        for ep in args.episodes:
            env.reset()
            obs = env.set_init_state(init_states[ep])
            plan = collections.deque()
            done = False
            for t in range(args.max_steps + args.num_steps_wait):
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    continue
                imgs = {v: prep(obs[f"{cam}_image"]) for v, cam in VIEWS.items()}
                wrist = prep(obs["robot0_eye_in_hand_image"])
                state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
                if (t - args.num_steps_wait) % args.snapshot_every == 0:
                    for v in VIEWS:
                        snaps[v].append(imgs[v])
                    snaps["wrist"].append(wrist)
                    snaps["state"].append(state.astype(np.float32))
                    snaps["task_id"].append(task_id)
                    snaps["episode"].append(ep)
                    snaps["t"].append(t)
                    snaps["prompt"].append(str(task.language))
                if not plan:
                    chunk = client.infer({"observation/image": imgs["original"], "observation/wrist_image": wrist,
                                          "observation/state": state, "prompt": str(task.language)})["actions"]
                    plan.extend(chunk[: args.replan_steps])
                obs, _, done, _ = env.step(plan.popleft().tolist())
                if done:
                    break
            snaps["ep_success"] += [bool(done)] * (len(snaps["t"]) - len(snaps["ep_success"]))
            logging.info("task %d ep %d success=%s snapshots=%d", task_id, ep, done, len(snaps["t"]))
        env.close()
    np.savez(args.out, **{k: np.asarray(v) for k, v in snaps.items()})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
