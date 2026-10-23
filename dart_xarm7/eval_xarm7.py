"""LIBERO(-view) 평가 클라이언트 — Panda 또는 xArm7. ~/hayeon/dart_exp2/run_eval.py (공식 examples/libero-view/main.py 수정본) 기반.

공식 평가 프로토콜과 같은 것: 태스크별 고정 초기 상태 50개 중 앞 N개, env.seed(7), 대기 10스텝, 5스텝마다 재계획,
이미지 180° 회전 + 224 resize_with_pad, max_steps(libero_spatial 220), done=True 면 성공.
바뀐 것:
  1. --robot XArm7: 같은 Panda 용 초기 상태에서 물체 상태만 옮기고 팔은 Panda 그리퍼 포즈에 IK (libero_xarm7.transfer_init_state)
     state 의 그리퍼는 Panda 형식으로 환산 (pi05_libero 는 state 를 모델 입력으로 쓰지 않음)
  2. 카메라 2개만 렌더링 (OSMesa CPU 속도), 에피소드 단위로 --shard/--nshard 분할해 여러 프로세스 병렬
  3. 에피소드마다 JSONL 기록, 영상은 태스크별 episode 0 만 저장

  source ~/hayeon/dart_xarm7/env.sh && source ~/hayeon/dart_xarm7/client_env.sh
  $LIBERO_PY ~/hayeon/dart_xarm7/eval_xarm7.py --robot XArm7 --port 9100 --task-ids 0 1 2 --trials 10 --shard 0 --nshard 5 ...
"""
from __future__ import annotations

import collections
import dataclasses
import json
import logging
import os
import pathlib
import time

import imageio
import numpy as np
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

import libero_xarm7 as X
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 9100
    robot: str = "XArm7"
    task_suite_name: str = "libero_spatial"
    task_ids: tuple[int, ...] = (0,)
    trials: int = 10
    shard: int = 0
    nshard: int = 1
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    seed: int = 7
    tag: str = ""
    results_path: str = "results.jsonl"
    video_out_path: str = "videos"


def make_env(task, robot, res, seed):
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), robots=[robot], camera_heights=res, camera_widths=res,
                             camera_names=["agentview", "robot0_eye_in_hand"])
    env.seed(seed)  # 공식 평가와 동일
    return env


def main(args: Args) -> None:
    # 실험 6: 이 클라이언트에 실제로 적용된 조건·체크포인트를 첫 줄에 남긴다 (로그만, 동작 무관). EVAL_CKPT_INFO 는 러너가 export.
    logging.info("eval env: XARM7_COLOR=%s XARM7_DYNAMICS=%s robot=%s ckpt=%s tag=%s", getattr(X, "COLOR", "white(분기 없음)"),
                 X.DYNAMICS, args.robot, os.environ.get("EVAL_CKPT_INFO", "(미지정)"), args.tag)
    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = MAX_STEPS[args.task_suite_name]
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    jobs = [(t, e) for t in args.task_ids for e in range(args.trials)]
    jobs = jobs[args.shard::args.nshard]
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    out = open(args.results_path, "a")
    envs = {}
    for task_id, ep in jobs:
        task = suite.get_task(task_id)
        if task_id not in envs:
            for e in envs.values():
                for x in e:
                    if x is not None:  # Panda 모드는 보조 env 가 None
                        x.close()
            envs = {task_id: (make_env(task, args.robot, 256, args.seed),
                              make_env(task, "Panda", 64, args.seed) if args.robot != "Panda" else None)}
        env, panda = envs[task_id]
        init = suite.get_task_init_states(task_id)[ep]
        t_start = time.time()
        if args.robot == "Panda":
            env.reset()
            obs = env.set_init_state(init)
        else:
            flat, tinfo = X.transfer_init_state(panda, init, env)  # 내부에서 env.reset()
            obs = env.set_init_state(flat)
        action_plan = collections.deque()
        t, done, frames, err = 0, False, [], None
        while t < max_steps + args.num_steps_wait:
            try:
                if t < args.num_steps_wait:
                    obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size))
                if ep == 0:
                    frames.append(np.concatenate([img, wrist], axis=1))
                if not action_plan:
                    element = {"observation/image": img, "observation/wrist_image": wrist,
                               "observation/state": X.policy_state(env, obs), "prompt": str(task.language)}
                    chunk = client.infer(element)["actions"]
                    assert len(chunk) >= args.replan_steps
                    action_plan.extend(chunk[: args.replan_steps])
                obs, _, done, _ = env.step(action_plan.popleft().tolist())
                if done:
                    break
                t += 1
            except Exception as e:  # noqa: BLE001
                err = repr(e)
                logging.error("exception: %s", err)
                break
        rec = {"tag": args.tag, "robot": args.robot, "task_id": task_id, "episode": ep, "task": task.language,
               "success": bool(done), "steps": t - args.num_steps_wait, "sec": round(time.time() - t_start, 1), "error": err}
        if args.robot != "Panda":
            rec["ik_pos_err"] = tinfo["ik_pos_err"]
            rec["xarm7_color"], rec["xarm7_dynamics"] = getattr(X, "COLOR", "white"), X.DYNAMICS  # 기록만
        out.write(json.dumps(rec) + "\n")
        out.flush()
        logging.info("%s task %d ep %d success=%s steps=%d %.0fs", args.tag, task_id, ep, done, rec["steps"], rec["sec"])
        if ep == 0 and frames:
            imageio.mimwrite(pathlib.Path(args.video_out_path) / f"{args.tag}_t{task_id}_ep0_{'success' if done else 'failure'}.mp4",
                             frames, fps=10)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")  # 시각: 연결 끊김 시점 진단용
    main(tyro.cli(Args))
