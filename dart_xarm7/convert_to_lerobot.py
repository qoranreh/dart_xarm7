"""make_demo_pair.py 의 녹화(npz)를 dart 저장소가 기대하는 LeRobot 데이터셋으로 변환한다.

스키마는 examples/libero-view/convert_libero_data_to_lerobot.py 와 같다 (공개 SNUMPR/dart-libero-viewpoints 와도 동일):
  image / wrist_image (256,256,3 image), state (8 float32), actions (7 float32), fps 10, task = 언어 지시문
태스크 문자열은 평가 프롬프트(task.language)와 같은 깨끗한 문장을 쓴다 (공개 DART 데이터는 끝에 " demo d" 가 붙어 있음).

  source ~/hayeon/dart_xarm7/env.sh && cd $DART_REPO
  uv run python ~/hayeon/dart_xarm7/convert_to_lerobot.py
학습: DATA_REPO_ID=<출력 경로> (절대경로면 HF_LEROBOT_HOME 과 무관하게 그 경로를 씀)
"""
import argparse
import os
import pathlib
import shutil

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

p = argparse.ArgumentParser()
p.add_argument("--demos", default=os.path.expandvars("$DART_ROOT/demos"))
p.add_argument("--tag", default="libero_spatial_t0_demo_0")
p.add_argument("--out", default=os.path.expanduser("~/hayeon/dart_xarm7/data/lerobot"))
p.add_argument("--pairs", default="Panda:src_panda,XArm7_retarget:tgt_xarm7_retarget",
               help="<npz 접미사>:<데이터셋 이름> 목록. XArm7 = 액션 재생판(tgt_xarm7_replay), XArm7_retarget = MimicGen 방식")
args = p.parse_args()

FEATURES = {
    "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}

for robot, name in (x.split(":") for x in args.pairs.split(",")):
    z = np.load(pathlib.Path(args.demos) / f"{args.tag}_{robot}.npz")
    repo_id = f"{args.tag}_{name}"
    root = pathlib.Path(args.out) / repo_id
    if root.exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(repo_id=repo_id, root=root, robot_type=robot.split("_")[0].lower(), fps=10, features=FEATURES,
                               image_writer_threads=4, image_writer_processes=0)
    task = str(z["task"])
    for t in range(len(z["actions"])):
        ds.add_frame({"image": z["image"][t], "wrist_image": z["wrist_image"][t], "state": z["state"][t].astype(np.float32),
                      "actions": z["actions"][t].astype(np.float32), "task": task})
    ds.save_episode()
    print(f"{robot}: {root}  frames={ds.meta.total_frames} episodes={ds.meta.total_episodes} task={task!r}")
