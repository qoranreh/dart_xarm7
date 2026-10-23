"""평가 영상(mp4, 프레임 = [agentview | 손목] 224×448)에서 프레임을 고르게 뽑아 한 장의 PNG 로 붙인다 — 색·로봇 확인, 실패 유형 판정용.

  $LIBERO_PY ~/hayeon/dart_xarm7/video_frames.py <in.mp4> <out.png> [--n 8] [--cols 4] [--wrist]
  기본은 agentview 만 (손목 포함은 --wrist). 각 칸 위에 프레임 번호를 적는다 (영상 fps 10 → 프레임 = 정책 스텝, 대기 10스텝 제외).
"""
import argparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument("inp")
p.add_argument("out")
p.add_argument("--n", type=int, default=8)
p.add_argument("--cols", type=int, default=4)
p.add_argument("--wrist", action="store_true")
a = p.parse_args()

frames = [f for f in imageio.get_reader(a.inp)]
idx = np.linspace(0, len(frames) - 1, a.n).round().astype(int)
tiles = []
for i in idx:
    f = frames[i] if a.wrist else frames[i][:, : frames[i].shape[1] // 2]
    im = Image.fromarray(f)
    ImageDraw.Draw(im).text((4, 2), f"{i}/{len(frames) - 1}", fill=(255, 255, 0))
    tiles.append(np.asarray(im))
rows = [np.concatenate(tiles[r * a.cols:(r + 1) * a.cols] + [np.zeros_like(tiles[0])] * max(0, (r + 1) * a.cols - len(tiles)), 1)
        for r in range((len(tiles) + a.cols - 1) // a.cols)]
Image.fromarray(np.concatenate(rows, 0)).save(a.out)
print(f"{a.inp}: {len(frames)} 프레임 → {a.out} ({len(idx)}칸)")
