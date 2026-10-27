"""실험 7 섭동 확인 — eval_perturb.prepare_episode 를 그대로 써서 (평가와 같은 코드 경로)
(1) 조건별로 정책이 처음 보는 agentview·손목 화면 (대기 10스텝 뒤) 을 한 장으로 렌더하고,
(2) joint 섭동의 재추첨 횟수·손끝 이동량, 카메라 섭동의 장면 중심·거리를 태스크 0~9 × 초기 상태 범위에서 집계한다. 정책 서버 불필요.

  source ~/hayeon/dart_xarm7/env.sh && source ~/hayeon/dart_xarm7/client_env.sh
  $LIBERO_PY ~/hayeon/dart_xarm7/check_perturb.py --out ~/hayeon/dart_xarm7/renders/exp7 --init-offset 20 --trials 5
"""
import dataclasses
import json
import pathlib
import sys

import imageio
import numpy as np
import tyro
from PIL import Image, ImageDraw
from openpi_client import image_tools

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import eval_perturb as EP  # noqa: E402
from eval_xarm7 import LIBERO_DUMMY_ACTION, make_env  # noqa: E402
from libero.libero import benchmark  # noqa: E402

CONDS = [("none", 0.0), ("cam_dist", 1.2), ("cam_dist", 1.5), ("cam_azim", 15.0), ("cam_azim", 30.0), ("joint", 0.1), ("joint", 0.2)]


@dataclasses.dataclass
class Args:
    out: str
    render_tasks: tuple[int, ...] = (0, 4, 7)
    init_offset: int = 20
    trials: int = 5
    seed: int = 7


def first_view(env, obs):
    for _ in range(10):  # 평가와 같은 대기 10스텝
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    f = lambda k: image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs[k][::-1, ::-1]), 224, 224))
    return f("agentview_image"), f("robot0_eye_in_hand_image")


def main(a: Args) -> None:
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    stats, tiles = [], []
    for task_id in range(10):
        task = suite.get_task(task_id)
        env, panda = make_env(task, "XArm7", 256, a.seed), make_env(task, "Panda", 64, a.seed)
        rows = []
        for perturb, level in CONDS:
            if perturb != "joint" and task_id not in a.render_tasks:  # 재추첨 통계는 joint 만 전체 태스크로
                continue
            args = EP.Args(perturb=perturb, level=level, seed=a.seed, init_offset=a.init_offset, trials=a.trials)
            for ep in range(a.init_offset, a.init_offset + a.trials):
                init = suite.get_task_init_states(task_id)[ep]
                obs, tinfo, pinfo = EP.prepare_episode(env, panda, init, args, task_id, ep)
                stats.append({"task_id": task_id, "episode": ep, "perturb": perturb, "level": level, "pinfo": pinfo})
                if task_id in a.render_tasks and ep == a.init_offset:
                    ag, wr = first_view(env, obs)
                    (out / "tiles").mkdir(exist_ok=True)
                    stem = f"t{task_id}_ep{ep}_{perturb}_{level:g}"
                    imageio.imwrite(out / "tiles" / f"{stem}_agentview.png", ag)
                    imageio.imwrite(out / "tiles" / f"{stem}_wrist.png", wr)
                    tiles.append({"task_id": task_id, "episode": ep, "perturb": perturb, "level": level, "stem": stem,
                                  "task": task.language, "pinfo": pinfo})
                    tile = Image.fromarray(np.concatenate([ag, wr], axis=0))
                    ImageDraw.Draw(tile).text((4, 4), f"{perturb} {level:g}", fill=(255, 255, 0))
                    rows.append(np.asarray(tile))
        env.close(); panda.close()
        if rows:
            imageio.imwrite(out / f"exp7_perturb_t{task_id}_ep{a.init_offset}.png", np.concatenate(rows, axis=1))
    (out / "perturb_stats.json").write_text(json.dumps(stats, indent=0))
    (out / "tiles.json").write_text(json.dumps(tiles, indent=1, ensure_ascii=False))
    for perturb, level in CONDS[1:]:
        s = [x["pinfo"] for x in stats if x["perturb"] == perturb and x["level"] == level]
        if perturb == "joint":
            tr = np.array([p["tries"] for p in s]); sh = np.array([p["eef_shift_cm"] for p in s]); rt = np.array([p["eef_rot_deg"] for p in s])
            print(f"{perturb} {level}: n={len(s)} 재추첨 {int((tr > 1).sum())}회 (최대 {tr.max()}), "
                  f"손끝 이동 평균 {sh.mean():.1f} cm (최대 {sh.max():.1f}), 회전 평균 {rt.mean():.1f}° (최대 {rt.max():.1f})")
        else:
            d0 = np.array([p["dist0"] for p in s]); d = np.array([p["dist"] for p in s]); sg = np.array([p["sign"] for p in s])
            print(f"{perturb} {level}: n={len(s)} 장면 중심까지 {d0.mean():.3f} → {d.mean():.3f} m, 부호 +{int((sg > 0).sum())}/−{int((sg < 0).sum())}, "
                  f"테이블 geom '{s[0]['table']}', 중심 {s[0]['center']}")


if __name__ == "__main__":
    main(tyro.cli(Args))
