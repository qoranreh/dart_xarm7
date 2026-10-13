"""각 LIBERO suite 의 task 0 을 xArm7 로 로딩 (마운트형/바닥형 둘 다 확인) + Panda 용 init state 를 넣었을 때의 동작 확인."""
import pathlib, traceback
import imageio, numpy as np
import libero_xarm7  # noqa: F401
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

out = pathlib.Path(__file__).with_name("renders") / "suites"
out.mkdir(parents=True, exist_ok=True)
for name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    s = benchmark.get_benchmark_dict()[name]()
    t = s.get_task(0)
    bddl = str(pathlib.Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file)
    try:
        env = OffScreenRenderEnv(bddl_file_name=bddl, robots=["XArm7"], camera_heights=256, camera_widths=256,
                                 camera_names=["agentview"])
        env.seed(7)
        o = env.reset()
        for _ in range(10):
            o, *_ = env.step([0.0] * 6 + [-1.0])
        imageio.imwrite(out / f"XArm7_{name}_t0_agentview.png", o["agentview_image"][::-1, ::-1])
        msg = f"OK  robot={type(env.env.robots[0].robot_model).__name__} problem={env.env.__class__.__name__}"
        init = s.get_task_init_states(0)[0]
        try:
            env.set_init_state(init)
            msg += f" | Panda init state (len {len(init)}) → 에러 없이 들어감 (값이 맞는지는 별도)"
        except Exception as e:
            msg += f" | Panda init state (len {len(init)}, xArm7 sim 기대 {env.env.sim.get_state().flatten().shape[0]}) → {type(e).__name__}: {str(e)[:80]}"
        env.close()
    except Exception as e:
        msg = "FAIL " + "".join(traceback.format_exception_only(type(e), e)).strip()[:300]
    print(f"[{name}] {t.language}\n   {msg}")
