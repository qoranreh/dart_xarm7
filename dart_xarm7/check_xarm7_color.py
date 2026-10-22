"""XARM7_COLOR 분기 점검: 이 프로세스의 XARM7_COLOR 로 xArm7 libero_spatial t0 env 를 만들어
(1) 합쳐진 MJCF 문자열과 (2) 컴파일된 모델의 시각·물리 배열을 npz 로 저장한다. 두 실행(white / dark)의 결과를 비교하는 데 쓴다.

  XARM7_COLOR=white $LIBERO_PY check_xarm7_color.py <out.npz>
  /usr/bin/python3 check_xarm7_color.py --compare a.npz b.npz
"""
import sys

import numpy as np

if sys.argv[1] == "--compare":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    xa, xb = str(a["xml"]).splitlines(), str(b["xml"]).splitlines()
    diff = [(i, x, y) for i, (x, y) in enumerate(zip(xa, xb)) if x != y]
    print(f"MJCF 줄 수 {len(xa)} / {len(xb)}, 다른 줄 {len(diff)}" + ("" if len(xa) == len(xb) else " (줄 수 다름)"))
    for i, x, y in diff[:10]:
        print(f"  {i}: {x.strip()}\n   → {y.strip()}")
    for k in sorted(a.files):
        if k in ("xml", "color", "dynamics"):
            continue
        same = a[k].shape == b[k].shape and np.array_equal(a[k], b[k])
        print(f"  {k:24s} {'같음' if same else '다름: ' + str(np.argwhere(a[k] != b[k])[:, 0].tolist()[:12])}")
    print("color", str(a["color"]), "vs", str(b["color"]), "| dynamics", str(a["dynamics"]), "vs", str(b["dynamics"]))
    sys.exit(0)

import pathlib  # noqa: E402

import libero_xarm7 as X  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

t = benchmark.get_benchmark_dict()["libero_spatial"]().get_task(0)
env = OffScreenRenderEnv(bddl_file_name=str(pathlib.Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file),
                         robots=["XArm7"], camera_heights=64, camera_widths=64, camera_names=["agentview"])
env.seed(7)
env.reset()
m = env.env.sim.model._model
arrs = {k: np.array(getattr(m, k)) for k in ("mat_rgba", "geom_rgba", "geom_matid", "geom_group", "geom_contype", "geom_conaffinity",
                                              "geom_friction", "geom_size", "body_mass", "body_inertia", "dof_frictionloss",
                                              "dof_damping", "dof_armature", "actuator_gainprm", "jnt_range")}
np.savez(sys.argv[1], xml=env.env.model.get_xml(), color=getattr(X, "COLOR", "(분기 없음)"), dynamics=X.DYNAMICS, **arrs)
names = [m.mat(i).name for i in range(m.nmat)]
print("COLOR", getattr(X, "COLOR", "(분기 없음)"), "| materials:", {n: m.mat_rgba[i].round(3).tolist() for i, n in enumerate(names) if n.startswith(("robot0_", "gripper0_"))})
env.close()
