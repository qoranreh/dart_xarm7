"""xArm7 을 LIBERO(-view) / robosuite 1.4.1 에 등록한다.

import 하면 다음이 등록되어 `OffScreenRenderEnv(..., robots=["XArm7"])` 로 쓸 수 있다
(LIBERO 의 problem 클래스가 "XArm7" 을 "MountedXArm7" / "OnTheGroundXArm7" 로 바꿔 찾는다):
  - 로봇 모델 MountedXArm7, OnTheGroundXArm7  (ManipulatorModel 메타클래스가 자동 등록)
  - 로봇 클래스 매핑 → SingleArm               (robosuite.robots.ROBOT_CLASS_MAPPING)
  - 그리퍼 XArm7Gripper                        (robosuite.models.grippers.GRIPPER_MAPPING)

모델 파일(assets/)은 robosuite v1.5.2 (MIT) 의 xarm7 / xarm7_gripper 를 그대로 가져왔다. 출처는 assets/SOURCE.md.
robosuite 1.5 → 1.4 API 차이에 맞춘 부분:
  - default_base → default_mount, default_gripper / default_controller_config 는 dict 가 아니라 문자열
  - base_xpos_offset 에 LIBERO 전용 키(study_table, kitchen_table, coffee_table, living_room_table)를 Panda 와 같은 값으로 추가
  - 그리퍼 _important_geoms 를 실제 XML 의 패드 geom 이름으로 수정 (1.5.2 원본은 없는 이름 finger1_pad_collision 을 가리킴)

동역학 (환경 변수 XARM7_DYNAMICS, 기본 "panda"):
  - "panda"     : Panda 모델과 같은 시뮬 규약으로 맞춤 — 팔 관절 frictionloss 0 (Panda XML 에 마찰 없음),
                  그리퍼 speed 0.01 (PandaGripper 와 같은 값 → 닫힘 13스텝으로 Panda 와 같음). 본 실험용.
  - "robosuite" : v1.5.2 원본 그대로 — frictionloss 1 (OSC 이동 응답 ~30% 작음), speed 0.2 (닫힘 7스텝). 동역학 차이 ablation 용.
  측정값은 action_space_check.json / XARM7_NOTES.md.

외형 색 (환경 변수 XARM7_COLOR, 기본 "white", 실험 6 의 시각 요인):
  - "white" : XML 그대로 (아무것도 바꾸지 않음 → 이전 코드와 같은 모델·렌더).
  - "dark"  : XML 로드 후 material "white" 의 rgba 만 "0.15 0.15 0.15 1" 로 바꿈. 이 material 을 쓰는 geom 은
              팔 link_base~link6 (link7 은 visual geom 없음) + 그리퍼 몸통 base_link 뿐. 손가락·너클(material black),
              패드(geom rgba), site, 물리 속성은 그대로. XML 파일은 수정하지 않는다.
"""

import os

import numpy as np
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.single_arm import SingleArm

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

DYNAMICS = os.environ.get("XARM7_DYNAMICS", "panda")
assert DYNAMICS in ("panda", "robosuite"), f"XARM7_DYNAMICS={DYNAMICS!r} (panda | robosuite)"
ARM_FRICTIONLOSS = 0.0 if DYNAMICS == "panda" else None  # None = XML 값(1.0) 유지
GRIPPER_SPEED = 0.01 if DYNAMICS == "panda" else 0.2

COLOR = os.environ.get("XARM7_COLOR", "white")
COLOR_RGBA = {"white": None, "dark": "0.15 0.15 0.15 1"}  # None = XML 값(1 1 1 1) 유지
assert COLOR in COLOR_RGBA, f"XARM7_COLOR={COLOR!r} ({' | '.join(COLOR_RGBA)})"


def _recolor_white_material(model):
    """XARM7_COLOR 가 white 가 아니면 이 모델(로봇 또는 그리퍼)의 material "white" rgba 를 바꾼다. white 면 아무것도 안 함."""
    rgba = COLOR_RGBA[COLOR]
    if rgba is None:
        return
    mats = [m for m in model.asset.findall("material") if m.get("name") == model.naming_prefix + "white"]
    assert len(mats) == 1, f"{type(model).__name__}: material {model.naming_prefix}white 가 {len(mats)}개"
    mats[0].set("rgba", rgba)

# LIBERO MountedPanda 의 초기 그리퍼 포즈와 맞춘 xArm7 관절값 (ik_init_qpos.py, 결과 ik_init_qpos.json).
# 목표: grip_site 위치 = Panda 것, right_hand 자세 = Panda 것 → 손가락 축과 eef_quat 관측값이 Panda 와 같아진다.
# robosuite 1.5.2 기본값 [0, 0, 0, 0.39π, 0, 0.39π, 0] 은 그리퍼가 3~4cm 높고 z축으로 180° 돌아가 있었다.
# joint7 은 IK 해 3.1411 대신 같은 자세인 3.1411-2π 를 쓴다. +π 쪽이면 eef_quat 부호가 Panda 와 반대(-q)가 되어
# 공식 평가 코드의 _quat2axisangle 결과가 (+π·n) ↔ (-π·n) 으로 뒤집힌다 (같은 회전이지만 정책 입력 state 가 달라짐).
XARM7_INIT_QPOS = np.array([0.0, -0.022835, 0.0, 1.098896, 0.0, 1.178532, -3.142085])


class XArm7GripperBase(GripperModel):
    """UFactory xArm 그리퍼 (평행 2지, 6관절: 좌우 driver/finger/inner_knuckle, 텐던 하나로 구동)."""

    def __init__(self, idn=0):
        super().__init__(os.path.join(_ASSETS, "grippers/xarm7_gripper.xml"), idn=idn)
        _recolor_white_material(self)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.02, 0.0, 0.0, -0.02, 0.0, 0.0])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["left_finger_pad_1", "left_finger_pad_2"],
            "right_finger": ["right_finger_pad_1", "right_finger_pad_2"],
            "left_fingerpad": ["left_finger_pad_1", "left_finger_pad_2"],
            "right_fingerpad": ["right_finger_pad_1", "right_finger_pad_2"],
        }


class XArm7Gripper(XArm7GripperBase):
    """액션 1개: -1 => 열기, +1 => 닫기. 목표값을 시뮬 서브스텝마다 speed 만큼 이동 (제어 스텝당 25회)."""

    def format_action(self, action):
        assert len(action) == self.dof
        self.current_action = np.clip(self.current_action + self.speed * np.sign(action), -1.0, 1.0)
        return self.current_action

    @property
    def speed(self):
        return GRIPPER_SPEED

    @property
    def dof(self):
        return 1


class MountedXArm7(ManipulatorModel):
    """테이블 앞 RethinkMount 위의 xArm7 (LIBERO MountedPanda 대응)."""

    def __init__(self, idn=0):
        super().__init__(os.path.join(_ASSETS, "robots/xarm7/robot.xml"), idn=idn)
        self.set_joint_attribute(attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01)))
        if ARM_FRICTIONLOSS is not None:
            self.set_joint_attribute(attrib="frictionloss", values=np.full(7, ARM_FRICTIONLOSS))
        _recolor_white_material(self)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "XArm7Gripper"

    @property
    def default_controller_config(self):
        # LIBERO ControlEnv 는 load_controller_config(default_controller="OSC_POSE") 를 쓰므로 실제로는 안 쓰임.
        # robosuite 1.4.1 에는 default_xarm7.json 이 없어 Panda 것을 지정해 둔다.
        return "default_panda"

    @property
    def init_qpos(self):
        return XARM7_INIT_QPOS

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
            "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"


class OnTheGroundXArm7(MountedXArm7):
    """마운트 없이 바닥에 놓인 xArm7 (LIBERO OnTheGroundPanda 대응: living room / coffee table / floor 씬)."""

    @property
    def default_mount(self):
        return None

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "coffee_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.41),
            "living_room_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.42),
        }


ROBOT_CLASS_MAPPING.update({"MountedXArm7": SingleArm, "OnTheGroundXArm7": SingleArm})
GRIPPER_MAPPING["XArm7Gripper"] = XArm7Gripper


# ---------------------------------------------------------------------------------------------------------------------
# 데모 재생·평가용 유틸 (2026-10-08). env 는 LIBERO OffScreenRenderEnv (env.env 가 robosuite env).
# ---------------------------------------------------------------------------------------------------------------------

# Panda 의 robot0_gripper_qpos[0] = 패드 안쪽 면 간격/2 + 0.0004 (실측: 0.0387↔0.0382, 0.0204↔0.0201, 0.0010↔0.0005)
PANDA_GRIP_QPOS_OFFSET = 0.0004


def pad_surface_gap(env):
    """두 손가락 패드의 마주보는 면 사이 간격 (m). Panda / xArm7 공통."""
    sim, rb = env.env.sim, env.env.robots[0]
    m, d = sim.model._model, sim.data._data
    ids = [[m.geom(g).id for g in rb.gripper.important_geoms[k]] for k in ("left_fingerpad", "right_fingerpad")]
    cL, cR = d.geom_xpos[ids[0]].mean(0), d.geom_xpos[ids[1]].mean(0)
    n = (cL - cR) / np.linalg.norm(cL - cR)

    def half(g):  # 닫힘축 방향 반두께 (box 는 회전 고려, 그 외는 size[0])
        if m.geom_type[g] == 6:
            return float(np.abs(d.geom_xmat[g].reshape(3, 3).T @ n) @ m.geom_size[g])
        return float(m.geom_size[g][0])

    return float(np.linalg.norm(cL - cR) - half(ids[0][0]) - half(ids[1][0]))


def panda_equiv_gripper_qpos(env):
    """xArm7 그리퍼(6관절)를 Panda 형식 robot0_gripper_qpos (2차원, (+q, -q), m) 로 환산 — 같은 패드 간격이면 같은 값."""
    q = max(pad_surface_gap(env), 0.0) / 2 + PANDA_GRIP_QPOS_OFFSET
    return np.array([q, -q])


def quat2axisangle(quat):
    """examples/libero-view/main.py 의 _quat2axisangle 과 같은 식 (xyzw)."""
    import math

    quat = np.array(quat, dtype=float)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def policy_state(env, obs):
    """정책 입력 state 8차원 = eef_pos(3) + axis-angle(3) + gripper_qpos(2). xArm7 은 그리퍼를 Panda 형식으로 환산."""
    grip = obs["robot0_gripper_qpos"]
    if isinstance(env.env.robots[0].gripper, XArm7GripperBase):
        grip = panda_equiv_gripper_qpos(env)
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), grip]).astype(np.float32)


def gripper_pose(env):
    """(grip_site 월드 위치, right_hand 월드 회전행렬)."""
    sim, rb = env.env.sim, env.env.robots[0]
    m, d = sim.model._model, sim.data._data
    return d.site_xpos[rb.eef_site_id].copy(), d.xmat[m.body(rb.robot_model.eef_name).id].reshape(3, 3).copy()


def solve_ik(env, pos_goal, hand_mat_goal, q_seed, iters=3000):
    """grip_site 위치 = pos_goal, right_hand 자세 = hand_mat_goal 이 되는 팔 관절값 (damped least squares, null-space 로 q_seed 쪽 유도).

    xArm7 은 grip_site 와 right_hand 의 자세가 같아 grip_site 자세 목표 = hand_mat_goal.
    반환: (q, 위치오차 m, 자세오차 rad). sim 의 qpos 를 바꾸므로 호출 뒤 상태를 다시 세팅할 것.
    """
    import mujoco

    sim, rb = env.env.sim, env.env.robots[0]
    m, d = sim.model._model, sim.data._data
    qidx, didx = np.array(rb._ref_joint_pos_indexes), np.array(rb._ref_joint_vel_indexes)
    jids = [m.joint(n).id for n in rb.robot_model.joints]
    lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
    sid = rb.eef_site_id
    hand = m.body(rb.robot_model.eef_name).id
    q = np.array(q_seed, dtype=float)
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    qtmp, v = np.zeros(4), np.zeros(3)
    for _ in range(iters):
        d.qpos[qidx] = q
        mujoco.mj_forward(m, d)
        mujoco.mju_mat2Quat(qtmp, (hand_mat_goal @ d.xmat[hand].reshape(3, 3).T).flatten())
        mujoco.mju_quat2Vel(v, qtmp, 1.0)
        e = np.concatenate([pos_goal - d.site_xpos[sid], v])
        if np.linalg.norm(e[:3]) < 1e-6 and np.linalg.norm(e[3:]) < 1e-5:
            break
        mujoco.mj_jacSite(m, d, jacp, jacr, sid)
        J = np.vstack([jacp[:, didx], jacr[:, didx]])
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
        q = np.clip(q + 0.5 * (Jp @ e + 0.05 * (np.eye(len(q)) - Jp @ J) @ (np.asarray(q_seed) - q)), lo, hi)
    return q, float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))


def _joint_slices(m):
    """{조인트 이름: (qpos slice, qvel slice)}."""
    import mujoco

    nq = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}
    nv = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}
    out = {}
    for j in range(m.njnt):
        t = m.jnt_type[j]
        qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
        out[m.joint(j).name] = (slice(qa, qa + nq.get(t, 1)), slice(va, va + nv.get(t, 1)))
    return out


def transfer_init_state(panda_env, panda_flat_state, xarm_env, q_seed=None):
    """Panda 용 LIBERO 시뮬 상태(flattened)를 xArm7 env 의 같은 장면 상태로 옮긴다.

    - 물체(로봇·그리퍼가 아닌 모든 조인트)의 qpos/qvel 은 이름으로 그대로 복사
    - xArm7 팔은 Panda 의 그 순간 그리퍼 포즈(grip_site 위치 + right_hand 자세)에 IK 로 맞춤, 그리퍼는 기본(열림) 자세
    반환: (xArm7 flattened state, info). xarm_env.set_init_state(반환값) 으로 쓰면 된다.
    """
    panda_env.set_init_state(panda_flat_state)
    pos_goal, hand_goal = gripper_pose(panda_env)
    xarm_env.reset()  # LIBERO 는 hard_reset=True 라 reset 마다 MjSim 을 새로 만든다 → sim 은 reset 뒤에 가져올 것
    psim, xsim = panda_env.env.sim, xarm_env.env.sim
    pj, xj = _joint_slices(psim.model._model), _joint_slices(xsim.model._model)
    robot = lambda n: n.startswith(("robot0_", "gripper0_", "mount0_"))
    copied = [n for n in pj if not robot(n) and n in xj]
    missing = [n for n in pj if not robot(n) and n not in xj]
    rb = xarm_env.env.robots[0]
    q, ep, er = solve_ik(xarm_env, pos_goal, hand_goal, XARM7_INIT_QPOS if q_seed is None else q_seed)
    xd, pd = xsim.data._data, psim.data._data
    for n in copied:
        xd.qpos[xj[n][0]] = pd.qpos[pj[n][0]]
        xd.qvel[xj[n][1]] = pd.qvel[pj[n][1]]
    xd.qpos[rb._ref_joint_pos_indexes] = q
    xd.qvel[rb._ref_joint_vel_indexes] = 0.0
    xd.qpos[rb._ref_gripper_joint_pos_indexes] = rb.gripper.init_qpos
    xd.qvel[rb._ref_gripper_joint_vel_indexes] = 0.0
    xd.time = pd.time
    xsim.forward()
    flat = xsim.get_state().flatten()
    return flat, {"ik_q": q.tolist(), "ik_pos_err": ep, "ik_rot_err": er, "copied_joints": copied, "missing_joints": missing}
