# LIBERO Panda → xArm7 교체 메모 — 2026-10-08 (잡 129138)

## 결론

- **xArm7 이 LIBERO-view 의 모든 suite 에서 로딩·렌더링된다** (spatial / goal = 마운트형, object / libero_10 = 바닥형 확인).
- **선택: (a) robosuite 생태계의 xArm7 모델을 가져와 등록.** UR5e 우회(b) 는 필요 없었다.
  xArm7 은 robosuite 1.4.1(LIBERO 고정 버전) 에 없고 v1.5.2 에 정식 포함 → v1.5.2 의 XML·메쉬를 1.4.1 API 클래스로 감싸 등록.
  robosuite 를 1.5 로 올리는 길은 컨트롤러·로봇 API 가 바뀌어 LIBERO 전체를 고쳐야 해서 택하지 않음.
- **dart 저장소·LIBERO-view 서브모듈은 한 줄도 수정하지 않았다.** 모든 추가분은 홈(`~/hayeon/dart_xarm7/`, 5.4M) 에 있음.
- **액션 공간: 같은 7차원 OSC_POSE (Δpos 3 + Δrot 3 + gripper 1, 범위 [-1,1], 월드 프레임 델타).** 방향·의미는 같고 응답 크기에 차이가 있음 (아래).

## 사용법

```bash
source ~/hayeon/dart_xarm7/env.sh && source ~/hayeon/dart_xarm7/client_env.sh   # PYTHONPATH 에 src/ 가 들어감
```
```python
import libero_xarm7                                   # MountedXArm7 / OnTheGroundXArm7 / XArm7Gripper 등록
env = OffScreenRenderEnv(bddl_file_name=..., robots=["XArm7"], ...)   # problem 클래스가 "Mounted"/"OnTheGround" 를 붙임
```

## 1. Panda 가 지정되는 지점 (LIBERO-view @2e8af0f, `third_party/libero-view/libero/libero/envs/`)

| 위치 | 내용 |
|---|---|
| `env_wrapper.py:17` | `ControlEnv(robots=["Panda"], controller="OSC_POSE", ...)` 기본값. `OffScreenRenderEnv` 가 상속. 공식 평가 `examples/libero-view/main.py:212` 는 robots 를 안 넘김 → 기본값 Panda |
| `env_wrapper.py:56` | `suite.load_controller_config(default_controller="OSC_POSE")` — 로봇 무관한 범용 OSC 설정 (로봇의 default_controller_config 는 안 쓰임) |
| `problems/libero_tabletop_manipulation.py:23` 외 6개 파일 | `f"Mounted{robot_name}"` / `f"OnTheGround{robot_name}"` 로 이름 변환 (tabletop·kitchen·study = Mounted, floor·coffee·living room = OnTheGround) |
| `robots/__init__.py:7-13` | `ROBOT_CLASS_MAPPING` 에 MountedPanda 등 → SingleArm 등록 |
| `robots/mounted_panda.py`, `on_the_ground_panda.py` | 로봇 모델 클래스 (마운트, 그리퍼, init_qpos, base 오프셋) |

## 2. 수정·추가한 것 (전부 신규 파일, 기존 파일 수정 없음)

`src/libero_xarm7/__init__.py` (≈150줄) + `assets/` (robosuite v1.5.2 원본 XML 2개 + 메쉬 16개, 출처 `assets/SOURCE.md`)

| 터질 후보 | 결과 / 처리 |
|---|---|
| 로봇 등록 | `MountedXArm7`, `OnTheGroundXArm7` (ManipulatorModel) 정의 → 자동 등록, `ROBOT_CLASS_MAPPING` 에 SingleArm 추가. 1.5→1.4 API: `default_base`→`default_mount`, dict→문자열 |
| 그리퍼 매핑 | `XArm7Gripper` 를 `GRIPPER_MAPPING` 에 추가. 1.5.2 원본의 `_important_geoms` 가 XML 에 없는 이름(`finger1_pad_collision`)을 가리켜서 실제 패드 이름(`left/right_finger_pad_1,2`)으로 수정 |
| base 마운트 위치 | RethinkMount, base_xpos_offset 을 Panda 와 같은 값으로 (LIBERO 전용 키 study_table / kitchen_table / coffee_table / living_room_table 추가). base 월드 위치 (-0.66, 0, 0.912) 로 Panda 와 동일 |
| **초기 관절값** | robosuite 기본값은 그리퍼가 3~4cm 높고 z축으로 180° 돌아가 있었음 → IK 로 재계산 (`ik_init_qpos.py`). 아래 함정 2개 참고 |
| 카메라 | agentview 계열은 씬(월드)에 고정이라 영향 없음. 손목 카메라는 right_hand 기준 위치가 Panda 와 같음(+5cm). 초기 자세 수정 후 손목 시점 방향도 Panda 와 같아짐 |

**초기 관절값 함정:**
1. Panda 는 `grip_site` 가 `right_hand` 대비 접근축으로 -90° 돌아가 있고 xArm7 은 같다. `grip_site` 자세를 맞추면 손가락 방향이 90° 어긋난다.
   → 목표를 "grip_site **위치** + right_hand **자세**" 로 둠. 그 결과 그리퍼 위치(오차 <0.01mm), 손가락 방향, `eef_quat` 관측값이 모두 Panda 와 같다.
2. IK 해의 joint7 = +π 이면 `eef_quat` 부호가 Panda 와 반대(-q)가 되어, 공식 평가 코드의 `_quat2axisangle` 이 (+π·n) 대신 (-π·n) 을 낸다.
   같은 회전이지만 정책 입력 state 가 크게 달라짐 → 같은 자세인 joint7 = π-2π 를 씀.
   최종 `XARM7_INIT_QPOS = [0, -0.022835, 0, 1.098896, 0, 1.178532, -3.142085]`

## 3. 렌더링 (`renders/`)

- `compare_panda_xarm7_libero_spatial_t0.png` — Panda / xArm7 의 agentview·손목 2×2 비교
- `XArm7_libero_spatial_t0_{agentview,robot0_eye_in_hand}.png` (256×256, 공식 평가처럼 180° 회전한 모델 입력 방향)
- `suites/XArm7_<suite>_t0_agentview.png` — 4개 suite
- **EGL 은 쓸 수 없다**: 컨테이너에 GPU graphics capability 가 없음 (`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`, libEGL 없음).
  robosuite 가 기본으로 EGL 을 잡다가 `eglQueryString` AttributeError 로 실패하는 것을 확인. 런처(`/tools/`) 설정이라 바꿀 수 없음
  → OSMesa (conda-forge `mesalib<25`, CPU 렌더, headless). `client_env.sh` 가 `MUJOCO_GL=osmesa` 설정.

## 4. 액션 공간 동일 여부 (`action_space_check.py`, 결과 `action_space_check.json`)

| 항목 | Panda | xArm7 | 판정 |
|---|---|---|---|
| 차원 / 범위 | 7 / [-1,1] | 7 / [-1,1] | 같음 |
| 컨트롤러 | OSC_POSE, control_delta, ±0.05m / ±0.5rad, kp 150 | 같은 설정 | 같음 |
| 프레임 | 월드 프레임 델타 | 월드 프레임 델타 | 같음 (+x→+x, +y→+y, +z→+z, yaw→월드 z, roll→월드 x 확인) |
| 그리퍼 액션 | 1차원, -1 열림 / +1 닫힘 | 같음 | 같음 |
### 동역학 — 2026-10-08 결정: Panda 시뮬 규약에 맞춤 (`XARM7_DYNAMICS=panda`, 기본값)

액션 0.3 × 10스텝의 이동량 (이동은 벡터 크기), 그리퍼는 +1/-1 을 줬을 때 패드 간격이 최종값 2mm 이내에 드는 스텝 수.

| 항목 | Panda | xArm7 기본 (`panda`) | xArm7 원본 (`robosuite`) |
|---|---|---|---|
| +x / +y / +z | 3.09 / 3.68 / 3.57cm | 2.84 / 3.71 / 3.74cm (−8 / +1 / +5%) | 2.35 / 2.66 / 2.63cm (−24 / −28 / −26%) |
| yaw / roll | 0.306 / 0.345rad | 0.366 / 0.364rad (+20 / +5%) | 0.328 / 0.340rad (+7 / −1%) |
| 그리퍼 닫힘 / 열림 | 13 / 13스텝 | 13 / 12스텝 | 7 / 6스텝 |
| 설정 | — | 팔 관절 frictionloss 0, 그리퍼 speed 0.01 | frictionloss 1, speed 0.2 (v1.5.2 원본) |

- 이동 응답이 작던 원인은 **xArm7 XML 의 관절 마찰 `frictionloss="1"`** (Panda XML 에는 없음, OSC 는 마찰을 보상하지 않음).
- 그리퍼 차이는 `speed` 값 (Panda 0.01, xArm7 원본 0.2). 그리퍼 명령은 시뮬 서브스텝마다 적용됨 (`robosuite/robots/single_arm.py:258`, 제어 스텝당 25회).
  Panda 와 같은 0.01 로 두면 닫힘 시간과 폭 곡선이 Panda 와 거의 같다 (0.006 → 18스텝, 0.004 → 23스텝).
- 맞춘 뒤에도 남는 차이(yaw +20%, x −8%)는 팔 길이·관성 같은 실제 기구학 차이라 그대로 둔다. DART 가 다뤄야 하는 embodiment 차이에 해당.
- **맞춘 이유:** frictionloss 1 은 실측값이 아니라 robosuite 모델링 선택이다. 그대로 두면 실험이 실패했을 때 DART 가 외형·기구학 차이를 못 다룬 건지,
  시뮬 설정값에서 온 동역학 차이를 못 다룬 건지 분리가 안 된다. 원본은 `XARM7_DYNAMICS=robosuite` 로 남겨 두어
  "DART 가 동역학 차이도 흡수하는가" ablation 에 쓴다. 결과: `action_space_check.json` (기본), `action_space_check.robosuite_dynamics.json` (원본).

**정책 입력 state 는 다르다 (액션은 아님):** state = eef_pos(3) + axis-angle(3) + `robot0_gripper_qpos`.
앞 6차원은 초기 자세에서 Panda 와 완전히 같지만, 그리퍼가 Panda 2차원(손가락 위치 m) / xArm7 6차원(관절각 rad) → **8차원 vs 12차원**.
pi05 는 state 를 패딩하므로 바로 터지지는 않지만 의미가 다름 → 데이터·평가 단계에서 2차원 손가락 폭 등가값으로 매핑 필요.

## 남은 문제 (다음 단계)

1. **Panda 용 init state 를 못 쓴다.** `env.set_init_state(...)` 에 공식 init 파일을 넣으면 `ValueError: could not broadcast (39,) into (47,)`
   (xArm7 그리퍼 관절이 4개 많아 qpos+qvel 이 8 길어짐). 공식 평가가 이걸 씀 → 물체 상태만 이름으로 옮겨 xArm7 용 init 파일을 만들어야 함.
2. 공식 평가 `examples/libero-view/main.py` 는 robots 를 안 넘김 → xArm7 평가 시 `robots=["XArm7"]` + `import libero_xarm7` 추가 필요.
3. ~~관절 마찰·그리퍼 speed~~ → Panda 규약에 맞춤으로 결정 (위 "동역학" 절). 손목 카메라는 그대로 둠
   (xArm7 그리퍼가 5cm 길고 커서 손목 화면 아래 약 28% 를 가림, Panda 약 17% — 실제 하드웨어 차이).
