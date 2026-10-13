# one-shot 데모 쌍 (2026-10-08, 잡 129138)

리포트(영상 포함): https://claude.ai/artifact/XkXvY2GqCuvywJGGnpRYR8  (로컬: report/demo_pair_report.html)

| | 경로 (LeRobot v2.1, 1 에피소드, 98 프레임, 13MB) |
|---|---|
| source Panda | `~/hayeon/dart_xarm7/data/lerobot/libero_spatial_t0_demo_0_src_panda` |
| **target xArm7 (채택: MimicGen + 회전 보정 제한)** | `~/hayeon/dart_xarm7/data/lerobot/libero_spatial_t0_demo_0_tgt_xarm7_retarget_rotclip` |
| target xArm7 (MimicGen 제한 없음) | `~/hayeon/dart_xarm7/data/lerobot/libero_spatial_t0_demo_0_tgt_xarm7_retarget` |
| target xArm7 (액션 재생, 이전 버전) | `~/hayeon/dart_xarm7/data/lerobot/libero_spatial_t0_demo_0_tgt_xarm7_replay` (원래 이름 `_tgt_xarm7`) |

학습: `DATA_REPO_ID=<위 절대경로> CAMERA_VIEW=original bash exec/finetune.sh` (src/tgt 마다 CHECKPOINT_BASE_DIR 다르게).

## 만든 방법 (재현)
1. `LIBERO-datasets` hdf5 (libero_spatial task 0, 509MB) 에서 demo_0 의 actions/states 만 HTTP 범위 요청으로 읽음
   → `$DART_ROOT/demos/spatial_t0_hdf5_demo0-4.npz` (demo_0~4). 공개 DART 데이터의 이 태스크 에피소드 = demo_0 액션 98개 그대로 (no-op 필터 없음).
2. `make_demo_pair.py` (libero venv): reset → set_init_state(states[0]) → 대기 10스텝 → [기록 → step] × 98.
   xArm7 은 `libero_xarm7.transfer_init_state` (물체 조인트 9개 이름으로 복사 + Panda 그리퍼 포즈에 IK) 로 같은 시작 장면.
   state 그리퍼는 `policy_state` 가 Panda 형식으로 환산 (패드 안쪽 면 간격/2 + 0.0004). 동역학 XARM7_DYNAMICS=panda.
3. `convert_to_lerobot.py` (dart .venv, `uv run`): 저장소 `examples/libero-view/convert_libero_data_to_lerobot.py` 와 같은 스키마.
4. `report_assets.py` → `build_report.py` → Artifact.

## 결과
- 둘 다 성공 (첫 성공 스텝 Panda 81, xArm7 76). 그리퍼 위치 차이 평균 1.7 / 최대 3.0 cm. 보정 불필요.
- Panda 녹화 = 공식 DART 에피소드 재현 (state 차 ≤1e-4, 액션 동일, 이미지는 EGL vs OSMesa 차이로 평균 2.3/255).
- 액션은 재생이라 두 데모가 동일 → norm stats 쪽 액션 스케일 문제 없음 (FT 는 pi05_libero 체크포인트의 LIBERO 통계를 그대로 씀).
- xArm7 state 그리퍼 열림값 0.044 > q99 0.040 — pi05 가 state 를 안 쓰므로 영향 없음 (자르지 않음).

## 2026-10-08 추가: MimicGen 방식 target (`make_retarget_demo.py`)
DART 논문 부록 C.3: UR5e 데모는 공식 MimicGen 파이프라인으로 생성 ("first generated trajectory"). 같은 규칙으로 만듦:
목표 = Panda 데모의 명령 목표 포즈(현재+액션×0.05m/0.5rad), 액션 = clip((목표−xArm7 현재)/0.05, 회전/0.5), 그리퍼 복사, 노이즈 없음.
그리퍼 좌표계 90° 차이는 xArm7 에 Panda right_hand 목표 자세를 주는 것으로 보정.
- 성공 (첫 성공 74). Panda 궤적과 위치차 평균 1.8 / 최대 3.6 cm. 27스텝에서 액션이 ±1 로 잘림.
- **액션 스케일 경고:** Δpitch 정규화 최대 2.62, 프레임 32% 가 pi05_libero norm q01~q99 밖 (학습 데이터 회전 액션은 ±0.18 안쪽).
  원인: x 이동 시 손목 기울어짐(±2~5°) + 그릇 내려놓을 때 최대 13° → MimicGen 보정이 큰 회전 액션이 됨.
  논문은 norm stats 를 UR5e 포함 데이터로 계산해서 범위 안이었음.
- 시험: `--rot-clip quantile` (회전을 norm q01~q99 로 제한) 도 성공 (73). npz 만 있고 데이터셋은 안 만듦.
- state 는 pi05 (discrete_state_input=False) 에서 모델 입력이 아니므로 그리퍼 state 자르기는 결과에 영향 없음.

## 2026-10-08 채택: MimicGen + 회전 보정 제한 (사용자 선택, 3안 중 2번)
`make_retarget_demo.py --rot-clip quantile` → npz `..._XArm7_retarget_rotq.npz` → 데이터셋 `_tgt_xarm7_retarget_rotclip`.
회전 액션(Δroll/Δpitch/Δyaw)만 pi05_libero norm q01~q99 로 자름 (MimicGen 원래 규칙에 없는 단계).
- 성공 (첫 성공 73). Panda 궤적과 위치차 평균 1.8 / 최대 3.3 cm. 추종오차 위치 4.0/6.0 cm, 회전 5.9/17.1°.
- 정규화 액션: 회전 3차원 모두 ±1 안 (범위 밖 0%). Δx·Δz 는 프레임 22%·21% 가 1.07 (액션 상한 ±1 > q99 0.94).
- openpi pi05_libero_oneshotft 파이프라인으로 읽힘 확인.
