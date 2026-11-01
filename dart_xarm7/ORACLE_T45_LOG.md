# 실험 4-후속1 — t4·t5 정답 궤적 실행 검사 (Panda 원본 데모 → xArm7 retarget) — 2026-11-02, 노트북

**실험 문장 (사용자 확정):** 이건 Panda 정답 궤적의 xArm7 retarget 으로 바꿔서 t4·t5 성공 수를 보고, 1/5 이하면 H_body(몸이 못 함) / 4/5 이상이면 H_policy(정책이 못 배움).
rot-clip 은 quantile·none 둘 다 보고, 나머지 경우(rot-clip 있을 때만 실패 = 데모 규칙 문제, 2~3/5 = 판단 불가)는 STOP2 규칙대로. 학습·합성·정책 평가 없음.

## 최종 결론 (demo 0~9, 2026-11-02 08:51 — 아래 9~11절)
1. Panda 재생 → xArm7 retarget 성공 (끝에서 성공, 괄호 = 도중에 한 번이라도): **t4 9/10 → 5/10 (rot-clip) · 2/10 (4/10) (없음)**, **t5 10/10 → 5/10 · 5/10**.
   t4 demo6 은 Panda 재생부터 실패 (비교 제외). 같은 retarget 40개를 세 번 돌려도 demo별 성공/실패가 전부 같음.
2. 실패 21개 (t4 demo6 제외) = **파지 10 (빈손으로 닫힘 6, 기운 채 잡힘 4) · 들기 2 · 놓기 9** · 접근 0.
   t4 (11): 서랍 안 파지 실패 7, 쥐었는데 서랍에서 못 들어 올림 2, 접시에 놓아 성공했다가 그릇이 넘어짐 2 (rot-clip 없음).
   t5 (10): **램킨이 그릇과 같이 들려 접시 위 램킨에 그릇이 얹힘 6 (xArm7 에서만, 같은 demo 의 Panda 재생에선 램킨 안 움직임)**, 접시 중심 3.6cm 1, 파지 3.
3. 판정: H_body (≤1/5, 10개면 ≤2/10) 도 H_policy (≥4/5, ≥8/10) 도 아님 — 두 task 모두 정답 궤적으로 절반쯤 성공. demo 를 늘려도 판단 불가 구간 그대로.
   rot-clip: t5 는 같고, t4 는 자르는 쪽이 더 잘 됨 → "자를 때만 실패" (데모 규칙 문제) 아님.
   참고: θ0 의 xArm7 평가는 t4 6/10·2/10, t5 0/10·0/10 (EVAL_RESULTS.md, 장면이 다름) → t5 는 정답 궤적(5/10)보다 정책이 크게 낮음.

**정정 (2026-11-02):** 처음 STOP2 보고와 4절에 "t5 demo4 는 Panda 원본 재생에서도 램킨이 같이 들렸다" 고 적었는데 **틀림**.
그 근거였던 일회성 점검 스크립트가 `Panda` 인자에도 xArm7 env 를 만들어 (xArm7 에 Panda 액션을 넣은 것) 잘못 읽었다. 11절의 실제 Panda 재생에서는 램킨이 2cm 이상 움직이지 않음.

## 첫 보고 (STOP2, demo 0~4)
1. Panda 재생 → xArm7 retarget 성공: **t4 5/5 → 2/5 (rot-clip) · 2/5 (없음)**, **t5 5/5 → 3/5 (rot-clip) · 3/5 (없음)**.
2. 실패 10개 = **파지 7 (빈손으로 닫힘 5, 기운 채 잡힘 2) · 놓기 3** · 접근 0 · 들기 0. t4 는 6개 전부 서랍 안 파지 실패
   (Panda 보다 1.8~3.1cm 높거나 1.2~1.5cm 옆에서 닫혀 양 패드가 3~9스텝만 닿고 빠짐, 그릇은 ≤1.5cm 들림). t5 놓기 3개 중 2개(demo4)는 램킨이 그릇에 붙어 같이 들린 경우 (xArm7 에서만 — 위 정정).
   추종오차 위치 평균은 t0 와 비슷 (3.5~4.3 vs 4.0cm) 하지만 최대가 큼 (10~12 vs 6.0cm), 회전 quantile t4 9.4/24.1° · t5 5.4/21.7° vs t0 5.9/17.1°. ±1 클립 10~53스텝 (t0 27, 단 액션 수 119~158 vs 98).
3. 예상과 대조: 네 칸 모두 2~3/5 → **판단 불가**. H_body(≤1/5) 도 H_policy(≥4/5) 도 아님. 두 rot-clip 의 성공 수가 같아 데모 규칙 문제도 아님
   (t5 demo2·demo3 만 서로 엇갈림). → **demo_5~9 추가 제안** (같은 러너에 `DEMOS="5 6 7 8 9"`, fetch 의 `names[:5]` 를 늘려야 함 — 아래 8절).

---

## 0. 환경 (07:49 → 08:01) — ECE 서버가 아니라 노트북

이 세션은 노트북에서 돌았고 ECE 클러스터에 SSH 키·설정이 없어 접속 불가 → 사용자 선택으로 노트북에서 1~6번 실행 (7번은 건너뜀, 5절).

| 항목 | 값 |
|---|---|
| 날짜 / 잡 ID | 2026-11-02 / **없음** (PBS 잡 아님) |
| 노드 | `qoranreh-ASUS-TUF-Gaming-A14-FA401UM` — AMD Ryzen 7 260 (16 스레드), RAM 30GB, RTX 5060 Laptop 8GB (드라이버 595.91.07) |
| 조건 env | `exp6_cond.sh` 의 `exp6_env v0` → **XARM7_COLOR=white XARM7_DYNAMICS=panda** (retarget npz 의 `env_meta` 로 실제 읽은 값 확인) |
| `$DART_ROOT` | `~/dart_laptop` (영구 디스크, 2.8G) |
| 저장소 | `$DART_REPO=~/dart_laptop/dart` = snumprlab/dart @1b23c4f, 서브모듈은 **libero-view 2e8af0f 만** 초기화 (aloha·libero 미초기화). 수정 없음. 클론 172s |
| 레포 (이 문서) | `~/hayeon` = github qoranreh/dart_xarm7 @a6aaac1 클론 (서버와 같은 배치 → `~/hayeon/dart_xarm7/src`) |
| 클라이언트 venv | `~/dart_laptop/venvs/dart-libero` — uv, `/usr/bin/python3.10` (3.10.12). mujoco 3.2.3, robosuite 1.4.1 (`--no-deps`), gym 0.25.2 (`--no-deps`), bddl 1.0.1, numpy 1.24.4, numba 0.57.1, torch 2.1.0+cpu, h5py 3.16.0, fsspec 2026.9.0 (+aiohttp 3.14.5), future 1.0.0, imageio 2.38.0 (+ffmpeg), opencv-python-headless 4.11.0.86, scipy 1.15.3, pillow 12.3.0. 설치 65s |
| 렌더 | **EGL** (NVIDIA, `MUJOCO_GL=egl`). 서버는 OSMesa → 이미지 픽셀만 다를 수 있고 물리는 무관 |
| norm_stats | `$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero` → `~/ckpts/pi05_libero` 링크 (make_retarget_demo.py 의 rot-clip 기준) |
| 환경 변수 | `source ~/hayeon/dart_xarm7/env_laptop.sh` (신규 — env.sh 는 `PBS_JOBID` 필수라 못 씀. env.sh + client_env.sh 와 같은 변수, 경로만 노트북) |

- 서버 venv 의 requirements (`~/venvs/dart-libero-requirements.txt`) 는 서버 홈에만 있고 레포에 없음 → LIBERO-view `requirements.txt`·`examples/libero-view/requirements.txt` 의 핵심 버전(mujoco·robosuite·gym·bddl)만 맞춰 새로 구성.
- **노트북 재현 확인 (`run_oracle_t45.sh check0`, t0 demo_0 → `$DART_ROOT/demos/oracle_t0_check/demo0`, 22s)** — DEMO_PAIR.md 와 표시 자릿수까지 같음 → 노트북 환경으로 진행.

| t0 demo_0 | DEMO_PAIR.md (서버 10-08) | 노트북 |
|---|---|---|
| Panda 재생 첫 성공 | 81 | 81 |
| retarget quantile: 성공 / 첫 성공 / 추종 위치 / 회전 / Panda 대비 / ±1 클립 | 성공 / 73 / 4.0·6.0cm / 5.9·17.1° / 1.8·3.3cm / 27 | 성공 / 73 / 4.01·6.01 / 5.90·17.06 / 1.78·3.31 / 27 |
| retarget none: 성공 / 첫 성공 / Panda 대비 / ±1 클립 | 성공 / 74 / 1.8·3.6 / 27 | 성공 / 74 / 1.82·3.65 / 27 |
| IK 오차 (transfer_init_state) | — | 위치 7.9e-7, 회전 6.7e-7, 물체 조인트 9개 복사, 누락 0 |

## 1. STOP1 (08:00 → 08:03, 시작부터 약 17분)

### 1-1. `fetch_hdf5_demo.py` 일반화 (수정)
- `--suite`(기본 libero_spatial), `--task`(기본 0) 추가. hdf5 이름 = benchmark 의 `suite.get_task(task).name + "_demo.hdf5"`, 기본 출력 `$DART_ROOT/demos/<suite 에서 libero_ 뺀 것>_t<task>_hdf5_demo0-4.npz`.
  기존 위치 인자(출력 경로)는 그대로 받음. 404 (`FileNotFoundError`) 면 메시지와 함께 종료.
- dart `.venv` 에는 libero 가 없어서 (pyproject 에 robosuite·bddl 없음) 기존 실행법 (`uv run`, task 0) 이 깨지지 않게, libero import 가 안 되고 인자가 기본값일 때만 t0 이름 상수를 씀.
  task 0 의 benchmark 이름 = 기존 하드코딩 URL 의 파일명 (확인) → 기본 동작 동일.
- 명령: `$LIBERO_PY ~/hayeon/dart_xarm7/fetch_hdf5_demo.py --task {0,4,5}` (동시 3개)

| task | URL 파일명 | demo 수 | demo_0 액션 / states | 시간 |
|---|---|---|---|---|
| 0 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5 | 50 | 98 / (98, 92) — 서버 기록과 같음 | 72s |
| 4 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5 | 50 | 153 / (153, 92) | 63s |
| 5 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5 | 50 | 137 / (137, 92) | 63s |

→ `$DART_ROOT/demos/spatial_t{0,4,5}_hdf5_demo0-4.npz`. 404 없음.

### 1-2. Panda 재생 (`run_oracle_t45.sh panda`, 08:02:39 → 08:02:58, 19s, 6개 동시)
`make_demo_pair.py --robots Panda --task <N> --demo demo_<k> --hdf5-npz <npz> --out $DART_ROOT/demos/oracle_t<N>/demo<k>` × 10, 전부 rc=0.

| task | 성공 (끝에서) | 첫 성공 스텝 demo0~4 | 액션 수 |
|---|---|---|---|
| t4 | **5/5** | 117 110 150 117 123 | 153 142 158 129 143 |
| t5 | **5/5** | 123 103 111 98 91 | 137 131 121 121 119 |

### 1-3. 시작 장면 (`run_oracle_t45.sh init` → `oracle_init_view.py` 신규, 9s)
`results/oracle_t45/init_t4.png`, `init_t5.png` (왼쪽 Panda, 오른쪽 xArm7, set_init_state 직후, 모델 입력 방향 180° 회전), 같은 이름 `.json`.
- IK 오차 t4 위치 8.3e-7 / 회전 1.2e-6, t5 5.6e-7 / 2.1e-6. 물체 조인트 9개 복사, 누락 0. 눈으로 보면 물체 배치가 같음 (agentview 픽셀 차 평균 4.8 / 4.4 — 로봇 부분).

**STOP1 자동 통과 기준 (task 마다 Panda 재생 ≥ 4/5): t4 5/5, t5 5/5 → 통과 → 본 실행 진행.**

## 2. 본 실행 — xArm7 retarget 20개 (`run_oracle_t45.sh retarget`, 08:03:42 → 08:04:41, 59s, 6개 동시)
`make_retarget_demo.py --task <N> --demo demo_<k> --hdf5-npz <npz> --out $DART_ROOT/demos/oracle_t<N>/demo<k> --rot-clip {quantile,none}`, 전부 rc=0 (개당 14~16s).
출력 `..._XArm7_retarget_rotq.npz` (quantile) / `..._XArm7_retarget.npz` (none) + `_summary.json`. 성공 = 액션을 다 실행한 끝에서 `_check_success()`.

## 3. 표 (5번, `oracle_t45_summary.py` 신규 → `results/oracle_t45/summary.{md,json}`)

| task | 실행 | 성공/5 | 첫 성공 스텝 (demo0~4) | 추종오차 위치 평균/최대 cm | 회전 평균/최대 ° | ±1 클립 스텝 (demo0~4) | q01~q99 회전 클립 스텝 |
|---|---|---|---|---|---|---|---|
| t0 기준 (DEMO_PAIR) | retarget quantile | 성공 | 73 | 4.0 / 6.0 | 5.9 / 17.1 | 27 | (33, 노트북 재현에서 셈) |
| t0 노트북 | retarget none | 성공 | 74 | 4.0 / 6.0 | 4.9 / 14.1 | 27 | – |
| t4 | Panda 재생 | 5/5 | 117 110 150 117 123 | – | – | – | – |
| t4 | **retarget quantile** | **2/5** | – 107 – 117 – | 3.6 / 10.2 | 9.4 / 24.1 | 26 36 29 34 42 | 102 77 112 60 103 |
| t4 | **retarget none** | **2/5** | – 108 – 114 – | 4.3 / 12.4 | 6.9 / 16.3 | 32 38 39 34 53 | – |
| t5 | Panda 재생 | 5/5 | 123 103 111 98 91 | – | – | – | – |
| t5 | **retarget quantile** | **3/5** | 121 105 – 100 – | 3.5 / 11.0 | 5.4 / 21.7 | 10 19 14 33 29 | 74 37 82 24 37 |
| t5 | **retarget none** | **3/5** | 117 104 112 – – | 3.8 / 11.0 | 4.3 / 14.4 | 17 23 25 47 31 | – |

- 추종오차 평균 = demo 별 평균의 평균, 최대 = 5개 중 최대. "±1 클립" = 자르기 전 액션 중 한 차원이라도 |a|>1 인 스텝 (summary 의 `steps_with_clipped_action`).
  "q01~q99 회전 클립" = quantile 자르기로 회전 액션이 바뀐 스텝 (npz 의 `rot_raw` 와 `actions` 비교, t0 재현 33/98 = EXP6_LOG 1-3 의 34% 와 일치).
- 성공한 demo 는 두 rot-clip 이 같음 (t4 demo1·3, t5 demo0·1). 엇갈린 건 t5 demo2 (quantile 실패 / none 성공) 와 t5 demo3 (quantile 성공 / none 실패) 둘뿐.
- t4 는 quantile 회전 클립이 매우 많음 (액션 수의 47~72%, t0 34%) — 서랍에 넣을 때 회전 보정이 큼. 하지만 성공 수는 none 과 같음.

## 4. 실패 분석 (6번, `oracle_fail_analysis.py` 신규, 실패 10개, 개당 ~37s, 5개 동시 75~78s)

방법: 같은 시작 장면에서 저장된 액션을 다시 재생하며 그릇(`akita_black_bowl_1`)·접시(`plate_1`) 상태를 기록. Panda 원본 데모도 같이 재생해 기준으로 씀.
재생 = 원래 실행: 매 스텝 eef 차이 xArm7 ≤6e-9m (8개), 1.7mm (t4_demo4_quantile), Panda ≤3e-5m, 0.9mm (t5_demo2) — npz 액션이 float32 라 접촉에서 조금 벌어짐. **성공 여부는 10개 모두 summary 와 같음.**
산출물: `results/oracle_t45/fail/<name>.json` (사건·튀는 스텝), `_strip.png` (위 Panda / 아래 xArm7 같은 스텝), `_ts.png` (pad_gap·추종오차·그릇 높이·접시 거리), `_replay.npz`,
영상 `results/oracle_t45/videos/<name>.mp4` ([agentview | 손목], 10fps, 스텝·pad_gap·추종오차·파지 표시, git 제외).

**판정 규칙** (첫 번째로 어긋난 단계, 데이터를 본 뒤 정함 — 사후 기준): 파지 성립 = 양 패드가 그릇에 닿은 스텝 ≥10 이고 그릇이 3cm 이상 들림.
"기운 채 잡힘" = 접촉 중 그릇 기울기 > max(Panda +10°, 30°). 파지 성립 후 접시 8cm 밖에서 놓치면 들기, 안에서면 놓기. 10개 모두 프레임 띠를 보고 같은 판정 확인.

| 실패 | 실패 지점 | 근거 (xArm7 / Panda) | 추종오차 튀는 스텝 (>평균+2σ) | pad_gap |
|---|---|---|---|---|
| t4 demo0 quantile | 파지(빈손으로 닫힘) | 닫는 순간 grip_site 가 그릇 중심 위 6.0 / 4.1cm, 거리 8.7 / 6.2cm. 양 패드 접촉 3스텝 (58~60), 그릇 0.4cm 들림 / 12cm | 위치 25~26 (서랍 접근 중, 6.7cm), 회전 142~152 (끝, 빈손) | 튐 없음. 닫힘 최소 3.0mm (빈손) |
| t4 demo0 none | 파지(빈손으로 닫힘) | 높이 7.2 / 4.1cm, 거리 10.4 / 6.2. 접촉 4스텝, 0.4cm | 회전 138~146 (끝) | 없음, 최소 2.8mm |
| t4 demo2 quantile | 파지(기운 채 잡힘) | 높이 비슷 (2.6 / 3.0) 이나 수평 5.8 / 4.6cm. 접촉 8스텝 동안 그릇 34.5° 기울어짐 (Panda 10.5°), 1.5cm 들림 후 빠짐 | 위치 37~38 (접근) | 없음 |
| t4 demo2 none | 파지(빈손으로 닫힘) | 높이 2.3 / 3.0, 거리 6.1 / 4.6cm. 접촉 7스텝, 0.8cm, 기울기 18° | 위치 37~38 (접근) | 없음 |
| t4 demo4 quantile | 파지(빈손으로 닫힘) | 높이 6.2 / 4.3cm (거리 8.7 / 8.6). 접촉 9스텝 (57~65), 1.1cm | 위치 72~79 (빈손으로 이동) | 없음 |
| t4 demo4 none | 파지(빈손으로 닫힘) | 높이 6.1 / 4.3cm. 접촉 9스텝, 0.9cm | 위치 73~84 (빈손으로 이동, 최대 12.4cm) | 없음 |
| t5 demo2 quantile | 파지(기운 채 잡힘) | 접촉 5스텝 (44~54) 동안 그릇 51.5° 기울어짐 (Panda 38.1°), 2.4cm 들림 → 램킨 옆 탁자에 56° 기운 채 떨어짐 | 위치 26~33 (접근), 회전 97~103 (빈손) | 없음 |
| t5 demo3 none | 놓기 | 66스텝 쥐고 14.3cm 들어 접시에 놓음 (접시와 접촉) 그러나 중심에서 **3.6cm** (성공 술어 On 은 xy < 3cm; Panda 1.2cm) | 위치 94~101, 회전 93~101 (놓기 구간, 최대 10.2cm) | 없음 |
| t5 demo4 quantile | 놓기 | 54스텝 쥐고 접시 중심 1.4cm 까지 옮김. 끝에 **그릇이 램킨 위에** 얹혀 접시와 비접촉 (램킨이 그릇과 같이 8.4cm 들려 30.6cm 이동) | 위치 88~93, 회전 87~94 (놓기) | 없음 |
| t5 demo4 none | 놓기 | 같음 (램킨 31.1cm 이동, 그릇-접시 xy 1.6cm, 접시 비접촉) | 위치 88~93, 회전 89~92·100~103 | 없음 |

- **분포:** 파지(빈손으로 닫힘) 5 (전부 t4), 파지(기운 채 잡힘) 2 (t4 1, t5 1), 놓기 3 (t5), 접근 0, 들기 0.
- **t4 공통:** 서랍 안 그릇을 Panda 보다 높거나 옆에서 집어 테두리만 잠깐 걸렸다가 들어 올리기 시작할 때 빠짐. 그리퍼-캐비닛 접촉은 Panda 도 같이 있어 (닫기 전후) xArm7 만의 충돌 증거는 없음.
  demo0·4 는 닫는 순간 grip_site 가 Panda 보다 1.8~3.1cm 높음 (목표 깊이에 못 미침), demo2 는 높이는 비슷하고 수평으로 1.2~1.5cm 벗어남.
  위치 추종오차 최대는 demo0·2 에서 서랍으로 내려가는 구간 (닫기 22~24스텝 전), demo4 에서는 빈손으로 이동하는 구간.
- ~~t5 demo4 는 Panda 원본 재생에서도 램킨이 같이 들림~~ → **틀림 (맨 위 정정)**. 램킨이 같이 들려 옮겨지는 건 xArm7 에서만. 끝에 그릇의 접촉 상대가 램킨뿐인 것은 맞음.
- **pad_gap 튐:** 10개 모두 없음 (그리퍼 명령이 일정한데 한 스텝에 2mm 넘게 변한 곳, 명령 변경 후 15스텝 제외). t4 xArm7 닫힘 최소 2.8~3.5mm (빈손 수준), 같은 데모의 Panda 최소 3.6~6.2mm.
- 액션 뒤 20스텝 더 기다려도 (그리퍼 열림) 늦게 성공한 경우 0 (`success_after_extra_wait`).

## 5. θ0 실패 유형 (7번, 선택) — 건너뜀
노트북에 `$DART_ROOT/eval/theta0/videos/` 가 없고 (서버 휘발성), θ0 재평가는 pi05 정책 서버(GPU)가 필요해 8GB 노트북 GPU 로는 하지 않음 (사용자에게 실행 위치를 물을 때 미리 알림).
참고로 기존 기록 (EVAL_RESULTS.md, 서버 영상 판정): θ0 t5 = "파지 실패 → 빈손으로 접시까지 이동 후 맴돎", θ* t4 = "서랍 안에서 팔이 꺾인 자세로 재파지 반복", θ* t5 = "기운 채 잡혀 접시 가장자리에 떨어뜨림".
→ 정답 궤적 실패 유형 (t4 서랍 안 파지, t5 기운 파지) 과 겉보기에 닮음. 같은 기준으로 센 것은 아님.

## 6. 확인용 화면 (사용자 요청)
- 실패 영상 10개 VLC 재생목록.
- `oracle_viewer.py` (신규): MuJoCo 대화형 뷰어 (GLFW 창) 로 저장된 액션을 20Hz 실시간 재생, `--loop`. 카메라 관측 없는 `ControlEnv` 라 물리는 녹화와 같음.
  `MUJOCO_GL=glfw $LIBERO_PY oracle_viewer.py --task 4 --demo 0 --robot {Panda|XArm7} --variant quantile --loop` (t4 demo0 두 창 실행).

## 7. 이상 징후
- 첫 check0 실행 3개 rc=1: venv 에 `future` 없음 (bddl 이 import) → `future==1.0.0` 설치 후 재실행 rc=0 (실패한 실행은 출력 파일을 안 만들어 덮어쓰기 없음).
- 재생 eef 차이 1.7mm (t4_demo4_quantile xArm7), 0.9mm (t5_demo2 Panda) — float32 액션 저장 탓으로 보임 (추정). 성공 판정은 동일.
- 성공 술어 `On` (LIBERO `check_ontop`) = 접시와 접촉 + xy 3cm 미만 + 그릇 z ≤ 접시 z. retarget 위치 추종오차 (평균 3.5~4.3cm) 와 같은 크기라 놓기 실패가 여기서 갈림 (t5 demo3: 3.6cm).
- 서버와 다른 환경 (노트북, EGL, venv 재구성). t0 재현은 표시 자릿수까지 같았지만 비트 단위 비교는 못 함 (서버 npz 없음).

## 8. 산출물 / 재실행

| 종류 | 경로 |
|---|---|
| 원본 hdf5 일부 | `~/dart_laptop/demos/spatial_t{0,4,5}_hdf5_demo0-4.npz` |
| Panda 재생·retarget npz·summary | `~/dart_laptop/demos/oracle_t{4,5}/demo{0..4}/` (t4 280M, t5 257M), t0 재현 `~/dart_laptop/demos/oracle_t0_check/demo0/` |
| 요약 (홈, 커밋) | `results/oracle_t45/summary.{md,json}`, `init_t{4,5}.{png,json}`, `fail/` (json·strip·ts·replay npz) |
| 영상 (커밋 안 함) | `results/oracle_t45/videos/*.mp4` |
| 실행 로그 | `~/dart_laptop/logs/oracle_t45/` (`status.txt`, 실행별 `.log`) |

신규: `env_laptop.sh`, `run_oracle_t45.sh`, `oracle_init_view.py`, `oracle_t45_summary.py`, `oracle_fail_analysis.py`, `oracle_viewer.py`, `ORACLE_T45_LOG.md`.
수정: `fetch_hdf5_demo.py` (인자 추가, 기본 동작 동일). `make_demo_pair.py`·`make_retarget_demo.py`·dart 저장소·LIBERO-view 는 수정 없음.

```bash
source ~/hayeon/dart_xarm7/env_laptop.sh           # 서버면 run_oracle_t45.sh 가 env.sh + client_env.sh 를 씀
for T in 4 5; do $LIBERO_PY ~/hayeon/dart_xarm7/fetch_hdf5_demo.py --task $T; done
bash ~/hayeon/dart_xarm7/run_oracle_t45.sh check0 && bash ~/hayeon/dart_xarm7/run_oracle_t45.sh panda
bash ~/hayeon/dart_xarm7/run_oracle_t45.sh init && bash ~/hayeon/dart_xarm7/run_oracle_t45.sh retarget
$LIBERO_PY ~/hayeon/dart_xarm7/oracle_t45_summary.py
source ~/hayeon/dart_xarm7/exp6_cond.sh && exp6_env v0 && $LIBERO_PY ~/hayeon/dart_xarm7/oracle_fail_analysis.py --task 4 --demo 0 --variant quantile
```
demo_5~9 를 하려면: `fetch_hdf5_demo.py` 는 `names[:5]` (demo_0~4) 만 저장 → 개수 인자 추가 후 새 npz 이름으로 받고, `DEMOS="5 6 7 8 9"` 로 러너 실행 (러너의 npz 경로도 맞춰야 함).

---

## 9. demo_5~9 추가 (사용자 요청, 08:25 → 08:29)
- `fetch_hdf5_demo.py` 에 `--demos a-b` 추가 (기본 0-4 = 이전 `names[:5]` 와 같음, 출력 이름 `..._hdf5_demo<a>-<b>.npz`). `--task {4,5} --demos 5-9` 각 58·65s.
  액션 수 t4 158 167 156 159 132, t5 100 101 129 123 118.
- 러너: 원본 npz 를 demo 5개 단위로 고름 (`npz <task> <demo>`). `oracle_viewer.py`·`oracle_fail_analysis.py` 도 같은 규칙.
- `DEMOS="5 6 7 8 9" run_oracle_t45.sh panda` (08:27:14 → 08:27:35) → `retarget` (08:27:35 → 08:28:41), 30개 rc=0.
- `oracle_t45_summary.py` 를 demo 0~9 로 확장 (0~4 / 5~9 / 합계, "한 번이라도" 열).

| task | 실행 | 0~4 | 5~9 | 합계 (끝) | 한 번이라도 |
|---|---|---|---|---|---|
| t4 | Panda 재생 | 5/5 | 4/5 (demo6 실패) | 9/10 | 9/10 |
| t4 | retarget quantile | 2/5 | 3/5 | **5/10** | 5/10 |
| t4 | retarget none | 2/5 | 0/5 | **2/10** | 4/10 (demo7·9 는 도중 성공 후 끝에 실패) |
| t5 | Panda 재생 | 5/5 | 5/5 | 10/10 | 10/10 |
| t5 | retarget quantile | 3/5 | 2/5 | **5/10** | 5/10 |
| t5 | retarget none | 3/5 | 2/5 | **5/10** | 5/10 |

추종오차 (위치 평균/최대 cm, 회전 평균/최대 °): t4 quantile 3.5/10.2, 10.6/28.2 · t4 none 5.8/37.0 (demo5 가 17.3/37.0 으로 궤적 이탈), 7.3/17.3 · t5 quantile 3.6/11.7, 4.9/21.7 · t5 none 3.9/12.0, 4.1/15.8 (t0 4.0/6.0, 5.9/17.1).

## 10. 반복성 (같은 retarget 을 다시 돌리면 같은가)
- 계기: 실패 재생에서 t4 demo8 none 이 원래 실행과 0.35m 벌어짐 (float64 액션을 똑같이 복원해도). 같은 retarget 을 별도 폴더에서 두 번 다시 돌리니 둘은 서로 같고 원래 실행과는 40스텝부터 갈라짐
  (0스텝부터 1e-19 차이, 39스텝 1e-12 → 40스텝 접촉에서 1mm → 이후 수십 cm). 스레드 1개로 돌린 것도 또 다름 → 멀티스레드 OpenBLAS 반올림 + CPU 부하 탓으로 추정.
- 러너에 `ROOT=<다른 루트>` 추가 (Panda npz 는 기본 폴더 것을 링크, 기존 결과 덮어쓰기 없음).
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ROOT=$DART_ROOT/demos/oracle_rep/omp1_{a,b} DEMOS="0..9" P=7 run_oracle_t45.sh retarget` (08:41:04 → 08:45:02, 80개 rc=0).
- 결과 (`results/oracle_t45/repeat_runs.json`):
  - **성공/실패: 원래 실행 · omp1_a · omp1_b 가 40개 모두 같음** (끝 성공·첫 성공 스텝 패턴까지). 위 표의 숫자는 실행에 따라 안 바뀜.
  - omp1_a vs omp1_b: 물리 (eef·액션) 40/40 비트 동일. 이미지는 26/40 만 동일 (EGL 렌더 차이, 물리 무관).
  - 원래 실행 vs omp1: eef 차이 39개 ≤ 2.2mm, t4 demo8 none 만 18.8cm.
- → 재현이 필요하면 `OMP_NUM_THREADS=1`. 실패 분석·영상은 omp1_a 기준으로 다시 만듦 (11절).

## 11. 실패 분석 갱신 (08:46 → 08:51, 23개, `--src-root $DART_ROOT/demos/oracle_rep/omp1_a`, OMP_NUM_THREADS=1)
`oracle_fail_analysis.py` 수정:
- 재생 액션을 원래 실행과 같은 float64 로: Panda = hdf5 원본 액션, xArm7 = npz `pos_raw`·`rot_raw` 를 make_retarget_demo.py 와 같은 식으로 다시 자름 (npz `actions` 와 float32 로 같은지 assert).
  → **23개 모두 재생 eef 차이 0, 성공 여부 summary 와 같음.**
- 그릇 말고 2cm 넘게 움직인 물체 (`moved_objects_cm`), 추가 대기 뒤 그릇의 접촉 상대 (`bowl_contacts`, 패드 외 그리퍼 = gripper_body, 팔 = arm).
- 판정 규칙 보강 (+ 원인 태그 `cause_tags`): 도중 성공 → 놓기 ("성공 후 놓침"). 접시 8cm 안까지 못 가져감 → 기울기 > max(Panda+10°, 30°) 면 파지(기운 채),
  양 패드 접촉 < 10스텝 이면 파지(빈손), 그 밖은 들기 ("못 들어 올림"/"옮기다 놓침"). 가져갔는데 실패 → 놓기 ("3cm 밖", "같이 옮겨짐: 램킨", "기운 채 운반"). Panda 도 실패면 "비교 제외".
- 비교 영상 `<name>_cmp.mp4` ([Panda | xArm7] 같은 스텝). 영상은 임시 파일에 쓴 뒤 이름 바꾸기 (재생 중 플레이어 보호).
- 프레임으로 직접 본 것 10개 (유형마다 1개 이상): t4 demo0 q·demo2 q·demo4 n·demo5 n·demo8 n·demo9 n, t5 demo2 q·demo4 n·demo6 n·demo8 q — 모두 규칙 판정과 같음.
  경계: t4 demo4 는 양 패드 접촉 none 10스텝 (들기) / quantile 8스텝 (파지).

| 실패 지점 | t4 rot-clip | t4 없음 | t5 rot-clip | t5 없음 | 합계 |
|---|---|---|---|---|---|
| 파지(빈손으로 닫힘) | 2 (d0, d4) | 2 (d0, d2) | 1 (d6) | 1 (d6) | 6 |
| 파지(기운 채 잡힘) | 2 (d2, d8) | 1 (d8) | 1 (d2) | 0 | 4 |
| 들기 | 0 | 2 (d4, d5 — 서랍에서 못 들어 올림) | 0 | 0 | 2 |
| 놓기 | 0 | 2 (d7, d9 — 성공 후 그릇 넘어짐) | 3 (d4, d8, d9 — 램킨째) | 4 (d3 3.6cm, d4·d8·d9 램킨째) | 9 |
| (Panda 도 실패, 제외) | 1 (d6) | 1 (d6) | – | – | 2 |

- t4 파지 실패 공통: 닫는 순간 grip_site 가 Panda 보다 높거나 (d0·d4 약 2~3cm) 옆 (d2). 양 패드 접촉 빈손 3~8스텝, 기운 채 8~24스텝 (Panda 72~89). 기운 채 (d8) 는 들면서 그릇이 약 50° 기울어 캐비닛 위에 떨어짐.
- t5 램킨째 6개: 램킨이 28~32cm 같이 옮겨지고 그릇이 그 위에 얹혀 접시와 안 닿음 (끝 접촉 상대 = 램킨). 같은 demo 의 Panda 재생은 램킨 이동 < 2cm.

## 12. 결과 페이지·노션 (사용자 요청)
- 아티팩트 (비공개): https://claude.ai/artifact/GXUT3GSbitqH32VhepjPiM — 판정, demo 격자 (칸 누르면 영상), 실패 영상 23개 플레이어 ([Panda | xArm7] · [xArm7 | 손목] · 그래프, 필터, ← → 이동), 분포, 추종오차, 시작 장면.
  빌더는 세션 임시 폴더라 레포에 없음. 영상 원본: `results/oracle_t45/videos/` (git 제외).
- 노션 [4-후속1] 페이지 끝에 "5. 데모 MuJoCo 실행 명령" 절 추가 (oracle_viewer.py 사용법·두 창 비교·VLC·npz 다시 만들기).

신규·수정 (9~11절): 수정 `fetch_hdf5_demo.py` (`--demos`), `run_oracle_t45.sh` (demo 블록, `ROOT`), `oracle_t45_summary.py` (demo 0~9, 한 번이라도),
`oracle_fail_analysis.py` (`--src-root`, float64 재생, 접촉·이동 물체, 규칙·태그, 비교 영상), `oracle_viewer.py` (demo 블록) / 결과 `results/oracle_t45/{summary.*,repeat_runs.json,fail/}`.
