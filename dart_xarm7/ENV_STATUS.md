# DART (Panda→xArm7) 환경 상태 메모 — 2026-10-07

## 한눈에

| 항목 | 상태 |
|---|---|
| GPU | 잡 **129138** (`coss_agpu`, 노드 `ece-agpu15`), **A100-SXM4-80GB 1장**, CPU 6, 노드 RAM 2TB |
| 저장소 | `/var/tmp/pbs.129138.ECE-util1/dart/dart` — snumprlab/dart @`1b23c4f` (서브모듈 aloha d1dc83a, libero f78abd6, libero-view 2e8af0f), 1.6G |
| venv | `$DART_REPO/.venv` — uv, Python 3.11.15, jax 0.5.3 (`CudaDevice(id=0)` 인식), flax 0.10.2, torch 2.7.1+cu126, transformers 4.53.2, 240개 패키지, 7.7G |
| θ0 체크포인트 | `$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero` — **11.58 GiB** (12,439,085,481 B, GCS 와 일치), fp32 3.35B params, norm_stats 포함 |
| 토크나이저 | `$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model` |
| 환경 변수 | `source ~/hayeon/dart_xarm7/env.sh` |
| 홈 사용량 | **27.0 GiB / 쿼터 ≈28GB (추정)** — 여유 1GB 미만. 이 중 `~/.singularity/cache` 20G |
| `/var/tmp` 여유 | 셋업 후 **36G** (406G 디스크, 같은 노드 다른 사용자 잡 6개와 공유) |

모든 큰 파일은 `/var/tmp/pbs.129138.ECE-util1/dart/` 아래에 있고 **휘발성**이다 (잡이 끝나면 사라질 수 있음 — 확인 필요).

### 2026-10-08 추가 — LIBERO 평가 클라이언트 (xArm7 교체 작업, 자세한 건 XARM7_NOTES.md)

| 항목 | 상태 |
|---|---|
| 평가 venv | `$DART_ROOT/venvs/dart-libero` — `/usr/bin/python3 --system-site-packages --without-pip`, `~/venvs/dart-libero-requirements.txt` 그대로 (robosuite 1.4.1, mujoco 3.2.3), 735M. torch 는 이미지 것 그대로 확인 |
| 렌더러 | `$DART_ROOT/osmesa` — conda-forge `mesalib 24.3.4` (`/tools/anaconda3/bin/conda create -p`, `CONDA_PKGS_DIRS` 노드 로컬). **EGL 불가** (graphics capability 없음) |
| 환경 변수 | `source env.sh && source client_env.sh` (MUJOCO_GL=osmesa, LIBERO_CONFIG_PATH, PYTHONPATH 에 `src/`·libero-view) |
| xArm7 | `src/libero_xarm7/` — `import libero_xarm7` 후 `robots=["XArm7"]` |
| **`/var/tmp` 여유** | **8.0G (2026-10-08)** — 다른 사용자 잡들이 하루 새 ~28G 더 씀. 학습 계획(≈18G 필요)은 이 노드에서 지금은 불가 |

conda 로 설치하면 `conda-pkgs/cache` 에 conda-forge 인덱스가 1.5G 쌓인다 → `conda clean -y --index-cache --tarballs --packages` (설치된 prefix 는 영향 없음).

## 셋업 로그 요약 (logs/setup_20261007/)

1. **clone** — 성공. 홈 쿼터 때문에 홈이 아닌 노드 로컬에 클론함 (서브모듈 포함 1.6G).
2. **GPU 확보** — 이미 잡 129138 안에서 작업. A100 1장 확보, 거절 없음.
   단, 컨테이너 안에서는 `qstat`/`pbsnodes` 가 `cannot connect to server` 로 실패함 (PBS 인증이 컨테이너 안에서 안 되는 것으로 보임 — 추정).
   다른 A100 잡이 있는지와 계정 한도는 **로그인 노드에서 `qstat -u introai34` 로 확인 필요.**
   이 노드의 `/var/tmp` 에는 이 계정의 잡 디렉터리가 129138 하나뿐 (실험 2 잡 128706 디렉터리는 없음).
3. **uv sync** — 성공 (3분 7초). uv 0.11.8 은 `~/.local/bin/uv` 에 이미 있어 새로 설치하지 않음.
   `uv pip install -e .` 도 성공 (openpi editable 재설치, `uv sync` 가 이미 editable 로 깔기 때문에 사실상 중복).
   2026-10-04 freeze 와 비교: 핵심 패키지 동일, 이번엔 `--no-dev` 없이 해서 dev 그룹(pytest, ruff, ipykernel, matplotlib 등) 추가.
4. **체크포인트** — openpi 자체 다운로더로 성공 (7분 34초, ~29MB/s). gsutil 이 없어 gcsfs 로 넘어가고 경고가 뜨지만
   익명 접근으로 정상 동작. CPU 에서 `restore_params` 로 51개 배열 복원 확인.
5. **임포트** — `openpi`, `openpi_client`, `domain_arithmetic`, `jax`, `flax`, `torch`, `transformers`, `lerobot` 전부 성공.
   `get_config('pi05_libero_oneshotft')` OK (config 기본 batch 32 — finetune.sh 가 64 로 덮어씀, `fsdp_devices=1`).

시스템 패키지(apt) 를 요구하는 에러는 없었음. git-lfs 는 없지만 `GIT_LFS_SKIP_SMUDGE=1` 이라 문제 없음.

## exec/finetune.sh 가 받는 환경 변수

`cd` 로 저장소 루트로 이동한 뒤 `uv run scripts/train.py $CONFIG ...` 를 실행한다 (→ `$DART_REPO/.venv` 사용).

| 변수 | 기본값 | 쓰이는 곳 / 주의 |
|---|---|---|
| `DATA_REPO_ID` | **필수** | `--data.repo-id`. LeRobot 형식 LIBERO 데이터셋 경로 또는 HF repo id |
| `CHECKPOINT_BASE_DIR` | **필수** | `--checkpoint-base-dir`. 실제 저장 위치 = `$CHECKPOINT_BASE_DIR/$CONFIG/$CAMERA_VIEW/999/` |
| `CAMERA_VIEW` | `original` | `--data.camera-view` + `--project-name` + `--exp-name`. 데이터셋 이미지 키 선택: original→`image`, left_back→`left_back_image`, left, small, medium, large |
| `CONFIG` | `pi05_libero_oneshotft` | θ0 = `gs://openpi-assets/checkpoints/pi05_libero/params` 에서 시작, lr 5e-5 상수, AdamW |
| `BATCH_SIZE` | 64 | `--batch-size` |
| `NUM_TRAIN_STEPS` | 1000 | 스텝 0..999, 마지막 스텝 999 에서 1회 저장 |
| `SAVE_INTERVAL` / `KEEP_PERIOD` | 1000 / 1000 | `max_to_keep=1` |
| `WANDB_MODE` | `online` | wandb 로그인 안 되어 있음 → `offline` 필요 (env.sh 가 설정) |
| `WANDB_ENTITY` | `domain-arithmetic` | 저자 팀 entity. online 으로 쓸 거면 내 entity 로 바꿀 것 |
| `CUDA_VISIBLE_DEVICES` | `0,1` | 런처가 이미 GPU UUID 로 설정 → 그 값 유지 |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | 0.9 | OOM 이면 0.95 까지 올려볼 수 있음 |

스크립트가 노출하지 않는 것: `--fsdp-devices` (플랜 B A6000×2 이면 명령에 직접 추가), `--overwrite`/`--resume`,
`OPENPI_DATA_HOME`·`HF_LEROBOT_HOME` (env.sh 로 설정).

## 다음 작업 전에 알아둘 위험

- **체크포인트 1개 ≈ 35GiB (추정).** `params`(fp32 ≈11.6GiB) 와 `train_state`(AdamW mu/nu 가 fp32 → ≈23GiB) 를 함께 저장한다.
  src/tgt 두 번이면 ≈70GiB 인데 `/var/tmp` 여유는 36G. DART 는 `params` 만 필요하므로
  **각 FT 가 끝나면 `train_state/` 를 지우고 `params/` 만 남기는 방식**이 필요 (첫 실행 후 실제 크기 확인).
- **bf16 저장은 결과에 영향 없음 — 검증함 (2026-10-07).** DART(`load_policy_params`)와 평가 서버(`create_trained_policy`)는
  `restore_params(dtype=bfloat16)` 로 읽고, Δ 도 `w_src - w_base` 를 bf16 상태에서 계산한다 (`domain_arithmetic/dart.py:128`).
  θ0 를 같은 호출로 bf16 변환·저장한 뒤 DART 로더로 원본과 비교 → 51개 배열, 3.35B 개 값 **비트 단위 동일**.
  bf16 체크포인트 실측 4.9G (fp32 11.6G). 스크립트 `bf16_equiv_test.py` (실행 시 `PYTHONPATH=$DART_REPO` 필요), 로그 `logs/setup_20261007/bf16_equiv_test.log`.
  `train_state` 는 `--resume` 때만 읽힌다. SAVE_INTERVAL=NUM_TRAIN_STEPS=1000 이면 저장이 마지막 스텝 한 번뿐이라 원래도 중간 재개에는 못 씀.
- **src 와 tgt 실행이 같은 경로에 저장된다.** xArm7 데이터도 `image` 키(original 시점)를 쓰면 두 실행 모두 `CAMERA_VIEW=original` →
  `exp_name` 이 같아져 두 번째 실행이 "already exists" 로 멈춘다. 실행마다 `CHECKPOINT_BASE_DIR` 을 다르게 줄 것.
- **`uv run` 은 실행할 때마다 lockfile 과 동기화한다.** `UV_CACHE_DIR` 없이 돌리면 캐시가 홈에 쌓이니 항상 env.sh 를 먼저 source.
- **메모리:** full FT 는 params+grads+Adam 만 ≈54GB(fp32 3.35B 기준 계산값) + 배치 64 활성값. 80GB 에서 여유가 적음.

## 새 잡에서 복원 (전부 휘발성이므로)

```bash
source ~/hayeon/dart_xarm7/env.sh            # 잡 ID 에 맞춰 경로가 바뀜
mkdir -p $DART_ROOT && cd $DART_ROOT
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/snumprlab/dart.git && cd dart
git checkout 1b23c4f && git submodule update --init --recursive
uv sync --python 3.11                        # ≈3분
uv run python -c "from openpi.shared import download as d; d.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')"   # ≈8분, 11.6GiB
```
