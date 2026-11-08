# 실험 4-후속3 (a5) — 적응 task 를 t0 → t5 로 바꾼 DART

질문: t5(ramekin 위 그릇) demo_0 쌍으로 만든 θ*_t5 가 xArm7 t5 성공률을 θ0 보다 올리나, 나머지 9 task 는 지키나.
조건: exp6_env v0 (XARM7_COLOR=white XARM7_DYNAMICS=panda), seed 7, α=0.5 고정. 새 산출물은 전부 a5 경로·a5_ 태그.

- **실험 문장 빈칸 상태로 시작** — 사용자 지시 ("빈칸 무시하고 시작", 2026-11-08). 판정 기준은 프롬프트의 STOP1 자동 통과 기준·STOP2 H_fix/H_nofix/trade-off 를 따름.
- 이 잡(A6000 46GB)에서는 STOP1 까지만. full FT 는 ≈54GB 필요 (ENV_STATUS.md 메모리 항목) → A100 잡에서.

## 0. git (2026-11-08 13:52, 잡 133746, ece-a6gpu7 컨테이너)

- 컨테이너에서는 SSH push/pull 불가 (`No user exists for uid 4001`) → 공개 레포를 HTTPS 로 fetch 후 `git rebase FETCH_HEAD`.
- 로컬 미push 커밋 2개 (실험 6 단계 4, 실험 7 파일럿) 를 노트북 커밋 2개 (e599613, 179a6ce — oracle t4·t5) 위로 옮김. 겹치는 파일 0, 충돌 없음.
- **push 는 로그인 노드에서 `git push` 필요** (로컬이 origin/main 보다 4개 앞으로 보이는 건 origin 참조가 옛것이라서).

## 1. 환경 복원 (restore_env.sh)

- 2026-11-08 13:52 시작, `setsid nohup bash ~/hayeon/dart_xarm7/restore_env.sh`. DART_ROOT=/var/tmp/pbs.133746.ECE-util1/dart (/dev/shm 링크 없음, /var/tmp 여유 99G).
- 13:52:17 → 14:12:39 ALL_DONE (20분). repo 42s, uv_sync 169s, osmesa 731s, hf_ckpts 581s (v0 oneshot_src/tgt·dart_a0.5), libero_venv 873s, theta0 1011s. 끝 /var/tmp 여유 52G, 사용 47G.
- 이상 징후: pip 다운로드 ~100kB/s (θ0·HF 동시 다운로드와 대역폭 경쟁) → libero_venv 가 평소보다 느림.
- **설치 추가 (2026-11-08, 사전 안내 후 사용자의 "시작" 지시로 진행 — 명시적 승인은 아님):** libero venv 에 `h5py==3.16.0` (`pip install --no-cache-dir -c ~/venvs/dart-libero-constraints.txt h5py`).
  노트북판 fetch_hdf5_demo.py 가 task≠0 이면 `$LIBERO_PY` 를 요구하는데 서버 venv·이미지에 h5py 가 없었음. numpy 1.22.2·torch 2.1.0a0+32f93b1 은 이미지 경로 그대로 (확인).
  `~/venvs/dart-libero-requirements.txt` 에 한 줄 추가 → 다음 잡부터 restore_env.sh 가 같이 설치.

## 2. STOP1 (2026-11-08 14:11 → 14:19, 잡 133746, ece-a6gpu7)

env: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, env.sh + client_env.sh (MUJOCO_GL=osmesa), `exp6_env v0` → XARM7_COLOR=white XARM7_DYNAMICS=panda.

| 단계 | 명령 | 시간 | 산출물 |
|---|---|---|---|
| 1 | `$LIBERO_PY fetch_hdf5_demo.py --task 5 --demos 0-4` | 58s | `$DART_ROOT/demos/spatial_t5_hdf5_demo0-4.npz` (demo 0~4 길이 137/131/121/121/119) |
| 2a | `$LIBERO_PY make_demo_pair.py --robots Panda --task 5 --demo demo_0 --hdf5-npz <1> --out $DART_ROOT/demos/a5` | 58s | `demos/a5/libero_spatial_t5_demo_0_Panda.npz`, `_summary.json` (로그 `logs/a5/panda.log`) |
| 2b | `$LIBERO_PY make_retarget_demo.py --rot-clip quantile --task 5 --demo demo_0 --hdf5-npz <1> --out $DART_ROOT/demos/a5` | 107s | `demos/a5/libero_spatial_t5_demo_0_XArm7_retarget_rotq.npz`, `_summary.json` (로그 `logs/a5/retarget_rotq.log`) |
| 3 | `cd $DART_REPO && uv run python convert_to_lerobot.py --demos $DART_ROOT/demos/a5 --tag libero_spatial_t5_demo_0 --out ~/hayeon/dart_xarm7/data/lerobot/a5 --pairs "Panda:src_panda,XArm7_retarget_rotq:tgt_xarm7_retarget_rotclip"` | 64s | `data/lerobot/a5/libero_spatial_t5_demo_0_{src_panda (18M), tgt_xarm7_retarget_rotclip (17M)}` |

자동 통과 기준:

| 기준 | 결과 | 노트북 (ORACLE_T45 summary) | 판정 |
|---|---|---|---|
| (a) Panda 재생 성공 | success_at_end=true, 첫 성공 123 | 123 | ✅ |
| (b) xArm7 retarget | xarm_success_at_end=true, 첫 성공 121, ±1 클립 10스텝, method 문자열에 white/panda | 121, 클립 10 | ✅ |
| (c) 데이터셋 | 둘 다 frames=137, episodes=1, fps 10, features 가 `libero_spatial_t0_demo_0_tgt_xarm7_retarget_rotclip` 과 동일 (src 는 robot_type=panda 만 다름 — t0 src_panda 와 같음) | 137 | ✅ |

→ **STOP1 통과.** 다음은 본 단계 1 (FT) — A6000 46GB 에서는 불가, A100 잡에서. 데이터셋은 홈에 있어 그대로 쓰면 됨.
- 휘발성 주의: `$DART_ROOT/demos/a5/*.npz` 는 이 잡 /var/tmp 에만 있음 (FT 에는 불필요, 데이터셋만 필요).
- 홈 사용량 27,878 MiB (데이터셋 35M 추가 후).

## 3. STOP1 산출물 HF 백업 (2026-11-08, 잡 133746) — FT 는 RunPod A100 에서 할 예정

- `upload_folder_hf.py ~/hayeon/dart_xarm7/data/lerobot/a5 a5/data/lerobot` → 10/10 파일, 크기 일치, 0.03 GiB
- `upload_folder_hf.py $DART_ROOT/a5_demos_up a5/demos` (demos/a5/* + spatial_t5_hdf5_demo0-4.npz) → 5/5 파일, 크기 일치, 0.04 GiB
- RunPod 에서 받기 (data/ 는 .gitignore 라 git 에 없음):
  `huggingface_hub.snapshot_download("qoranreh/dart-xarm7-oneshot", allow_patterns=["a5/data/*"], local_dir=<dir>)`
  → `<dir>/a5/data/lerobot/libero_spatial_t5_demo_0_{src_panda,tgt_xarm7_retarget_rotclip}` 를 `~/hayeon/dart_xarm7/data/lerobot/a5/` 로.

## 4. 본 단계 1 — FT 2회 (2026-11-08, 잡 133826, ece-agpu4, A100-SXM4-80GB, 드라이버 535.104.05)

- 계획 변경: RunPod 대신 이 잡에서 FT·합성·평가까지 (사용자 지시 "여기서 바로 FT 진행", 본 단계 1~3 재전송).
- `/var/tmp` 여유 **24G** (94% 사용) → `DART_SHM=/dev/shm/introai34_dart_133826` 로 복원 ($DART_ROOT 가 /dev/shm 링크, /dev/shm 여유 792G·노드 RAM 2TB).
  **잡 끝에 `rm -rf /dev/shm/introai34_dart_133826` 필요** (노드 공용 RAM, 자동으로 안 지워짐).
- 복원 17:00:34 → repo 207s, uv_sync 903s, hf_ckpts 596s, theta0 968s (17:35:12). osmesa 1차 rc=1 (conda-forge HTTP 오류, 일시적) → 백그라운드 재시도 (`logs/restore/osmesa_retry.log`, 평가 전에만 필요).
- **`dart_oneshot_ft.patch` 적용** (17:2x, `git apply`, `git diff` sha256 = 패치 파일 df7b581f…). restore_env.sh 는 패치를 적용하지 않음 — 러너가 `git apply --reverse --check` 로 확인하고 없으면 중단하도록 함.
- 러너 `run_oneshot_ft_a5.sh` = run_oneshot_ft.sh 에서 DATA/OUT/LOGDIR·로그 복사 경로만 변경 (학습 설정 diff 없음). 추가: 패치 확인, HF `a5/demos` 의 STOP1 npz 와 데이터셋 이미지·액션 프레임 단위 대조 + info.json·method 로그, loss 요약, 러닝마다 HF `a5/oneshot_{src,tgt}` 백업.
  - DRY_RUN 17:34 통과: src/tgt 모두 137 프레임 = npz, tgt env_meta `XARM7_COLOR=white, XARM7_DYNAMICS=panda`, method "MimicGen-style retarget … + rotation clipped to norm q01..q99 [white, panda]".
- 실행 17:35:31 `setsid nohup bash run_oneshot_ft_a5.sh` (env.sh: XLA mem 0.9, OPENPI_SAVE_TRAIN_STATE=0, JAX_COMPILATION_CACHE_DIR=$DART_ROOT/jax-cache, WANDB_MODE=offline).
  train.py 인자 확인: batch 64, 1000 step, `--log-interval=10 --overwrite`, checkpoint-base-dir=ckpts/a5/oneshot_src. 속도 7.0 s/step (v0 7.1), GPU 73.5GB·100%.
- 이어서 17:35:29 `run_a5_after_ft.sh` 대기 시작: FT ALL_DONE rc=0 → `run_dart_merge_a5.sh 0.5` (run_dart_merge.sh 에서 CK/OUT/LOG 경로만 변경) → HF a5/dart_a0.5 → 평가 6개 순차 → git commit.
- 예상: src ~19:40, tgt ~21:45, 합성 ~22:10, 평가 끝 ~23:45.
