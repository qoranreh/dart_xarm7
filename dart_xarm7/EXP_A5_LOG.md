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
- **18:35 사용자 결정 (가): 합성·평가는 A6000 으로** (CLAUDE.md "A100 은 80GB 필요한 단계만"). 이 잡의 대기 러너 `run_a5_after_ft.sh` 중단 (pgid 1615085 kill, FT 는 계속).
  이 A100 잡은 FT 2회 + HF 백업까지만 하고 exit. 노트북은 안 됨: RAM 30GB < 합성 45GiB, GPU 8GB, EGL 렌더라 v0(OSMesa) 와 픽셀 다름.

## 5. 다음 세션 할 일 — A6000 잡에서 합성·평가 (사용자가 21시쯤 `coss_a6gpu` 잡을 띄움)

FT 잡(133826)은 끝나면 `~/hayeon/dart_xarm7/logs/oneshot_ft_a5/status.txt` (홈, 노드 공용) 에 `src/tgt HF 백업 rc=0: … 크기 일치 True` 를 남긴다 (예상 21:50).
A6000 컨테이너 안에서:

1. 복원: `/var/tmp` 여유를 보고 — ≥ 75G 면 그대로, 모자라면 `DART_SHM=/dev/shm/introai34_dart_<잡번호>` 를 붙여서
   `setsid nohup bash ~/hayeon/dart_xarm7/restore_env.sh > /dev/null 2>&1 < /dev/null &` → `$DART_ROOT/logs/restore/status.txt` 에 ALL_DONE.
   (A6000 잡 메모리 한도 128GB. /dev/shm 사용이 잡 메모리로 잡히는지는 확인 필요 — 합성 RAM 최대 45GiB + /dev/shm ~70G 면 빠듯할 수 있음.)
   osmesa 단계가 conda HTTP 오류로 rc=1 이면 한 번 더 (이 잡에서 1회 실패 → 재시도 성공). 합성·평가에는 dart_oneshot_ft.patch 불필요.
2. `FROM_HF=1 setsid nohup bash ~/hayeon/dart_xarm7/run_a5_after_ft.sh > /dev/null 2>&1 < /dev/null &`
   → 홈 status 에 src·tgt 백업 완료가 찍힐 때까지 대기 → HF `a5/oneshot_{src,tgt}` 를 `ckpts/a5/oneshot_*/pi05_libero_oneshotft/original/999` 로 받기·대조
   → `run_dart_merge_a5.sh 0.5` (CPU) → HF `a5/dart_a0.5` 백업 → 평가 6개 순차 → results/a5_* git commit. 진행 `$DART_ROOT/eval/status_a5.txt`.
3. 끝나면: 프롬프트 7번 (θ*_t5 t5 실패 영상 ≤10개 유형: 빈손으로 닫힘 / 기운 채 잡힘 / 램킨째 들림 / 놓기 3cm 밖 / 기타, ORACLE_T45_LOG.md 11절) → STOP2 3줄 보고
   (t5×50 4개 + Fisher p 2개, 나머지 9 task 90회 θ*_t5 vs θ*_t0(기존 results/v0_dart_a0.5, t5 제외 70/90) vs θ_tgt_t5 + 95% CI, H_fix/H_nofix/trade-off 판정).
   평가 GPU 가 v0(A100) 와 다름 → 표에 명시. /dev/shm 을 썼으면 잡 끝에 rm -rf.

### 4-1. FT src 결과 (잡 133826)
| 항목 | a5 θ_src (t5 Panda) | v0 θ_src (t0, ONESHOT_FT.md) |
|---|---|---|
| 학습 시간 | 17:35:31 → 19:38:40, train_wall 7389s (2h03m), bf16 변환 47s | ≈2h |
| loss step 0 → 990 | 0.0071 → 0.0025 | 0.0041 → 0.0015 |
| 마지막 100스텝 평균 (900~990, 10개) | 0.0026 | 0.0018 |
| 체크포인트 | `ckpts/a5/oneshot_src/pi05_libero_oneshotft/original/999` bf16 4.9G | |
| HF 백업 | `a5/oneshot_src`: 64/64 파일, 크기 일치 True, 4.90 GiB (19:44:38) | |
- OOM 없음 (mem 0.9, batch 64 첫 시도). tgt 19:39:27 시작.

### 4-2. FT tgt 결과 + 마무리 (잡 133826)
| 항목 | a5 θ_tgt (t5 xArm7 retarget rotclip) | v0 θ_tgt (t0) |
|---|---|---|
| 학습 시간 | 19:39:27 → 21:38:51, train_wall 7164s (1h59m), bf16 변환 61s | ≈2h |
| loss step 0 → 990 | 0.0875 → 0.0023 | 0.1218 → 0.0026 |
| 마지막 100스텝 평균 (900~990, 10개) | 0.0028 | 0.0022 |
| 체크포인트 | `ckpts/a5/oneshot_tgt/pi05_libero_oneshotft/original/999` bf16 4.9G | |
| HF 백업 | `a5/oneshot_tgt`: 71/71 파일, 크기 일치 True, 4.90 GiB (21:45:45) | |
- `=== ALL_DONE rc=0 ===` 21:45:45. OOM 없음. 이상 징후: 없음 (src 64 파일 / tgt 71 파일로 파일 수가 다르지만 둘 다 HF 와 일치, 4.90 GiB — orbax 저장 조각 수 차이로 보임, 확인 안 함).
- step 0 loss: tgt 0.0875 ≫ src 0.0071 — v0 와 같은 패턴 (θ0 가 Panda 는 이미 잘하고 xArm7 retarget 액션은 낯섦).
- 로그 홈 복사: `logs/oneshot_ft_a5/` (status·data_check·src/tgt 학습 로그·HF 업로드 로그 — `logs/` 는 .gitignore 라 git 에는 없음, 홈에만). 홈 status 에 A6000 러너 신호 2줄 확인.
- run_a5_after_ft.sh 마지막 git add 에서 logs/ 를 빼고 results/a5_* + EXP_A5_LOG.md 만 (ignore 경로를 넣으면 add 전체가 실패해 커밋이 안 됨).
- 정리: `/dev/shm/introai34_dart_133826` 삭제 (체크포인트는 HF 에 대조 완료). 이 잡은 exit.

## 6. 본 단계 2·3 — 합성·평가 (2026-11-08 21:53 → 11-09 00:02, 잡 133903, ece-a6gpu7, RTX A6000)

- 복원 21:53:12 → 22:12:47 ALL_DONE (/var/tmp 직접, 여유 97G → 50G). osmesa 이번엔 1회에 성공 (785s).
- `FROM_HF=1 run_a5_after_ft.sh` (복원 ALL_DONE 후 자동 시작 22:13:16, exp6_env v0 → client env OK white panda).
  - HF a5/oneshot_src 64/64, a5/oneshot_tgt 71/71 받기·대조 OK (22:16, 22:18).
  - 합성 `run_dart_merge_a5.sh 0.5`: END rc=0, wall 1033s, peak RSS 45.1GiB, 4.9G → HF `a5/dart_a0.5` 22/22 파일 크기 일치 (22:41).
  - 평가 (run_eval_model.sh, serve_ckpt, 클라이언트 5개, seed 7, A6000 — v0 결과는 A100 잡이었음):

| 태그 | 모델 | task / 회 | 결과 | 시간 |
|---|---|---|---|---|
| a5_theta0_t5 | θ0 | t5 × 50 | 7/50 = 14.0% | 796s |
| a5_dartT0_t5 | θ*_t0 (v0 dart_a0.5) | t5 × 50 | 6/50 = 12.0% | 780s |
| a5_theta_tgt_t5 | θ_tgt_t5 | t5 × 50 | 42/50 = 84.0% | 574s |
| a5_dart_t5 | θ*_t5 | t5 × 50 | 22/50 = 44.0% | 725s |
| a5_theta_tgt_rest | θ_tgt_t5 | 나머지 9 × 10 | 7/90 = 7.8% (t0 6/10, t8 1/10, 나머지 0) | 1336s |
| a5_dart_rest | θ*_t5 | 나머지 9 × 10 | 77/90 = 85.6% (t0 10 t1 10 t2 9 t3 10 t4 7 t6 10 t7 8 t8 10 t9 3) | 961s |

  - 기존 재사용 (잡 다름, A100): results/v0_theta0 t5 제외 56/90 = 62.2%, results/v0_dart_a0.5 t5 제외 70/90 = 77.8% (t0 8 t1 10 t2 10 t3 10 t4 4 t6 10 t7 5 t8 10 t9 3).
  - 통계 (Fisher 양측, CI = Clopper-Pearson 95%): θ*_t5 vs θ0 (t5) p=0.0018, θ*_t5 vs θ*_t0 (t5) p=0.00067, θ_tgt_t5 vs θ*_t5 (t5) p=5.6e-5.
    나머지 90회: θ*_t5 85.6% [76.6, 92.1] / θ*_t0 77.8% [67.8, 85.9] (p=0.25) / θ_tgt_t5 7.8% [3.2, 15.4] / θ0 62.2% [51.4, 72.2].
  - git commit 4db93c2 (러너 자동). 이상 징후: 없음 (errors=0 전부).
- 7번 실패 영상: eval_xarm7.py 가 ep0 영상만 저장 → θ*_t5 t5 실패 28개 중 영상 1개. `--save-all-videos` 플래그 추가 (기본 off, 결과 무관) + run_eval_model.sh `EVAL_EXTRA_ARGS` 통과,
  θ*_t5 t5 × 50 을 새 태그 `a5_dart_t5_vid` 로 재실행해 실패 영상 확보 (재현성 확인 겸).

## 7. θ*_t5 t5 실패 유형 (프롬프트 7번, 2026-11-09 00:04 → 00:40, 잡 133903)

- `eval_xarm7.py` 에 `--save-all-videos` (기본 off) 추가 — 프레임 기록(109줄)과 저장(133줄) 둘 다 ep0 한정이었음. 1회차 `a5_dart_t5_vid` 는 저장만 고쳐서 영상이 안 남음 (결과 26/50 = 52%).
  2회차 `a5_dart_t5_vid2`: 20/50 = 40%, 실패 영상 30개. → θ*_t5 t5 세 번 22 / 26 / 20 (평균 45%) — 반복해도 같은 수준.
- vid2 실패 30개 중 10개 (ep 1·5·10·16·19·24·29·34·40·46, 고르게) 를 `video_frames.py --n 8` 로 → `renders/a5_fail/ep*.png` 보고 판정 (224px 8프레임 육안 — 확신 낮은 것 표시).

| 유형 (ORACLE_T45_LOG 11절 이름) | 개수 | 에피소드 |
|---|---|---|
| 램킨째 들림 (램킨이 같이 접시로 옮겨지고 그릇이 그 위나 옆에 얹힘) | 7 | 1, 16, 24, 29, 34, 40(확신 낮음), 46 — 34·46 은 그릇이 램킨 위에 얹힌 채 접시 위에 놓임 |
| 빈손으로 닫힘 (그릇 위에서 잡기 반복, 시간 초과) | 1 | 5 |
| 기운 채 잡힘 | 0 | |
| 놓기 3cm 밖 | 0 | |
| 기타: 잡으러 내려가다 그릇을 램킨에서 떨어뜨려 뒤집음 → 이후 램킨을 집어 옮김 | 2 | 10, 19 |

- 주 실패 = **램킨째 들림 (7/10)**. ORACLE_T45 에서 t5 retarget 정답 궤적도 실패 6개가 전부 같은 유형이었음 (demo 4·8·9) → xArm7 그리퍼가 그릇을 잡을 때 램킨까지 같이 끼우는 몸체 차이 문제로 보임 (추정).

## 8. STOP2 (2026-11-09 00:40)

1. t5 × 50: θ0 14% / θ*_t0 12% / θ_tgt_t5 84% / θ*_t5 44% (재실행 52%, 40%). θ*_t5 vs θ0 Fisher p=0.0018, θ*_t5 vs θ*_t0 p=0.00067.
2. 나머지 9 task 90회: θ*_t5 85.6% [76.6, 92.1] vs θ*_t0 77.8% [67.8, 85.9] (기존, A100 잡) vs θ_tgt_t5 7.8% [3.2, 15.4]. (θ0 62.2%, 기존)
3. 판정: **H_fix 맞음** — t5 +30%p (≥ +20%p, p<0.05), 나머지는 θ*_t0 대비 +7.8%p (±15%p 안, p=0.25). trade-off 아님.
   단 θ_tgt_t5 (84%) 의 t5 이득 중 약 절반만 θ* 로 옴. θ*_t5 실패 10개 중 7개 = 램킨째 들림 (정답 궤적 retarget 실패와 같은 유형).
- 주의: 이번 평가는 A6000 (v0 기존 결과는 A100) — 나머지 9 task 비교는 GPU·잡이 다름. t5 4개는 모두 같은 잡·같은 노드.
- 잡 133903 의 $DART_ROOT 는 /var/tmp (shm 아님) → 별도 정리 불필요. 체크포인트는 전부 HF a5/ 에 대조 완료.

## 9. 영상 백업 (2026-11-09, 잡 133903)
- 평가 영상 전체 (a5_* 8개 태그, 73개 mp4) + 실패 판정 프레임 10장 → HF `a5/videos/<태그>/`, `a5/videos/a5_fail_frames/`: 83/83 파일 크기 일치, 0.02 GiB.
- 이후 실험부터 평가 영상은 체크포인트처럼 잡 끝나기 전에 HF 로 올림 (사용자 지적).
