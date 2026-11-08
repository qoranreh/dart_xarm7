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
- **설치 추가 (사용자 승인 2026-11-08):** libero venv 에 `h5py==3.16.0` (`pip install --no-cache-dir -c ~/venvs/dart-libero-constraints.txt h5py`).
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
