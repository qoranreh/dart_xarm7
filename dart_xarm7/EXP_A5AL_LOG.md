# 실험 4-후속5 (a5al) — α 곡선: α 만 바꿔 θ*_t5 가 t5 에서 θ_tgt_t5 쪽으로 가면서 나머지 9 task 를 지키나

- 실험 문장 (사용자, 2026-11-10): "이건 알파값을 바꿔서 나머지 task 성공률이 오르는지 내리는지 보고~" (H1/H2 칸은 비어 있음 — 사용자가 "바로 시작" 지시).
- α: 0.3 / 0.4 / 0.5 (HF a5/dart_a0.5, 다시 합성 안 함) / 0.7 / 1.0 (+ 규칙에 따른 가운데 α 최대 2개). src·tgt = HF a5/oneshot_src, a5/oneshot_tgt.
- 조건: exp6_env v0 (white, panda), seed 7, NCLIENT 5, 모든 평가 `--save-all-videos` (사용자 지시: 에피소드 영상 전부 저장. 프롬프트의 EVAL_VIDEO_EPISODES 는 코드에 없음 — 4-후속4 의 (나) 선택과 같음).
- 러너 `run_a5al.sh` (신규): HF 받기·대조 → α=1.0 합성 → compare_merge 비례 확인 → t5×20 → STOP1 자동 판정 → 합성 0.3·0.4·0.7 → HF 백업 → 평가 10개 → 가운데 α 규칙. 렌더 캐시는 잡 로컬 (`MESA_SHADER_CACHE_DIR`, 4-후속4 함정).

## STOP1-0. git (2026-11-10 11:0x, 잡 134433 컨테이너)
- HTTPS fetch: origin/main 에 4-후속3·4 커밋 4개 없음 (4db93c2, 698f626, e49857e, 8dc7adf), fast-forward 가능, 충돌 없음.
  컨테이너는 SSH push 불가 → **로그인 노드에서 `cd ~/hayeon && git push origin main` 필요**.
- 추적 안 된 `results/a5_theta_tgt_t5_vid/` 있음 (4-후속3 끝에 사용자가 중단시킨 θ_tgt 성공 영상 재실행의 흔적으로 보임, 손대지 않음).

## STOP1-1. 복원 (잡 134433, ece-a6gpu8, RTX A6000)
- `/var/tmp` 여유 4G (100%) → `DART_SHM=/dev/shm/introai34_dart_134433` (/dev/shm 여유 225G, 노드 공용 RAM → **잡 끝에 rm -rf**).
- 사용자 결정: v0 체크포인트 3개 (15G) 는 받지 않음. 레포의 restore_env.sh 는 고치지 않고, `step hf_ckpts` 한 줄만 뺀 사본 (scratchpad `restore_env_a5al.sh`) 으로 실행. 11:00:04 시작.
- 복원 11:00:04 → 11:36:40 ALL_DONE: repo·osmesa 848s·uv_sync 1350s·libero_venv 489s·theta0 514s, hf_ckpts SKIP. /dev/shm 여유 203G, 사용 23G.

## STOP1-1~4 결과 (러너 `run_a5al.sh` 11:37 자동 시작)
| 항목 | 결과 |
|---|---|
| HF a5/oneshot_src | 64/64 파일, 크기 일치 True, 4.90 GiB (11:40) |
| HF a5/oneshot_tgt | 71/71, 크기 일치 True, 4.90 GiB (11:42) |
| HF a5/dart_a0.5 | 22/22, 크기 일치 True, 4.89 GiB (11:45) |
| 합성 α=1.0 `run_dart_merge_a5.sh 1.0` | END rc=0, wall 888s, peak RSS 45.4GiB, 4.9G → ckpts/a5/dart_a1.0 (로그의 free=4G 는 /var/tmp 값, 실제 위치는 /dev/shm) |
| 비례 `compare_merge.py dart_a1.0 dart_a0.5 θ0` | 전체 ‖δ_0.5‖/‖δ_1.0‖ = 0.5006, cos = 0.9851 · vision 0.501 / 0.990 · vlm 0.501 / 0.982 · expert 0.499 / 0.989 (전체 값은 모듈별 ‖δ‖·비·cos 로 합산) |
| 평가 a5al_a1.0_t5s (t5 × 20) | 18/20, errors=0, 영상 20, 230s |
| α=0.5 같은 초기상태 0~19 (results/a5_dart_t5) | 8/20 → 차이 **+10** |

→ (a) (b) (c) 만족, 차이 ≥ 4 → **STOP1 자동 통과** (12:05), 본 단계 시작 (합성 α=0.3 부터).
- cos 0.985 (1 이 아님): θ* 를 bf16 으로 저장해 δ 가 반올림된 영향으로 보임 (추정).
