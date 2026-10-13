# DART cross-embodiment 평가 결과 (Panda → xArm7, LIBERO libero_spatial) — 2026-10-09, 잡 129138

## 성공률 (태스크당 10 에피소드, 공식 초기 상태 0~9, t0 = adaptation 태스크, t1~9 = held-out)

| 모델 | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | adapt t0 | held-out t1–9 (95% CI) | 전체 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| θ0 on **Panda** (참고) | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 100% | 100% (96–100) | 100% |
| (a) θ0 | 3 | 7 | 10 | 6 | 6 | 0 | 8 | 6 | 8 | 1 | 30% | 58% (47–67) | 55% |
| (b) θ_tgt one-shot FT | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 60% | **0%** (0–4) | 6% |
| (c) θ* α=0.5 | 9 | 9 | 10 | 10 | 3 | 2 | 10 | 8 | 9 | 5 | **90%** | **73%** (63–81) | **75%** |
| (c) θ* α=0.8 (기본값) | 8 | 9 | 10 | 10 | 2 | 0 | 7 | 8 | 7 | 1 | 80% | 60% (50–70) | 62% |
| (c) θ* α=1.0 | 6 | 5 | 5 | 9 | 0 | 0 | 5 | 3 | 3 | 1 | 60% | 34% (25–45) | 37% |

held-out(90 에피소드) Fisher exact: θ*0.8 vs θ_tgt p=9e-22, θ*0.5 vs θ_tgt p=6e-29, θ*0.5 vs θ0 p=0.041,
θ*0.8 vs θ0 p=0.88, θ*1.0 vs θ0 p=0.003, θ*0.5 vs θ*0.8 p=0.08. adapt t0: θ*0.5 vs θ0 p=0.02.

**DART 핵심 주장 (held-out 에서 θ* > θ_tgt): 세 α 모두 성립.** θ_tgt 는 held-out 전부 실패.

## 체크포인트 (휘발성 /var/tmp)
- θ* α=0.8: `/var/tmp/pbs.129138.ECE-util1/dart/ckpts/dart_a0.8` (공식 CPU 합성, bf16 4.9G, assets 포함)
- α=0.5 / 1.0: 디스크 때문에 평가 후 삭제. `run_dart_merge.sh <α>` 로 다시 만들 수 있음 (CPU 약 20분).
- θ_src / θ_tgt: `.../ckpts/oneshot_{src,tgt}/pi05_libero_oneshotft/original/999`, θ0: `$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero`

## 관찰
1. **θ0 도 xArm7 에서 55%.** 두 로봇이 모두 흰색이고 그리퍼 시작 포즈·손가락 방향·손목 카메라 방향을 Panda 와 맞춰 놔서
   도메인 차이가 생각보다 작다. Panda 에서는 100% 이므로 교체로 인한 하락은 45%p.
2. **θ_tgt 는 데모 궤적을 외움.** held-out 장면에서도 adaptation 데모의 그릇 자리(접시 옆)로 가서 빈 곳을 잡고 시간초과 (영상 t2, t7).
3. **α 가 작을수록 좋음 (0.5 > 0.8 > 1.0).** α=1.0 은 θ0 보다 나쁨. 시험한 값 중 최적은 0.5 이고 그 아래는 안 봤다.
   논문도 MimicGen cross-embodiment 에서는 α=0.4 (LIBERO 시점 변화는 0.8) 를 써서 방향이 같다.
   주의: α 를 평가 결과 자체로 고른 것 (논문은 별도 seed 10 rollout 으로 튜닝).
4. **θ*0.8 은 adapt 태스크만 크게 오르고 (30→80%) held-out 은 θ0 와 같음 (58→60%).** α=0.5 에서는 held-out 도 15%p 오름 (p=0.04).
5. **계속 어려운 태스크: t4 (서랍 속 그릇), t5 (ramekin 위), t9 (캐비닛 위).** 모든 xArm7 모델에서 0~50%.
   t4 는 θ* 가 θ0 보다 나쁨 (6→2~3). 서랍 안에서 팔이 꺾인 자세로 재파지만 반복. t5 는 기운 채 잡혀 접시 가장자리에 떨어뜨림.

## 실패 모드 (영상: `$DART_ROOT/eval/<tag>/videos/<tag>_t<N>_ep0_*.mp4`, 요약 이미지 `results/failure_strips/`)
- 실패는 전부 시간초과 (220 스텝). 프로세스 에러 0건.
- θ0: 파지 실패 → 빈손으로 접시까지 이동 후 맴돎 (t5, t9), 그릇 위에서 재파지 반복 (t0). xArm7 손가락이 더 길고 넓어 Panda 기준 접근이 안 맞는 것으로 보임.
- θ_tgt: 장면 무시, 데모 궤적 재현.
- θ*: t4 서랍 안 재파지 반복, t5 기운 파지 → 떨어뜨림.

## 방법 메모
- DART: 공식 `uv run -m domain_arithmetic.dart` (CPU 강제). α=0.8 57분(평가 클라이언트와 CPU 경쟁 구간 포함), α=0.5/1.0 각 20~22분, RAM 최대 45GiB (잡 한도 192GB).
- **같은 main() 을 GPU 로 돌리면 θ* 가 크게 달라짐**: 기본(TF32) δ 상대차 17%, fp32 highest 71% (Gemma MLP 층은 δ 가 사실상 다름).
  그래서 모든 θ* 는 공식 CPU 경로로 만듦. 비교: `results/compare_cpu_gpu_a0.8.txt`, `results/compare_cpu_gpuhi_a0.8.txt`.
- 평가: Docker 없이 같은 컨테이너에서 `serve_ckpt.py`(create_trained_policy + JIT 웜업) + `eval_xarm7.py` 클라이언트 5개 (OSMesa).
  공식 프로토콜 동일 (seed 7, 대기 10스텝, 5스텝 재계획, max 220). xArm7 은 Panda 초기 상태에서 물체만 옮기고 팔은 IK.
  모델 config: θ0·θ* = pi05_libero, θ_tgt = pi05_libero_oneshotft (같은 모델·변환).
- pi05 행동 샘플링이 확률적이라 같은 에피소드도 실행마다 결과가 다를 수 있음 → 태스크별 10 에피소드는 노이즈가 큼, held-out 합계로 비교.
- 스크립트: `run_dart_merge.sh`, `run_dart_merge_gpu.sh`(비교용), `compare_merge.py`, `serve_ckpt.py`, `eval_xarm7.py`, `run_eval_model.sh`, `run_all_eval.sh`, `summarize_eval.py`
- 원자료: `results/<tag>/client*.jsonl`, `results/summary.json`, 로그 `$DART_ROOT/eval/`, `$DART_ROOT/logs/dart_a*.log`

---
# 재실행 (2026-10-10~11, 잡 129459, ece-agpu6) — θ_src/θ_tgt 를 다시 학습해 같은 절차 반복
결과 원자료 `results/<tag>/` (10-09 결과는 `results_20261009_job129138/` 로 옮김).

| 모델 | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | adapt t0 | held-out t1–9 (95% CI) | 전체 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) θ0 | 7 | 4 | 9 | 5 | 2 | 0 | 10 | 6 | 8 | 5 | 70% | 54% (44–64) | 56% |
| (b) θ_tgt | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 70% | **0%** (0–4) | 7% |
| (c) θ* α=0.5 | 7 | 9 | 10 | 10 | 3 | 2 | 9 | 8 | 10 | 5 | 70% | **73%** (63–81) | **73%** |
| (c) θ* α=0.8 | 6 | 9 | 10 | 10 | 0 | 1 | 6 | 6 | 9 | 4 | 60% | 61% (51–71) | 61% |
| (c) θ* α=1.0 | 6 | 10 | 5 | 10 | 0 | 0 | 6 | 7 | 7 | 5 | 60% | 56% (45–65) | 56% |

held-out Fisher: θ*0.8 vs θ_tgt p=3e-22, θ*0.5 vs θ0 p=0.013, θ*0.8 vs θ0 p=0.45, θ*1.0 vs θ0 p=1.0, θ*0.5 vs θ*0.8 p=0.11.

**두 번 실행의 공통 결론 (held-out 90 에피소드씩):**
- θ* > θ_tgt (0%) 는 모든 α 에서 압도적으로 재현.
- α=0.5 가 가장 좋음 (73% / 73%), θ0 보다 유의하게 높음 (p=0.04 / 0.013). α=0.8 은 θ0 와 비슷 (60 / 61% vs 58 / 54%), α=1.0 은 θ0 이하~비슷 (34 / 56%).
- **adapt 태스크 t0 는 노이즈가 큼**: 같은 θ0 가 3/10 (10-09) vs 7/10 (10-10). 10-09 의 "θ* 가 t0 를 30→80% 로" 는 대부분 우연으로 보임 (합산 θ0 10/20, θ*0.8 14/20).
- t4 (서랍), t5 (ramekin 위) 는 모든 모델에서 0~30%. 실패는 전부 시간초과, 에러 0건.

**체크포인트 (휘발성 /dev/shm + HF 백업):** `/var/tmp/pbs.129459.ECE-util1/dart/ckpts/dart_a{0.5,0.8,1.0}`
HF 비공개 `qoranreh/dart-xarm7-oneshot`: `oneshot_src/`, `oneshot_tgt/`, `dart_a0.5/`, `dart_a0.8/`, `dart_a1.0/` (모두 파일 수·크기 검증).
합성: 공식 CPU, α 당 21~22분, RAM 최대 45GiB. 실행 `run_all_eval_v2.sh`, 업로드 `upload_merges_watch.sh`.
