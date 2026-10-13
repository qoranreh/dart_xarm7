# one-shot full FT 2회 결과 (2026-10-08~09, 잡 129138, A100 80GB 1장)

| | θ_src (Panda 데모) | θ_tgt (xArm7 데모, MimicGen+회전 제한) |
|---|---|---|
| 체크포인트 (bf16, 4.9G, 휘발성) | `/var/tmp/pbs.129138.ECE-util1/dart/ckpts/oneshot_src/pi05_libero_oneshotft/original/999` | `/var/tmp/pbs.129138.ECE-util1/dart/ckpts/oneshot_tgt/pi05_libero_oneshotft/original/999` |
| 데이터 | `data/lerobot/libero_spatial_t0_demo_0_src_panda` | `data/lerobot/libero_spatial_t0_demo_0_tgt_xarm7_retarget_rotclip` |
| 설정 | pi05_libero_oneshotft, batch 64, 1000 스텝, mem 0.9, OOM 없음 | 같음 |
| 학습 시간 (1000 스텝) | 2:01:29 (전체 7355s, 로드·컴파일·저장 포함) | 1:58:19 (전체 7164s) |
| loss step 0 → 990 | 0.0041 → 0.0015 | 0.1218 → 0.0026 |
| 마지막 100 스텝 평균 loss | 0.0018 | 0.0023 |
| 처음 50 스텝 평균 loss | 0.0055 | 0.0285 |

- target 은 step 0 loss 가 source 의 약 30배 (θ0 가 xArm7 데이터를 처음 봄) → 50~100 스텝에 source 수준으로 내려옴.
- 체크포인트 폴더: params (bf16, convert_ckpt_bf16.py 로 변환·비트 단위 검증), assets (norm stats), params_dtype.txt. train_state 없음.
- 로그: `logs/oneshot_ft/{src,tgt}_mem0.9_bs64.log`, `status.txt` (loss 는 10 스텝마다 `Step N:` 줄)
- 실행: `run_oneshot_ft.sh` (저장소 수정분 `patches/dart_oneshot_ft.patch`)
- 다음: DART 합성 `exec/run_dart.sh` — BASE_POLICY=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero, POLICY_SRC/TGT = 위 경로 (step 폴더).

## 2026-10-10 재실행 (잡 129459, ece-agpu6) — 이전 잡 체크포인트가 사라져 다시 만듦
/var/tmp 여유 18G 라 저장소·venv 는 /var/tmp, θ0 캐시와 ckpts 는 /dev/shm (`$DART_ROOT/{openpi-cache,ckpts}` 가 링크).

| | θ_src | θ_tgt |
|---|---|---|
| 체크포인트 (bf16 4.9G, 실제 /dev/shm, 휘발성) | `/var/tmp/pbs.129459.ECE-util1/dart/ckpts/oneshot_src/pi05_libero_oneshotft/original/999` | `.../ckpts/oneshot_tgt/pi05_libero_oneshotft/original/999` |
| 전체 시간 | 7690s (2:08) | 7479s (2:05) |
| loss step 0 → 990 | 0.0041 → 0.0017 | 0.1218 → 0.0026 |
| 마지막 100 스텝 평균 | 0.0018 | 0.0022 |
- 10-08 실행과 거의 같음 (src 0.0018/0.0018, tgt 0.0023/0.0022). OOM 없음 (batch 64, mem 0.9). bf16 변환 검증 통과.
- 로그: `logs/oneshot_ft/` (10-08 로그는 `logs/oneshot_ft_20261008_job129138/`)
- **백업 (2026-10-10 22:01):** HuggingFace 비공개 저장소 `qoranreh/dart-xarm7-oneshot` — `oneshot_src/`, `oneshot_tgt/` (각 params bf16 + assets, 로컬과 파일 수·크기 일치 확인).
  복원: `huggingface_hub.snapshot_download("qoranreh/dart-xarm7-oneshot", local_dir=<ckpts>)` (토큰 `~/.cache/huggingface/token`, HF_TOKEN_PATH 로 지정). 업로드 스크립트 `upload_ckpts_hf.py`.
