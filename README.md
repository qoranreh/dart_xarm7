# DART 재현·확장 실험

DART (arXiv 2607.00666) 를 openpi pi05_libero 위에서 재현하고, LIBERO 에서 Panda→xArm7 cross-embodiment 로 변형 적용한 실험 코드와 결과 문서.
A100 PBS 클러스터 (NGC PyTorch 23.10 컨테이너) 기준. 체크포인트·데이터셋·영상은 포함하지 않는다.

| 폴더 | 내용 | 결과 문서 |
|---|---|---|
| `dart_exp2/` | 실험 2 — 모듈별 δ (vision / VLM / expert) ablation, LIBERO-view libero_spatial 성공률 | [RESULTS.md](dart_exp2/RESULTS.md) |
| `dart_exp3/` | 실험 3 — 레이어별 CKA, 행동 편차 분석 | [RESULTS.md](dart_exp3/RESULTS.md) |
| `dart_xarm7/` | Panda→xArm7 변형: robosuite xArm7 래퍼 (`src/libero_xarm7`), 데모 쌍 생성, one-shot FT, DART 합성, 평가 | [EVAL_RESULTS.md](dart_xarm7/EVAL_RESULTS.md), [ONESHOT_FT.md](dart_xarm7/ONESHOT_FT.md), [XARM7_NOTES.md](dart_xarm7/XARM7_NOTES.md), [ENV_STATUS.md](dart_xarm7/ENV_STATUS.md) |
| `dart_archive/` | 실험 2·3 클라이언트 환경·체크포인트 받기 스크립트 | — |

- 공식 dart 저장소 커밋 `1b23c4f` 기준. 수정분은 `dart_xarm7/patches/dart_oneshot_ft.patch`.
- xArm7 one-shot FT·DART 체크포인트: HF 비공개 `qoranreh/dart-xarm7-oneshot`.
