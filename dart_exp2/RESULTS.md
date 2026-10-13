# DART 실험 2 — 모듈별 δ 적용 ablation (2026-10-04~05, 잡 128706, ece-agpu15 A100)

설정: medium 시점, libero_spatial 10 태스크 × 50 에피소드, 조건당 1회 (seed 7). θ₀ = pi05_libero, θ* = SNUMPR/dart-pi05-viewpoint-medium.
모듈 마스크는 leaf 단위 교체 (vision = img/*, VLM = llm/* 중 *_1 아닌 것, expert = llm/*_1 + action/time proj).
평가: 공식 libero-view main.py 수정본 (렌더 카메라 2개, OSMesa, 클라이언트 5개 병렬). 자세한 표: analysis.md

| 조건 | 성공률 | 95% CI | θ₀ 대비 | 격차 회복 | 논문 Fig.13 (4 suite × S/M/L 평균, 판독치) |
|---|---|---|---|---|---|
| θ₀ (none) | 71.4% | 67.3–75.2 | — | 0% | ≈63.8 |
| θ* (all) | 85.0% | 81.6–87.9 | +13.6%p (p<0.001) | 100% | ≈79.6 |
| vision만 | 77.6% | 73.7–81.0 | +6.2%p (p=0.025) | 46% | ≈70.5 (42%) |
| VLM만 | 78.4% | 74.6–81.8 | +7.0%p (p=0.011) | 51% | ≈76.4 (80%) |
| expert만 | 71.8% | 67.7–75.6 | +0.4%p (p=0.89) | 3% | ≈69.5 (36%) |
| vision+VLM (10/05 추가) | 82.2% | 78.6–85.3 | +10.8%p (p<0.001) | 79% | ≈79.1 (97%) |

- vision+VLM 82.2% vs all 85.0% (p=0.23), vs vision p=0.07, vs VLM p=0.13 → 개별 회복분 합(97%)보다 작은 79% 회복. 완전히 더해지지는 않음. 논문은 Vis+LLM ≈ All.
- 논문과 일치: 전체 적용이 최고, expert 기여가 가장 작음, vision+VLM 회복분(31+35=66/68)이 전체에 근접 (논문 Vis+LLM 79.1 ≈ All 79.6).
- 논문과 다름: 논문은 LLM ≫ Vis 인데 여기서는 VLM ≈ vision (p=0.76). expert 단독은 논문 +5.7%p, 여기서는 효과 없음.
  차이 원인 후보(미확인): libero_spatial 1개 suite만, 체크포인트 1개(적응 태스크 조합 1개), 조건당 1회.
- 태스크별로 회복 양상이 다름: T5(ramekin 위 그릇) θ₀ 1 → vision 9 / VLM 20 / expert 5 / all 38.
  T4(서랍 속 그릇)는 θ* 에서 34 → 22 로 하락, VLM만(17)·expert만(18)에서도 하락, vision만(30)은 덜 하락.

## 논문 원문 대조 (arXiv 2607.00666v1 HTML, 2026-10-05 확인)
- Fig.13(층 선택)은 Medium 단독이 아니라 **Small/Medium/Large 평균**. Zero-shot 기준선이 63.8(발표 대본 판독)인지 54.5(Table 1 S/M/L 평균)인지 불확실 → 논문 격차 회복: Vis 42~64%, LLM 80~87%, Action 36~60%, Vis+LLM 97~98%.
- Table 23 suite별 medium (π0.5): Zero-shot / DART = Spatial 65.4 / 87.7, Object 87.8 / 96.3, Goal 69.8 / 76.2, Long 32.6 / 62.9.
  - 재현 spatial: θ₀ 71.4 (CI 67.3–75.2, 논문보다 ~6%p 높음, CI 밖) / θ* 85.0 (CI 81.6–87.9, 일치).
  - large: Spatial 10.4 / 69.5, Object 0.0 / 87.8, Goal 28.0 / 46.5, Long 6.8 / 53.9.

## 추가 실험 — large 시점, spatial (2026-10-05 19:52 → 10-06 03:27)
| 조건 | 성공률 | 95% CI | θ₀ 대비 | θ* 대비 | 격차 회복 |
|---|---|---|---|---|---|
| θ₀ | 12.6% | 10.0–15.8 | — | −53.4%p | 0% |
| θ* (dart-pi05-viewpoint-large) | 66.0% | 61.7–70.0 | +53.4%p | — | 100% |
| vision만 | 18.2% | 15.1–21.8 | +5.6%p (p=0.014) | −47.8%p | 10% |
| VLM만 | 69.4% | 65.2–73.3 | +56.8%p | +3.4%p (p=0.25) | 106% |
- 논문 Table 23 spatial-large (Zero-shot 10.4 / DART 69.5) 와 둘 다 CI 안에서 일치.
- medium 에서는 vision ≈ VLM 이었지만 large 에서는 VLM ≫ vision (p<0.0001). VLM 만으로 θ* 수준.
- 태스크별: VLM만이 θ* 보다 높은 태스크 T4 (42 vs 16), T9 (33 vs 16); θ* 가 높은 태스크 T5 (33 vs 25), T8 (43 vs 29).
- 그림: figs/fig3_recovery_medium_vs_large.png, fig1_success_rate_large.png, fig2_per_task_large.png. 표: results/analysis_large.md
