# DART 실험 3 — 레이어별 CKA (2026-10-06, 잡 128706, ece-agpu15 A100)

데이터: SNUMPR/dart-libero-viewpoints (5 에피소드, 1,085 프레임; 프레임마다 original/small/medium/large 시점 + 손목 이미지).
모델: θ₀(pi05_libero), θ*(dart-pi05-viewpoint-medium/large), 모듈별 δ(vision만 / VLM만; 실험 2와 같은 leaf 교체).
활성값: 레이어 출력을 토큰 평균 — SigLIP 27블록(3인칭 이미지), VLM 입력+18층(3인칭 이미지 / 손목 이미지 / 프롬프트 토큰),
action expert 입력+18층(flow matching 첫 스텝 t=1, 모든 프레임에 같은 노이즈). 수동 forward 는 openpi 모델과 v_t 상대오차 0.4%로 일치.
CKA: linear CKA, 1,085 프레임 (global). 에피소드 평균을 뺀 within 도 계산 — 경향 동일.
코드: extract.py → cka.py → plots.py, robustness.py. 활성값(4.4GB): /var/tmp/pbs.128706.ECE-util1/dart/exp3_acts_v2 (휘발성)

## 질문 A — 시점이 바뀐 입력에서, 각 모델의 표현이 "θ₀ 가 원래 시점을 볼 때"와 얼마나 같은가
회복률 = (CKA_모델 − CKA_θ₀) / (1 − CKA_θ₀). 괄호는 에피소드 1개씩 뺀 5회의 최소~최대.
VLM 은 17층(expert 가 key/value 로 읽는 마지막 prefix 표현; 18층 prefix 출력은 행동에 안 쓰임), expert 는 18층.

| 시점 | 위치 | θ₀ CKA | θ* | vision만 | VLM만 | (참고) 실험 2 성공률 회복 θ* / vision / VLM |
|---|---|---|---|---|---|---|
| medium | vision 26 | 0.789 | +39% | +39% | 0% | 100 / 46 / 51% |
| medium | VLM 이미지 17 | 0.662 | +39% | +33% | +16% | |
| medium | expert 18 | 0.954 | +62% (59~65) | +59% (57~61) | +68% (67~68) | |
| large | vision 26 | 0.775 | +19% | +19% | 0% | 100 / 10 / 106% |
| large | VLM 이미지 17 | 0.426 | +29% | +19% | +21% | |
| large | expert 18 | 0.878 | +84% (77~86) | +70% (50~80) | +89% (83~91) | |

## 질문 B/C — δ 는 어디서 표현을 바꾸나 (같은 입력에서 CKA(θ₀, θ*))
- vision encoder: 17블록까지 거의 그대로(≈1), 마지막 몇 블록만 바뀜 (마지막 0.93 medium / 0.88 large).
- VLM 이미지 토큰·expert: 시점이 바뀐 입력에서 원래 시점 입력보다 훨씬 많이 바뀜 (예: large expert 15층 0.66 vs 0.93)
  → δ 의 효과가 입력에 따라 선택적 (바뀐 시점일 때 주로 작동).
- VLM 프롬프트 토큰 마지막 층: 두 입력 모두 크게 바뀜 (0.22~0.55) → 시점과 무관한 변화 (공유 성분 / 행동에 안 쓰이는 18층 포함).

## 해석
- 행동을 직접 만드는 action expert 표현에서는 VLM δ만으로 θ* 와 같은 수준(89% vs 84%)으로 "원래 시점처럼" 돌아옴 — 실험 2 large 결과(VLM만 106%)와 방향 일치.
  expert 파라미터는 θ₀ 그대로이므로, VLM δ 가 expert 가 읽는 prefix(key/value)를 바꿔서 expert 표현을 되돌린 것.
- 그러나 vision만도 expert 표현을 70% 되돌리는데 성공률은 10%만 회복 → 평균 풀링한 CKA 는 행동 차이를 크게 압축함.
  CKA 만으로는 "왜 vision δ 만으로는 안 되는가"를 설명할 수 없음.
- VLM 이미지 토큰은 시점 변화에 가장 민감(large에서 θ₀ CKA 0.43)한데, 어떤 δ 도 여기를 원래처럼 되돌리지는 못함(≤29%).
  → δ 는 표현을 원래 시점처럼 "복원"하기보다, 바뀐 표현에서 expert 가 쓸 정보를 다시 맞춰 주는 쪽으로 작동하는 것으로 보임 (추정).

## 한계
- (10/06 추가) 데이터(dart-libero-viewpoints, 5 데모 × 4 시점)가 θ* 를 만든 one-shot fine-tune 데이터(viewpoints_5demo)일 가능성이 큼
  (merge_config.json 의 policy_b/c 경로, exec/finetune.sh 예시). → 학습 프레임 위의 CKA 라 회복률이 과대평가됐을 수 있음. 학습에 안 쓰인 장면으로 실험 3-2 진행.
- 평균 풀링 + linear CKA (토큰 위치 정보·비선형 관계는 버림). 에피소드 5개(프레임 1,085)라 장면 다양성이 작음.
- expert 는 flow matching 첫 스텝(t=1)만 봄. 성공률과의 직접 연결은 아직 없음 → 행동 수준 오프라인 지표(데모 action 과의 오차)가 다음 단계 후보.

## 실험 3-2 — 학습에 안 쓰인 장면에서 action 비교 (2026-10-06 18:10~)
장면: θ₀ 가 original 시점으로 libero_spatial 10 태스크 × init state 0~4 를 수행(50/50 성공)하는 동안 10스텝마다
같은 상태를 original/medium/large + 손목 카메라로 저장 → 546장 (collect_states.py, run_collect.sh).
예측: 각 모델·시점의 10스텝 action chunk, 모든 경우 같은 노이즈 (predict_actions.py). 분석: analyze_actions.py.
기준 = θ₀ · original action. 차이 = 처음 5스텝 팔 6차원 L2 (차원별 표준편차로 정규화). 바닥값 = θ₀ · original 을 다른 노이즈로 (0.31).

| 시점 | 모델 | action 차이 (95% CI, 에피소드 부트스트랩) | action 회복률 | 실험 2 성공률 (회복) |
|---|---|---|---|---|
| medium | θ₀ | 0.99 (0.94–1.05) | — | 71.4% |
| medium | θ* | 0.85 (0.79–0.90) | +22% | 85.0% (+100%) |
| medium | vision만 | 0.90 (0.84–0.95) | +15% | 77.6% (+46%) |
| medium | VLM만 | 0.90 (0.84–0.95) | +14% | 78.4% (+51%) |
| large | θ₀ | 1.27 (1.17–1.37) | — | 12.6% |
| large | θ* | 0.96 (0.90–1.01) | +33% | 66.0% (+100%) |
| large | vision만 | 1.20 (1.11–1.30) | +7% | 18.2% (+10%) |
| large | VLM만 | 0.93 (0.87–0.98) | +36% | 69.4% (+106%) |
- 그리퍼 불일치는 모든 조건에서 2~3% (바닥값 2.3%와 같음) → 차이는 팔 움직임에서 남.
- large 태스크×모델 40칸에서 action 차이와 실험 2 성공 수의 Spearman 상관 −0.73.
- 결론: action 수준에서는 실험 2 성공률 패턴이 그대로 재현됨 (large: vision만 ≈ θ₀, VLM만 ≈ θ*; medium: vision만 ≈ VLM만 < θ*).
  실험 3 의 CKA(expert 표현)는 vision만의 효과를 70%로 과대하게 보였는데, 이는 (1) 평균 풀링 CKA 의 압축, (2) 학습 프레임 사용 탓일 수 있음.
- 한계: 장면은 θ₀ 의 성공 궤적 위 상태만 (다른 모델이 실제로 가는 상태는 아님). 노이즈 1개. 회복률 절대값(22~36%)은 바닥값 기준이라 작게 나옴.
