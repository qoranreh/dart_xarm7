#!/usr/bin/env bash
# DART 실험 2 — 조건 여러 개 순차 실행 (VIEW 시점, 기본 medium, 태스크 10개 × 50 에피소드)
# none=θ₀(pi05_libero), all=θ*(DART medium, full δ), vision/vlm/expert = 해당 모듈에만 δ, "vision,vlm" 처럼 조합 가능
# 사용: SUITE=libero_object CONDS="none all vision vlm" bash run_all.sh
cd "$(dirname "$0")"
SUITE=${SUITE:-libero_spatial}
VIEW=${VIEW:-medium}
export SUITE VIEW
for COND in ${CONDS:-none all vision vlm expert}; do
  D="results/${VIEW}_${SUITE}_${COND}"
  if [ -f "$D/summary.txt" ]; then echo "skip $SUITE/$COND (done)"; continue; fi
  rm -f "$D"/client*.jsonl   # 중간에 끊긴 조건은 처음부터
  echo "[$(date '+%F %T')] start $SUITE/$COND"
  COND=$COND bash run_condition.sh
  echo "[$(date '+%F %T')] end $SUITE/$COND (exit=$?)"
done
echo "[$(date '+%F %T')] DONE $VIEW/$SUITE"; cat results/${VIEW}_${SUITE}_*/summary.txt 2>/dev/null | grep -v smoke
