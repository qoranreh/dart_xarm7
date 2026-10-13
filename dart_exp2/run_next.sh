#!/usr/bin/env bash
# 다음 단계 (2026-10-05): ① spatial-medium Vis+LLM (진행 중) 이 끝나면 → ② spatial-large 4개 조건
cd "$(dirname "$0")"
until [ -f "results/medium_libero_spatial_vision,vlm/summary.txt" ]; do sleep 60; done
VIEW=large SUITE=libero_spatial CONDS="none all vision vlm" bash run_all.sh
echo "[$(date '+%F %T')] NEXT ALL DONE"
