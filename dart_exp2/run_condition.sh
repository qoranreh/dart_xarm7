#!/usr/bin/env bash
# DART 실험 2 — 조건 하나 평가: 서버(θ₀ + mask·δ) 1개 + LIBERO-view 클라이언트 N개 (태스크 분할)
# 사용: COND=vision bash run_condition.sh      (COND: none | all | vision | vlm | expert)
set -uo pipefail
COND=${COND:?COND 필요}
VIEW=${VIEW:-medium}
SUITE=${SUITE:-libero_spatial}
TRIALS=${TRIALS:-50}
NCLIENT=${NCLIENT:-5}
PORT=${PORT:-9012}
W=${W:-/var/tmp/pbs.128706.ECE-util1/dart}
EXP=$HOME/hayeon/dart_exp2
NAME=${VIEW}_${SUITE}_${COND}${TAG:-}
OUT=$EXP/results/$NAME
LOG=$W/logs/$NAME
mkdir -p "$OUT" "$LOG" "$W/videos/$NAME"

# --- 서버 (독립 venv, Python 3.11) ---
(
  cd "$W/dart"
  export OPENPI_DATA_HOME=$W/openpi_cache XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
  exec "$W/venvs/dart-openpi/bin/python" "$EXP/serve_masked.py" \
    --base-dir "$W/ckpts/pi05_libero" --tuned-dir "$W/ckpts/dart-pi05-viewpoint-$VIEW" \
    --modules "$COND" --port "$PORT"
) > "$LOG/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for i in $(seq 1 180); do
  grep -q "Serving modules" "$LOG/server.log" 2>/dev/null && (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -20 "$LOG/server.log"; exit 1; }
  sleep 5
done
echo "[$(date +%T)] server up (cond=$COND)"

# --- 클라이언트 (이미지 Python venv + OSMesa), 태스크를 NCLIENT 개로 나눔 ---
source "$W/client_env.sh"
export LP_NUM_THREADS=1 OMP_NUM_THREADS=1
PIDS=()
for c in $(seq 0 $((NCLIENT - 1))); do
  TIDS=$(seq $c $NCLIENT 9 | tr '\n' ' ')
  "$W/venvs/dart-libero/bin/python" "$EXP/run_eval.py" \
    --args.host 127.0.0.1 --args.port "$PORT" --args.task-suite-name "$SUITE" \
    --args.camera-view "$VIEW" --args.num-trials-per-task "$TRIALS" \
    --args.task-ids $TIDS --args.results-path "$OUT/client$c.jsonl" \
    --args.video-out-path "$W/videos/$NAME" \
    > "$LOG/client$c.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for p in "${PIDS[@]}"; do wait $p || FAIL=1; done
echo "[$(date +%T)] clients done (fail=$FAIL)"
cat "$OUT"/client*.jsonl | /usr/bin/python3 -c "
import sys, json
r = [json.loads(l) for l in sys.stdin]
n = len(r); s = sum(x['success'] for x in r)
print(f'$COND: {s}/{n} = {100*s/max(n,1):.1f}%')
" | tee "$OUT/summary.txt"
exit $FAIL
