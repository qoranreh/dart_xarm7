#!/usr/bin/env bash
# 모델 하나 평가: 정책 서버(serve_ckpt.py) 1개 + LIBERO 클라이언트 NCLIENT 개 (에피소드 분할), Docker 없이 같은 컨테이너에서.
# 사용: TAG=dart_a0.8 CONFIG=pi05_libero DIR=<ckpt> ROBOT=XArm7 TASKS="0 1 2" TRIALS=10 bash run_eval_model.sh
set -uo pipefail
TAG=${TAG:?}; CONFIG=${CONFIG:?}; DIR=${DIR:?}
ROBOT=${ROBOT:-XArm7}; SUITE=${SUITE:-libero_spatial}; TASKS=${TASKS:-0}; TRIALS=${TRIALS:-10}
NCLIENT=${NCLIENT:-5}; PORT=${PORT:-9100}
source ~/hayeon/dart_xarm7/env.sh
EV=$DART_ROOT/eval/$TAG; mkdir -p "$EV/videos"
RES=$HOME/hayeon/dart_xarm7/results/$TAG; mkdir -p "$RES"
rm -f "$RES"/client*.jsonl

( cd "$DART_REPO" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 exec .venv/bin/python ~/hayeon/dart_xarm7/serve_ckpt.py \
    --config "$CONFIG" --dir "$DIR" --port "$PORT" ) > "$EV/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for i in $(seq 1 120); do
  grep -q "Serving config" "$EV/server.log" 2>/dev/null && (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -20 "$EV/server.log"; exit 1; }
  sleep 5
done
echo "[$(date +%T)] $TAG server up"

source ~/hayeon/dart_xarm7/client_env.sh
export LP_NUM_THREADS=1 OMP_NUM_THREADS=1
PIDS=()
for c in $(seq 0 $((NCLIENT - 1))); do
  $LIBERO_PY ~/hayeon/dart_xarm7/eval_xarm7.py --port "$PORT" --robot "$ROBOT" --task-suite-name "$SUITE" \
    --task-ids $TASKS --trials "$TRIALS" --shard "$c" --nshard "$NCLIENT" --tag "$TAG" \
    --results-path "$RES/client$c.jsonl" --video-out-path "$EV/videos" > "$EV/client$c.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for p in "${PIDS[@]}"; do wait $p || FAIL=1; done
cat "$RES"/client*.jsonl | /usr/bin/python3 -c "
import sys, json, collections
r = [json.loads(l) for l in sys.stdin]
by = collections.defaultdict(list)
for x in r: by[x['task_id']].append(x['success'])
line = ' '.join(f't{k}:{sum(v)}/{len(v)}' for k, v in sorted(by.items()))
n = len(r); s = sum(x['success'] for x in r); e = sum(1 for x in r if x.get('error'))
print(f'$TAG ($ROBOT): {s}/{n} = {100*s/max(n,1):.1f}%  | {line} | errors={e}')
" | tee "$RES/summary.txt"
echo "[$(date +%T)] $TAG done (client fail=$FAIL)"
exit $FAIL
