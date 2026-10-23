#!/usr/bin/env bash
# 실험 6 평가 러너 (run_eval_model.sh 복제): 서버 = ~/hayeon/dart_exp2/serve_masked.py (θ = BASE + mask·(TUNED − BASE), 수정 없이 사용)
# 1개 + xArm7 LIBERO 클라이언트 NCLIENT 개 (에피소드 분할), Docker 없이 같은 컨테이너에서.
# 조건(COND)별 XARM7_COLOR / XARM7_DYNAMICS 는 이 스크립트의 매핑으로만 정하고 클라이언트 프로세스에 export 한다
# (eval_xarm7.py 가 libero_xarm7 를 import 할 때 읽음 — 서버와는 무관).
#
# 사용:
#   TAG=v0_mask_vision COND=v0 BASE=<θ0 폴더> TUNED=<θ* 폴더> MODULES=vision bash run_eval_masked.sh
#   MODULES: none (= BASE 그대로) | all (= TUNED 그대로) | vision | vlm | expert | 쉼표 조합 (예: vision,vlm)
#   θ_tgt 단독 평가: BASE=TUNED=<θ_tgt 폴더> MODULES=none CONFIG=pi05_libero_oneshotft
#   선택: CONFIG(pi05_libero) TASKS("0 1 … 9") TRIALS(10) NCLIENT(5) PORT(9100) OVERWRITE(0)
# 결과: ~/hayeon/dart_xarm7/results/$TAG/{client*.jsonl,summary.txt,meta.txt}, 로그·영상: $DART_ROOT/eval/$TAG/
set -uo pipefail
TAG=${TAG:?}; COND=${COND:?}; BASE=${BASE:?}; TUNED=${TUNED:?}; MODULES=${MODULES:?}
CONFIG=${CONFIG:-pi05_libero}
ROBOT=XArm7; SUITE=${SUITE:-libero_spatial}; TASKS=${TASKS:-0 1 2 3 4 5 6 7 8 9}; TRIALS=${TRIALS:-10}
NCLIENT=${NCLIENT:-5}; PORT=${PORT:-9100}

source ~/hayeon/dart_xarm7/exp6_cond.sh   # 실험 6 조건표 (XARM7_* 를 정하는 유일한 곳)
exp6_env "$COND" || exit 2

source ~/hayeon/dart_xarm7/env.sh
EV=$DART_ROOT/eval/$TAG; mkdir -p "$EV/videos"
RES=$HOME/hayeon/dart_xarm7/results/$TAG
if ls "$RES"/client*.jsonl > /dev/null 2>&1 && [ "${OVERWRITE:-0}" != 1 ]; then
  echo "results/$TAG 에 결과가 이미 있음 — 덮어쓰려면 OVERWRITE=1"; exit 2
fi
mkdir -p "$RES"; rm -f "$RES"/client*.jsonl

# 클라이언트 env 가 libero_xarm7 에 실제로 반영되는지 서버를 띄우기 전에 확인 (XARM7_COLOR 미구현이면 white 만 통과)
( source ~/hayeon/dart_xarm7/client_env.sh
  export XARM7_COLOR XARM7_DYNAMICS
  $LIBERO_PY -c "
import os, libero_xarm7 as X
c, d = os.environ['XARM7_COLOR'], os.environ['XARM7_DYNAMICS']
assert X.DYNAMICS == d, f'libero_xarm7.DYNAMICS={X.DYNAMICS} != {d}'
col = getattr(X, 'COLOR', None)
assert col == c or (col is None and c == 'white'), f'XARM7_COLOR={c} 미구현/불일치 (libero_xarm7.COLOR={col})'
print('client env OK: DYNAMICS', X.DYNAMICS, 'COLOR', col or 'white(기본, 분기 없음)')
" ) || { echo "클라이언트 env 확인 실패 — 중단"; exit 2; }

T0=$(date +%s)
cat > "$RES/meta.txt" <<EOF
tag=$TAG
cond=$COND XARM7_COLOR=$XARM7_COLOR XARM7_DYNAMICS=$XARM7_DYNAMICS
base=$BASE
tuned=$TUNED
modules=$MODULES config=$CONFIG
suite=$SUITE tasks=$TASKS trials=$TRIALS nclient=$NCLIENT port=$PORT
job=$PBS_JOBID host=$(hostname) start=$(date '+%F %T')
EOF

( cd "$DART_REPO" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 exec .venv/bin/python ~/hayeon/dart_exp2/serve_masked.py \
    --base-dir "$BASE" --tuned-dir "$TUNED" --modules "$MODULES" --config "$CONFIG" --port "$PORT" ) > "$EV/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for i in $(seq 1 180); do
  grep -q "Serving modules" "$EV/server.log" 2>/dev/null && (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -20 "$EV/server.log"; exit 1; }
  sleep 5
done
grep -q "Serving modules" "$EV/server.log" || { echo "server 준비 시간 초과"; exit 1; }
echo "[$(date +%T)] $TAG server up ($(( $(date +%s) - T0 ))s)"
grep -E "module (vision|vlm|expert)" "$EV/server.log" | sed 's/^.*module/module/' >> "$RES/meta.txt"

source ~/hayeon/dart_xarm7/client_env.sh
export LP_NUM_THREADS=1 OMP_NUM_THREADS=1
export XARM7_COLOR XARM7_DYNAMICS
export EVAL_CKPT_INFO="serve_masked base=$BASE tuned=$TUNED modules=$MODULES config=$CONFIG"   # 클라이언트 로그 첫 줄용
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
ho = [x['success'] for x in r if x['task_id'] != 0]
print(f'$TAG ($COND $MODULES): {s}/{n} = {100*s/max(n,1):.1f}%  held-out {sum(ho)}/{len(ho)} | {line} | errors={e}')
" | tee "$RES/summary.txt"
echo "end=$(date '+%F %T') wall=$(( $(date +%s) - T0 ))s client_fail=$FAIL" >> "$RES/meta.txt"
echo "[$(date +%T)] $TAG done (client fail=$FAIL)"
exit $FAIL
