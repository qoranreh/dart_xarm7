#!/usr/bin/env bash
# θ₀ 서버 1개 + 수집 클라이언트 5개 (libero_spatial 10 태스크 × init state 5개, original 시점으로 행동)
set -uo pipefail
W=/var/tmp/pbs.128706.ECE-util1/dart; EXP=$HOME/hayeon/dart_exp3; PORT=9013
OUT=$W/exp3_states; LOG=$W/logs/exp3/collect; mkdir -p $OUT $LOG
( cd $W/dart; export OPENPI_DATA_HOME=$W/openpi_cache XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
  exec $W/venvs/dart-openpi/bin/python $HOME/hayeon/dart_exp2/serve_masked.py \
    --base-dir $W/ckpts/pi05_libero --tuned-dir $W/ckpts/dart-pi05-viewpoint-medium --modules none --port $PORT ) > $LOG/server.log 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null' EXIT
until grep -q "Serving modules" $LOG/server.log; do kill -0 $SERVER || { tail $LOG/server.log; exit 1; }; sleep 5; done
sleep 5; echo "[$(date +%T)] server up"
source $W/client_env.sh; export LP_NUM_THREADS=1 OMP_NUM_THREADS=1
PIDS=()
for c in 0 1 2 3 4; do
  $W/venvs/dart-libero/bin/python $EXP/collect_states.py --args.port $PORT --args.task-ids $c $((c+5)) \
    --args.out $OUT/states_$c.npz > $LOG/client$c.log 2>&1 &
  PIDS+=($!)
done
FAIL=0; for p in "${PIDS[@]}"; do wait $p || FAIL=1; done
echo "[$(date +%T)] done fail=$FAIL"; grep -h "success=" $LOG/client*.log | wc -l
