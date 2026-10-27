#!/usr/bin/env bash
# 실험 7 파일럿: θ0 서버 1개를 띄워 두고, 섭동 조건마다 eval_perturb.py 클라이언트 NCLIENT 개로 평가한다.
# 기본: 10 태스크 × 초기 상태 20~24 (본 평가용 10~19 는 쓰지 않음) = 조건당 50회. 조건 7개 순차.
# 완료 = 기대 에피소드 수 + errors=0 (아니면 1회 재시도, 실험 6 러너와 같은 기준). 이미 summary 가 있으면 건너뜀 (OVERWRITE=1 로 덮어씀).
# 영상(태스크별 첫 초기 상태)은 조건이 끝날 때마다 홈 결과 폴더로 복사 (노드 디스크는 휘발성).
#   setsid nohup bash ~/hayeon/dart_xarm7/run_exp7_pilot.sh > /dev/null 2>&1 < /dev/null &
#   진행: $DART_ROOT/eval/exp7pilot/status.txt
set -uo pipefail
source ~/hayeon/dart_xarm7/env.sh
CONDS=${CONDS:-"none:0 cam_dist:1.2 cam_dist:1.5 cam_azim:15 cam_azim:30 joint:0.1 joint:0.2"}
TASKS=${TASKS:-"0 1 2 3 4 5 6 7 8 9"}; OFFSET=${OFFSET:-20}; TRIALS=${TRIALS:-5}
NCLIENT=${NCLIENT:-5}; PORT=${PORT:-9100}
CONFIG=pi05_libero; DIR=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
EV=$DART_ROOT/eval/exp7pilot; mkdir -p "$EV"
RROOT=$HOME/hayeon/dart_xarm7/results/exp7pilot; mkdir -p "$RROOT"
ST=$EV/status.txt
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
NT=$(wc -w <<<"$TASKS"); EXPECT=$((NT * TRIALS))

(exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { log "포트 $PORT 에 이미 서버가 있음 — 중단"; exit 1; }
pgrep -f "domain_arithmetic.dart|scripts/train.py" > /dev/null && { log "합성·학습이 돌고 있음 — 중단"; exit 1; }

log "=== 파일럿 시작 (job $PBS_JOBID, $(hostname)) θ0=$DIR 조건: $CONDS / 태스크 $TASKS / 초기 상태 $OFFSET..$((OFFSET + TRIALS - 1)) ==="
( cd "$DART_REPO" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 exec .venv/bin/python ~/hayeon/dart_xarm7/serve_ckpt.py \
    --config "$CONFIG" --dir "$DIR" --port "$PORT" ) > "$EV/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for i in $(seq 1 120); do
  grep -q "Serving config" "$EV/server.log" 2>/dev/null && (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break
  kill -0 $SERVER 2>/dev/null || { log "server died"; tail -20 "$EV/server.log" >> "$ST"; exit 1; }
  sleep 5
done
log "server up"

source ~/hayeon/dart_xarm7/client_env.sh
export LP_NUM_THREADS=1 OMP_NUM_THREADS=1 EVAL_CKPT_INFO="serve_ckpt config=$CONFIG dir=$DIR"
for spec in $CONDS; do
  P=${spec%%:*}; L=${spec#*:}
  TAG=exp7pilot_theta0_${P}_${L}
  RES=$RROOT/$TAG
  if [ -f "$RES/summary.txt" ] && [ -z "${OVERWRITE:-}" ]; then log "$TAG 결과 있음 — 건너뜀"; continue; fi
  for attempt in 1 2; do
    rm -rf "$RES" "$EV/$TAG"; mkdir -p "$RES" "$EV/$TAG/videos"
    T0=$(date +%s); PIDS=()
    for c in $(seq 0 $((NCLIENT - 1))); do
      $LIBERO_PY ~/hayeon/dart_xarm7/eval_perturb.py --port "$PORT" --task-ids $TASKS --trials "$TRIALS" --init-offset "$OFFSET" \
        --perturb "$P" --level "$L" --shard "$c" --nshard "$NCLIENT" --tag "$TAG" \
        --results-path "$RES/client$c.jsonl" --video-out-path "$EV/$TAG/videos" > "$EV/$TAG/client$c.log" 2>&1 &
      PIDS+=($!)
    done
    for p in "${PIDS[@]}"; do wait $p; done
    read -r N E S <<<"$(cat "$RES"/client*.jsonl 2>/dev/null | /usr/bin/python3 -c "
import sys, json
r = [json.loads(l) for l in sys.stdin]
print(len(r), sum(1 for x in r if x.get('error')), sum(x['success'] for x in r))")"
    log "$TAG 시도 $attempt: n=$N/$EXPECT errors=$E 성공=$S ($(( $(date +%s) - T0 ))s)"
    [ "$N" = "$EXPECT" ] && [ "$E" = "0" ] && break
    kill -0 $SERVER 2>/dev/null || { log "server died"; exit 1; }
  done
  { echo "tag=$TAG"; echo "perturb=$P level=$L tasks=$TASKS init=$OFFSET..$((OFFSET + TRIALS - 1)) nclient=$NCLIENT"
    echo "server=serve_ckpt config=$CONFIG dir=$DIR"; echo "job=$PBS_JOBID host=$(hostname) end=$(date '+%F %T') wall=$(( $(date +%s) - T0 ))s attempt=$attempt"
  } > "$RES/meta.txt"
  echo "$TAG: $S/$N = $(( 100 * S / (N > 0 ? N : 1) ))% errors=$E" > "$RES/summary.txt"
  mkdir -p "$RES/videos" && cp "$EV/$TAG"/videos/*.mp4 "$RES/videos/" 2>/dev/null
done
log "=== ALL_DONE ==="
