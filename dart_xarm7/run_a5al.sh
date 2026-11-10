#!/usr/bin/env bash
# [실험 4-후속5 a5al] α 곡선: θ*_t5 를 α ∈ {0.3, 0.4, 0.5, 0.7, 1.0} (+ 규칙에 따른 가운데 α 최대 2개) 로 합성·평가.
# 학습 없음. 합성은 공식 CPU 경로 run_dart_merge_a5.sh <α> 만, 평가와 겹치지 않게 순차. α=0.5 는 다시 합성하지 않고 HF a5/dart_a0.5 사용.
# 평가: exp6_env v0, seed 7 (eval_xarm7.py 기본), 클라이언트 5개, --save-all-videos (모든 에피소드 영상), 렌더 캐시는 잡 로컬 (a5chk 함정).
#   setsid nohup bash ~/hayeon/dart_xarm7/run_a5al.sh > /dev/null 2>&1 < /dev/null &
# 진행: $DART_ROOT/eval/status_a5al.txt. 이미 끝난 합성(END rc=0)·평가(summary 있음)는 건너뜀 → 중간에 죽으면 다시 띄우면 이어서.
set -uo pipefail
source ~/hayeon/dart_xarm7/env.sh
source ~/hayeon/dart_xarm7/exp6_cond.sh
exp6_env v0 || exit 2
export MESA_SHADER_CACHE_DIR=$DART_ROOT/mesa-cache; mkdir -p "$MESA_SHADER_CACHE_DIR"
H=~/hayeon/dart_xarm7; CK=$DART_ROOT/ckpts; L=$DART_ROOT/logs
T0=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
REST="0 1 2 3 4 6 7 8 9"
ST=$DART_ROOT/eval/status_a5al.txt; RL=$DART_ROOT/eval/run_a5al.log; mkdir -p "$(dirname "$ST")" "$L"
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
hf_env() { HF_TOKEN_PATH=$HOME/.cache/huggingface/token HF_HUB_DISABLE_PROGRESS_BARS=1 "$@"; }
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader)

log "=== [a5al] 시작 (job $PBS_JOBID, $(hostname), $GPU, XARM7_COLOR=$XARM7_COLOR XARM7_DYNAMICS=$XARM7_DYNAMICS, DART_ROOT -> $(readlink -f $DART_ROOT)) ==="

# ---- STOP1-1. HF 체크포인트 받기·대조 (run_a5_after_ft.sh FROM_HF 방식) ----
fetch() {  # fetch <HF 이름> <놓을 경로>
  local n=$1 d=$2
  if [ -d "$d/params" ]; then log "FETCH $n 이미 있음: $d"; return 0; fi
  ( cd "$DART_REPO" && hf_env .venv/bin/python $H/fetch_ckpts_hf.py "$CK/_hf" "$n" ) >> $L/a5al_fetch_hf.log 2>&1 || { log "FETCH $n 실패 — 중단"; exit 1; }
  mkdir -p "$(dirname "$d")" && mv "$CK/_hf/$n" "$d"
  log "FETCH $(grep "^$n " $L/a5al_fetch_hf.log | tail -1)"
}
fetch a5/oneshot_src $CK/a5/oneshot_src/pi05_libero_oneshotft/original/999
fetch a5/oneshot_tgt $CK/a5/oneshot_tgt/pi05_libero_oneshotft/original/999
fetch a5/dart_a0.5   $CK/a5/dart_a0.5
[ "$(grep -c '크기 일치 True' $L/a5al_fetch_hf.log 2>/dev/null)" -ge 3 ] || log "주의: fetch 로그의 '크기 일치 True' 가 3개 미만 (이미 있던 것 포함 여부 확인)"

merge() {  # merge <α> — 공식 CPU 경로. 평가 클라이언트가 돌고 있으면 기다림 (합성·평가 동시 금지)
  local a=$1 lg=$L/dart_a5_a$1.log
  if grep -q "END rc=0" "$lg" 2>/dev/null && [ -d "$CK/a5/dart_a$a/params" ]; then log "MERGE α=$a 이미 완료 — 건너뜀"; return 0; fi
  while ps -eo args | grep -qE "[e]val_xarm7.py|[s]erve_ckpt.py"; do sleep 30; done
  log "MERGE α=$a start (free shm $(df -BG --output=avail $(readlink -f $DART_ROOT) | tail -1 | tr -dc 0-9)G)"
  bash $H/run_dart_merge_a5.sh "$a"
  log "MERGE α=$a $(tail -1 "$lg")"
  grep -q "END rc=0" "$lg" || { log "MERGE α=$a 실패 — 중단"; exit 1; }
}
ev() {  # ev <태그> <dir> <tasks> <trials>
  local tag=$1 dir=$2 tasks=$3 trials=$4 R=$H/results/$1
  [ -e "$R/summary.txt" ] && { log "EVAL $tag 이미 있음 — 건너뜀"; return 0; }
  while ps -eo args | grep -qE "[d]omain_arithmetic"; do sleep 30; done
  log "EVAL $tag start (tasks=[$tasks] × $trials, --save-all-videos)"
  mkdir -p "$R"; local t0=$(date +%s)
  printf 'tag=%s\ncond=v0 XARM7_COLOR=%s XARM7_DYNAMICS=%s\nserver=serve_ckpt config=pi05_libero dir=%s\nsuite=libero_spatial tasks=%s trials=%s nclient=5 seed=7 --save-all-videos MESA_SHADER_CACHE_DIR=local\njob=%s host=%s gpu=%s start=%s\n' \
    "$tag" "$XARM7_COLOR" "$XARM7_DYNAMICS" "$dir" "$tasks" "$trials" "$PBS_JOBID" "$(hostname)" "$GPU" "$(date '+%F %T')" > "$R/meta.txt"
  EVAL_EXTRA_ARGS=--save-all-videos TAG=$tag CONFIG=pi05_libero DIR=$dir ROBOT=XArm7 TASKS="$tasks" TRIALS=$trials NCLIENT=5 bash $H/run_eval_model.sh >> "$RL" 2>&1
  local rc=$?
  echo "end=$(date '+%F %T') wall=$(( $(date +%s) - t0 ))s rc=$rc" >> "$R/meta.txt"
  log "EVAL $tag rc=$rc wall=$(( $(date +%s) - t0 ))s: $(cat $R/summary.txt 2>/dev/null) | 영상 $(ls $DART_ROOT/eval/$tag/videos/*.mp4 2>/dev/null | wc -l)개"
}
hf_up() {  # hf_up <α> — 새 합성 HF 백업 (평가와 겹쳐도 됨, 합성과는 안 겹치게 호출 위치로 보장)
  local a=$1
  ( cd "$DART_REPO" && hf_env .venv/bin/python $H/upload_folder_hf.py "$CK/a5/dart_a$a" "a5/dart_a$a" "(4-후속5 a5al θ*_t5 α=$a)" > $L/a5al_hf_a$a.log 2>&1
    log "HF a5/dart_a$a rc=$?: $(grep "^a5/dart_a$a:" $L/a5al_hf_a$a.log | tail -1)" )
}

# ---- STOP1-2·3·4. α=1.0 합성, 비례 확인, t5 × 20 ----
merge 1.0
if [ ! -s $L/a5al_compare_1.0_vs_0.5.txt ]; then
  ( cd "$DART_REPO" && JAX_PLATFORMS=cpu .venv/bin/python $H/compare_merge.py $CK/a5/dart_a1.0 $CK/a5/dart_a0.5 $T0 ) > $L/a5al_compare_1.0_vs_0.5.txt 2>&1
fi
CMP=$(/usr/bin/python3 - $L/a5al_compare_1.0_vs_0.5.txt <<'PY'
import re, sys, math
a2 = b2 = ab = 0.0; rows = []
for l in open(sys.argv[1]):
    m = re.match(r"\s+(vision|vlm|expert)\s+([\d.]+) \| ([\d.]+) \| ([-\d.]+) \| ([\d.eE+-]+) \|", l)
    if m:
        name, _, r, c, na = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
        nb = r * na; a2 += na * na; b2 += nb * nb; ab += c * na * nb; rows.append(f"{name} 비 {r:.3f} cos {c:.3f}")
ratio, cos = math.sqrt(b2 / a2), ab / math.sqrt(a2 * b2)
print(f"전체 ‖δ_0.5‖/‖δ_1.0‖={ratio:.4f} cos={cos:.4f} | " + " / ".join(rows)); print("OK" if 0.4 <= ratio <= 0.6 and cos >= 0.9 else "FAIL")
PY
)
log "비례 확인: $(echo "$CMP" | head -1)"
ev a5al_a1.0_t5s $CK/a5/dart_a1.0 5 20

S1=$(/usr/bin/python3 - $H <<'PY'
import json, glob, sys
H = sys.argv[1]
r = [json.loads(l) for f in glob.glob(f"{H}/results/a5al_a1.0_t5s/client*.jsonl") for l in open(f)]
o = [json.loads(l) for f in glob.glob(f"{H}/results/a5_dart_t5/client*.jsonl") for l in open(f)]
err = sum(1 for x in r if x.get("error")); s1 = sum(x["success"] for x in r); s05 = sum(x["success"] for x in o if x["episode"] < 20)
print(f"α=1.0 t5 ep0~19 {s1}/{len(r)} errors={err} | α=0.5 (a5_dart_t5) 같은 ep0~19 {s05}/20 | 차이 {s1 - s05:+d}")
print("C_OK" if len(r) == 20 and err == 0 else "C_FAIL"); print("GO" if s1 - s05 >= 4 else "FLAT")
PY
)
log "STOP1: $(echo "$S1" | head -1)"
ok=1
[ "$(grep -c 'END rc=0' $L/dart_a5_a1.0.log)" -ge 1 ] || ok=0
echo "$CMP" | tail -1 | grep -q OK || ok=0
echo "$S1" | grep -q C_OK || ok=0
if [ $ok = 0 ]; then log "STOP1 (a)~(c) 미달 — 멈춤"; exit 0; fi
echo "$S1" | grep -q GO || { log "STOP1: 차이 3 이하 → α 가 t5 를 안 움직임, 본 단계 안 함 — 멈춤"; exit 0; }
log "STOP1 자동 통과 → 본 단계 (α 곡선)"

# ---- 본 단계 5. 합성 0.3 0.4 0.7 (순차) → HF 백업 4개 (평가와 겹치게, 합성과는 안 겹침) ----
for a in 0.3 0.4 0.7; do merge $a; done
for a in 1.0 0.3 0.4 0.7; do hf_up $a & done
UPS=$(jobs -p)

# ---- 6. 평가 10개 순차 ----
for a in 0.3 0.4 0.5 0.7 1.0; do
  ev a5al_a${a}_t5   $CK/a5/dart_a$a 5       50
  ev a5al_a${a}_rest $CK/a5/dart_a$a "$REST" 10
done

# ---- 8. 가운데 α 추가 (미리 정한 규칙, 최대 2개) ----
EXTRA=$(/usr/bin/python3 - $H <<'PY'
import json, glob, sys
H = sys.argv[1]; al = [0.3, 0.4, 0.5, 0.7, 1.0]
def rate(tag):
    r = [json.loads(l) for f in glob.glob(f"{H}/results/{tag}/client*.jsonl") for l in open(f)]
    return 100 * sum(x["success"] for x in r) / max(len(r), 1)
t5 = {a: rate(f"a5al_a{a}_t5") for a in al}; rest = {a: rate(f"a5al_a{a}_rest") for a in al}
cand = []
for a, b in zip(al, al[1:]):
    dr, dt = abs(rest[b] - rest[a]), abs(t5[b] - t5[a]); mid = round((a + b) / 2, 3)
    if (dr >= 15 or dt >= 20) and round((b - a) / 2, 3) >= 0.05:
        cand.append((max(dr / 15, dt / 20), a, b, mid, dr, dt))
cand.sort(reverse=True)
for _, a, b, mid, dr, dt in cand[:2]:
    print(f"{mid:g} {a:g}-{b:g} 나머지 {dr:.1f}%p t5 {dt:.1f}%p")
PY
)
if [ -z "$EXTRA" ]; then log "가운데 α 규칙: 해당 구간 없음 — 추가 안 함"; fi
while read -r mid why; do
  [ -z "$mid" ] && continue
  log "가운데 α=$mid 추가 (구간 $why)"
  wait $UPS 2>/dev/null   # 합성 전에 진행 중인 업로드가 있으면 끝내고 (합성과 업로드는 겹쳐도 되지만 CPU 경쟁 줄이려고)
  merge $mid
  hf_up $mid & UPS="$UPS $!"
  ev a5al_a${mid}_t5   $CK/a5/dart_a$mid 5       50
  ev a5al_a${mid}_rest $CK/a5/dart_a$mid "$REST" 10
done <<< "$EXTRA"

wait $UPS 2>/dev/null
log "=== [a5al] ALL_DONE ==="
