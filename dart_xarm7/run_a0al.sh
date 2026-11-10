#!/usr/bin/env bash
# [실험 4-후속6 a0al] t0 적응 α 곡선 (4-후속5 run_a5al.sh 와 같은 절차를 v0 = t0 demo_0 체크포인트로).
# 학습 없음. 합성은 공식 CPU 경로 run_dart_merge.sh <α> 만 (ckpts/dart_a<α>), 평가와 겹치지 않게 순차.
# dart_a0.5 (restore_env.sh 가 HF 에서 받음)·dart_a1.0·dart_a0.8 (HF) 은 다시 합성하지 않음.
# 평가: exp6_env v0, seed 7, 클라이언트 5개, --save-all-videos (모든 에피소드 영상), 렌더 캐시 잡 로컬.
#   setsid nohup bash ~/hayeon/dart_xarm7/run_a0al.sh > /dev/null 2>&1 < /dev/null &
# 진행: $DART_ROOT/eval/status_a0al.txt. 끝난 합성·평가는 건너뜀 → 다시 띄우면 이어서.
set -uo pipefail
source ~/hayeon/dart_xarm7/env.sh
source ~/hayeon/dart_xarm7/exp6_cond.sh
exp6_env v0 || exit 2
export MESA_SHADER_CACHE_DIR=$DART_ROOT/mesa-cache; mkdir -p "$MESA_SHADER_CACHE_DIR"
H=~/hayeon/dart_xarm7; CK=$DART_ROOT/ckpts; L=$DART_ROOT/logs
T0=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
TGT=$CK/oneshot_tgt/pi05_libero_oneshotft/original/999
REST="1 2 3 4 5 6 7 8 9"
ST=$DART_ROOT/eval/status_a0al.txt; RL=$DART_ROOT/eval/run_a0al.log; mkdir -p "$(dirname "$ST")" "$L"
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
hf_env() { HF_TOKEN_PATH=$HOME/.cache/huggingface/token HF_HUB_DISABLE_PROGRESS_BARS=1 "$@"; }
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader)

log "=== [a0al] 시작 (job $PBS_JOBID, $(hostname), $GPU, XARM7_COLOR=$XARM7_COLOR XARM7_DYNAMICS=$XARM7_DYNAMICS, DART_ROOT -> $(readlink -f $DART_ROOT)) ==="

# ---- STOP1-1. HF 받기·대조: restore 가 받은 3개 + dart_a1.0 + a5/dart_a0.5 ----
grep -E "oneshot_|dart_a0.5" $L/restore/hf_ckpts.log | sed 's/^/restore hf_ckpts: /' | while read -r l; do log "$l"; done
fetch() {  # fetch <HF 이름> — fetch_ckpts_hf.py 가 이름 그대로 $CK/<이름> 에 놓고 대조 (oneshot_* 아님)
  local n=$1
  if [ -d "$CK/$n/params" ]; then log "FETCH $n 이미 있음"; return 0; fi
  ( cd "$DART_REPO" && hf_env .venv/bin/python $H/fetch_ckpts_hf.py "$CK" "$n" ) >> $L/a0al_fetch_hf.log 2>&1 || { log "FETCH $n 실패 — 중단"; exit 1; }
  log "FETCH $(grep "^$n " $L/a0al_fetch_hf.log | tail -1)"
}
fetch dart_a1.0
fetch a5/dart_a0.5
for d in "$CK/oneshot_src/pi05_libero_oneshotft/original/999" "$TGT" "$CK/dart_a0.5" "$CK/dart_a1.0" "$CK/a5/dart_a0.5"; do
  [ -d "$d/params" ] || { log "체크포인트 없음 $d — 중단"; exit 1; }; done
NOK=$(( $(grep -c '크기 일치 True' $L/restore/hf_ckpts.log) + $(grep -c '크기 일치 True' $L/a0al_fetch_hf.log) ))
log "크기 일치 True 개수: $NOK (기대 5)"

# ---- STOP1-2·3. 비례 (t0 1.0 vs 0.5), 겹침 (t0 0.5 vs t5 0.5) ----
cmp() {  # cmp <A> <B> <출력>
  [ -s "$3" ] || ( cd "$DART_REPO" && JAX_PLATFORMS=cpu .venv/bin/python $H/compare_merge.py "$1" "$2" "$T0" ) > "$3" 2>&1
}
summ() {  # summ <파일> — 모듈별 줄에서 전체 ‖δ_B‖/‖δ_A‖, cos, ‖δ_A−δ_B‖/‖δ_A‖ 를 합산
  /usr/bin/python3 - "$1" <<'PY'
import re, sys, math
a2 = b2 = ab = d2 = 0.0; rows = []
for l in open(sys.argv[1]):
    m = re.match(r"\s+(vision|vlm|expert)\s+([\d.]+) \| ([\d.]+) \| ([-\d.]+) \| ([\d.eE+-]+) \|", l)
    if m:
        name, rd, r, c, na = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
        nb = r * na; a2 += na * na; b2 += nb * nb; ab += c * na * nb; d2 += (rd * na) ** 2
        rows.append(f"{name} 상대차 {rd:.3f} 비 {r:.3f} cos {c:.3f}")
print(f"전체 상대차 {math.sqrt(d2 / a2):.4f} 비(B/A) {math.sqrt(b2 / a2):.4f} cos {ab / math.sqrt(a2 * b2):.4f} | " + " / ".join(rows))
PY
}
cmp $CK/dart_a1.0 $CK/dart_a0.5 $L/a0al_compare_t0_1.0_vs_0.5.txt
P=$(summ $L/a0al_compare_t0_1.0_vs_0.5.txt); log "비례 (A=t0 α1.0, B=t0 α0.5): $P"
cmp $CK/dart_a0.5 $CK/a5/dart_a0.5 $L/a0al_compare_t0_vs_t5_0.5.txt
log "겹침 (A=δ_t0 α0.5, B=δ_t5 α0.5): $(summ $L/a0al_compare_t0_vs_t5_0.5.txt)"
ratio=$(echo "$P" | sed -E 's/.*비\(B\/A\) ([0-9.]+) cos ([0-9.-]+).*/\1 \2/')
ok=$(/usr/bin/python3 -c "r,c=map(float,'$ratio'.split()); print(1 if 0.4<=r<=0.6 and c>=0.9 else 0)")
[ "$NOK" -ge 5 ] || ok=0
[ "$ok" = 1 ] || { log "STOP1 (a)/(b) 미달 — 멈춤"; exit 0; }
log "STOP1 자동 통과 → 본 단계 (t0 α 곡선)"

merge() {  # merge <α>
  local a=$1 lg=$L/dart_a$1.log
  if grep -q "END rc=0" "$lg" 2>/dev/null && [ -d "$CK/dart_a$a/params" ]; then log "MERGE α=$a 이미 완료 — 건너뜀"; return 0; fi
  [ -d "$CK/dart_a$a/params" ] && { log "MERGE α=$a: ckpts/dart_a$a 가 이미 있음 (HF) — 합성 안 함"; return 0; }
  while ps -eo args | grep -qE "[e]val_xarm7.py|[s]erve_ckpt.py"; do sleep 30; done
  log "MERGE α=$a start"
  bash $H/run_dart_merge.sh "$a"
  log "MERGE α=$a $(tail -1 "$lg")"
  grep -q "END rc=0" "$lg" || { log "MERGE α=$a 실패 — 중단"; exit 1; }
}
ev() {  # ev <태그> <config> <dir> <tasks> <trials>
  local tag=$1 cfg=$2 dir=$3 tasks=$4 trials=$5 R=$H/results/$1
  [ -e "$R/summary.txt" ] && { log "EVAL $tag 이미 있음 — 건너뜀"; return 0; }
  while ps -eo args | grep -qE "[d]omain_arithmetic"; do sleep 30; done
  log "EVAL $tag start (config=$cfg tasks=[$tasks] × $trials, --save-all-videos)"
  mkdir -p "$R"; local t0=$(date +%s)
  printf 'tag=%s\ncond=v0 XARM7_COLOR=%s XARM7_DYNAMICS=%s\nserver=serve_ckpt config=%s dir=%s\nsuite=libero_spatial tasks=%s trials=%s nclient=5 seed=7 --save-all-videos MESA_SHADER_CACHE_DIR=local\njob=%s host=%s gpu=%s start=%s\n' \
    "$tag" "$XARM7_COLOR" "$XARM7_DYNAMICS" "$cfg" "$dir" "$tasks" "$trials" "$PBS_JOBID" "$(hostname)" "$GPU" "$(date '+%F %T')" > "$R/meta.txt"
  EVAL_EXTRA_ARGS=--save-all-videos TAG=$tag CONFIG=$cfg DIR=$dir ROBOT=XArm7 TASKS="$tasks" TRIALS=$trials NCLIENT=5 bash $H/run_eval_model.sh >> "$RL" 2>&1
  local rc=$?
  echo "end=$(date '+%F %T') wall=$(( $(date +%s) - t0 ))s rc=$rc" >> "$R/meta.txt"
  log "EVAL $tag rc=$rc wall=$(( $(date +%s) - t0 ))s: $(cat $R/summary.txt 2>/dev/null) | 영상 $(ls $DART_ROOT/eval/$tag/videos/*.mp4 2>/dev/null | wc -l)개"
}
hf_up() {  # hf_up <α> — HF 용량이 모자라면 실패를 기록만 (다시 합성 가능)
  local a=$1
  ( cd "$DART_REPO" && hf_env .venv/bin/python $H/upload_folder_hf.py "$CK/dart_a$a" "dart_a$a" "(4-후속6 a0al θ*_t0 α=$a)" > $L/a0al_hf_a$a.log 2>&1
    log "HF dart_a$a rc=$?: $(grep "^dart_a$a:" $L/a0al_hf_a$a.log | tail -1)$(grep -o 'Private storage limit reached' $L/a0al_hf_a$a.log | head -1)" )
}

# ---- 4. 합성 0.3 0.4 0.7 → HF 백업 (합성 끝난 뒤, 평가와 겹침) ----
for a in 0.3 0.4 0.7; do merge $a; done
for a in 0.3 0.4 0.7; do hf_up $a & done
UPS=$(jobs -p)

# ---- 5. 평가 12개 순차 ----
for a in 0.3 0.4 0.5 0.7 1.0; do
  ev a0al_a${a}_t0   pi05_libero $CK/dart_a$a 0       50
  ev a0al_a${a}_rest pi05_libero $CK/dart_a$a "$REST" 10
done
ev a0al_theta0_t0   pi05_libero "$T0" 0       50
ev a0al_theta0_rest pi05_libero "$T0" "$REST" 10

# ---- 7. 가운데 α (최대 2개; 0.7–1.0 이면 HF dart_a0.8) ----
EXTRA=$(/usr/bin/python3 - $H <<'PY'
import json, glob, sys
H = sys.argv[1]; al = [0.3, 0.4, 0.5, 0.7, 1.0]
def rate(tag):
    r = [json.loads(l) for f in glob.glob(f"{H}/results/{tag}/client*.jsonl") for l in open(f)]
    return 100 * sum(x["success"] for x in r) / max(len(r), 1)
t = {a: rate(f"a0al_a{a}_t0") for a in al}; rest = {a: rate(f"a0al_a{a}_rest") for a in al}
cand = []
for a, b in zip(al, al[1:]):
    dr, dt = abs(rest[b] - rest[a]), abs(t[b] - t[a]); mid = 0.8 if (a, b) == (0.7, 1.0) else round((a + b) / 2, 3)
    if (dr >= 15 or dt >= 20) and round((b - a) / 2, 3) >= 0.05:
        cand.append((max(dr / 15, dt / 20), a, b, mid, dr, dt))
cand.sort(reverse=True)
for _, a, b, mid, dr, dt in cand[:2]:
    print(f"{mid:g} {a:g}-{b:g} 나머지 {dr:.1f}%p t0 {dt:.1f}%p")
PY
)
[ -z "$EXTRA" ] && log "가운데 α 규칙: 해당 구간 없음 — 추가 안 함"
while read -r mid why; do
  [ -z "$mid" ] && continue
  log "가운데 α=$mid 추가 (구간 $why)"
  if [ "$mid" = 0.8 ]; then fetch dart_a0.8; else wait $UPS 2>/dev/null; merge $mid; hf_up $mid & UPS="$UPS $!"; fi
  ev a0al_a${mid}_t0   pi05_libero $CK/dart_a$mid 0       50
  ev a0al_a${mid}_rest pi05_libero $CK/dart_a$mid "$REST" 10
done <<< "$EXTRA"

# ---- 6. (선택) θ_tgt_t0 참고선 ----
ev a0al_theta_tgt_t0 pi05_libero_oneshotft "$TGT" 0 50

wait $UPS 2>/dev/null
log "=== [a0al] ALL_DONE ==="
