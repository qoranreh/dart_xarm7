#!/usr/bin/env bash
# [13] Baseline v2 (b13): 처음 보는 몸체 xArm7 에서 zero-shot(θ0) / 원샷 FT(θ_tgt) / DART(θ*, α*) 비교 + Panda 양성 대조.
# 순서: 복원 끝 대기 → STOP1 대조군 30회 (합성보다 먼저) → 합성 5개 (α 0.2 0.4 0.6 0.8 1.0) → 비례 확인
#       → 보정 task (t0 t2 t4 t6 × 초기 상태 0–19) 6개 → b13_alpha.py 로 α* 기록 → α* HF 백업 (b13/dart_a<α*>, 평가와 겹침)
#       → 남은 잡 시간 확인 → 보고 task (t1 t3 t5 t7 × 0–49) 4~5개 → b13_summary.py → 영상 HF 백업 (b13/videos/<태그>/)
# 학습 없음. 합성은 공식 CPU 경로 run_dart_merge.sh <α> 만 (ckpts/dart_a<α>), 평가·비교와 겹치지 않게 순차.
# 평가: run_eval_model.sh, exp6_env v0, seed 7 (eval_xarm7.py 기본), 클라이언트 5개, 영상 기본값 (task 마다 ep0), 렌더 캐시 잡 로컬.
# 무효 (errors≠0 또는 행 수≠task×trials) 태그는 원본을 두고 태그 전체를 <태그>_r2 로 1번만 다시 돌린다.
#   setsid nohup bash ~/hayeon/dart_xarm7/run_b13.sh > /dev/null 2>&1 < /dev/null &
# 진행: $DART_ROOT/eval/status_b13.txt. 끝난 합성(END rc=0)·평가(summary.txt 있음)는 건너뜀 → 중간에 죽으면 다시 띄우면 이어서.
# 잡 끝 시각을 알면 $DART_ROOT/eval/b13_job_end.txt 에 'YYYY-MM-DD HH:MM' 한 줄 → 본 평가 전에 남은 시간 확인 (없으면 기본 72h 로 봄).
set -uo pipefail
source ~/hayeon/dart_xarm7/env.sh
source ~/hayeon/dart_xarm7/exp6_cond.sh
exp6_env v0 || exit 2
export MESA_SHADER_CACHE_DIR=$DART_ROOT/mesa-cache; mkdir -p "$MESA_SHADER_CACHE_DIR"
H=~/hayeon/dart_xarm7; CK=$DART_ROOT/ckpts; L=$DART_ROOT/logs
T0=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero
TGT=$CK/oneshot_tgt/pi05_libero_oneshotft/original/999
ALPHAS="0.2 0.4 0.6 0.8 1.0"
CAL="0 2 4 6"; REP="1 3 5 7"
NEED_MIN=225   # 본 평가 3.5h + 마무리 15분
ST=$DART_ROOT/eval/status_b13.txt; RL=$DART_ROOT/eval/run_b13.log; mkdir -p "$(dirname "$ST")" "$L"
exec 9> "$DART_ROOT/eval/run_b13.lock"; flock -n 9 || { echo "run_b13.sh 가 이미 돌고 있음"; exit 1; }
log() { echo "$(date '+%F %T') $*" | tee -a "$ST"; }
hf_env() { HF_TOKEN_PATH=$HOME/.cache/huggingface/token HF_HUB_DISABLE_PROGRESS_BARS=1 "$@"; }
free_g() { df -BG --output=avail "$(readlink -f "$DART_ROOT")" | tail -1 | tr -dc 0-9; }
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader)
UPS=""

up_videos() {  # b13 평가 영상 전부 → HF b13/videos/<태그>/, 파일 수·크기 대조 (사용자 상시 지시)
  local VS=$DART_ROOT/b13_videos n=0 d t
  for d in "$DART_ROOT"/eval/b13_*/videos; do
    ls "$d"/*.mp4 > /dev/null 2>&1 || continue
    t=$(basename "$(dirname "$d")"); mkdir -p "$VS/$t"; ln -f "$d"/*.mp4 "$VS/$t/"; n=$(( n + $(ls "$d"/*.mp4 | wc -l) ))
  done
  [ "$n" -gt 0 ] || { log "영상 없음 — HF 업로드 안 함"; return 0; }
  ( cd "$DART_REPO" && hf_env .venv/bin/python $H/upload_folder_hf.py "$VS" b13/videos "(b13 평가 영상, 기본값 = task 마다 ep0)" ) > $L/b13_hf_videos.log 2>&1
  log "HF b13/videos rc=$? (영상 $n개): $(grep '^b13/videos:' $L/b13_hf_videos.log | tail -1)$(grep -o 'Private storage limit reached' $L/b13_hf_videos.log | head -1)"
}
gitc() {  # gitc <메시지> <경로...> — results 만 (logs/ 는 .gitignore)
  local m=$1; shift
  ( cd ~/hayeon && git add "$@" && git commit -q -m "$m

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "git commit $(git log --oneline -1)" ) || log "git commit 실패/변경 없음"
}
stop() { log "멈춤: $*"; [ -n "$UPS" ] && wait $UPS 2>/dev/null; up_videos; log "=== [b13] STOPPED ==="; exit 0; }

check() {  # check <태그> <기대 행 수> → "OK …" 또는 "BAD …" 한 줄
  /usr/bin/python3 - "$H/results/$1" "$2" <<'PY'
import collections, glob, json, sys
d, n = sys.argv[1], int(sys.argv[2])
r = [json.loads(l) for f in sorted(glob.glob(f"{d}/client*.jsonl")) for l in open(f) if l.strip()]
keys = collections.Counter((x["task_id"], x["episode"]) for x in r)
err = [x["error"] for x in r if x.get("error")]
dup = sum(c - 1 for c in keys.values() if c > 1)
msg = f"rows={len(r)}/{n} errors={len(err)} dup={dup} success={sum(x['success'] for x in r)}"
if err:
    msg += f" first_err={sorted(set(err))[0][:160]!r}"
print(("OK " if len(r) == n and not err and dup == 0 else "BAD ") + msg)
PY
}
pick() { [ -e "$H/results/${1}_r2/summary.txt" ] && echo "${1}_r2" || echo "$1"; }  # 유효한 쪽 (r2 가 있으면 r2)
run1() {  # run1 <태그> <config> <dir> <robot> <tasks> <trials> <메모>
  local tag=$1 cfg=$2 dir=$3 robot=$4 tasks=$5 trials=$6 note=$7 R=$H/results/$1 t0 rc
  [ -e "$R/summary.txt" ] && { log "EVAL $tag 이미 있음 — 건너뜀"; return 0; }
  while ps -eo args | grep -qE "[d]omain_arithmetic|[c]ompare_merge"; do sleep 30; done
  log "EVAL $tag start (config=$cfg robot=$robot tasks=[$tasks] × $trials${note:+ · $note})"
  mkdir -p "$R"; t0=$(date +%s)
  { printf 'tag=%s\ncond=v0 XARM7_COLOR=%s XARM7_DYNAMICS=%s\nserver=serve_ckpt config=%s dir=%s\n' "$tag" "$XARM7_COLOR" "$XARM7_DYNAMICS" "$cfg" "$dir"
    printf 'robot=%s suite=libero_spatial tasks=%s trials=%s nclient=5 seed=7 videos=default(ep0) MESA_SHADER_CACHE_DIR=local\n' "$robot" "$tasks" "$trials"
    [ -n "$note" ] && printf 'note=%s\n' "$note"
    printf 'job=%s host=%s gpu=%s start=%s\n' "$PBS_JOBID" "$(hostname)" "$GPU" "$(date '+%F %T')"; } > "$R/meta.txt"
  EVAL_CKPT_INFO="$cfg:$dir" TAG=$tag CONFIG=$cfg DIR=$dir ROBOT=$robot SUITE=libero_spatial TASKS="$tasks" TRIALS=$trials NCLIENT=5 \
    bash $H/run_eval_model.sh >> "$RL" 2>&1
  rc=$?
  echo "end=$(date '+%F %T') wall=$(( $(date +%s) - t0 ))s rc=$rc" >> "$R/meta.txt"
  log "EVAL $tag rc=$rc wall=$(( $(date +%s) - t0 ))s: $(cat $R/summary.txt 2>/dev/null) | 영상 $(ls $DART_ROOT/eval/$tag/videos/*.mp4 2>/dev/null | wc -l)개"
}
ev() {  # ev <태그> <config> <dir> <robot> <tasks> <trials> [메모] — 무효면 <태그>_r2 로 1번만 (원본 보존, 일부만 다시 돌리지 않음)
  local tag=$1 n=$(( $(wc -w <<< "$5") * $6 )) c
  run1 "$1" "$2" "$3" "$4" "$5" "$6" "${7:-}"
  c=$(check "$tag" $n); [[ $c == OK* ]] && return 0
  log "무효 $tag: $c → 태그 전체를 ${tag}_r2 로 1번만 다시 (원본 보존)"
  run1 "${tag}_r2" "$2" "$3" "$4" "$5" "$6" "${7:+$7 · }r2 (원본 $tag 무효: $c)"
  c=$(check "${tag}_r2" $n); [[ $c == OK* ]] || stop "${tag}_r2 도 무효: $c"
}
merge() {  # merge <α> — 공식 CPU 경로. 평가가 돌고 있으면 기다림 (합성·평가 동시 금지)
  local a=$1 lg=$L/dart_a$1.log
  if grep -q "END rc=0" "$lg" 2>/dev/null && [ -d "$CK/dart_a$a/params" ]; then log "MERGE α=$a 이미 완료 — 건너뜀"; return 0; fi
  while ps -eo args | grep -qE "[e]val_xarm7.py|[s]erve_ckpt.py"; do sleep 30; done
  log "MERGE α=$a start (free $(free_g)G)"
  bash $H/run_dart_merge.sh "$a"
  log "MERGE α=$a $(tail -1 "$lg")"
  grep -q "END rc=0" "$lg" || stop "합성 α=$a 실패 ($lg)"
}
hf_star() {  # α* 체크포인트 → HF b13/dart_a<α*> (기존 dart_a* 폴더를 덮어쓰지 않게 b13/ 아래). 용량 초과로 실패하면 기록만, 아무것도 안 지움
  local a=$1 dst=b13/dart_a$1
  if grep -q "^$dst: .*크기 일치 True" $L/b13_hf_a$a.log 2>/dev/null; then log "HF $dst 이미 올림 — 건너뜀"; return 0; fi
  ( cd "$DART_REPO" && hf_env .venv/bin/python $H/upload_folder_hf.py "$CK/dart_a$a" "$dst" "(b13 θ*_t0 α*=$a)" ) > $L/b13_hf_a$a.log 2>&1
  log "HF $dst rc=$?: $(grep "^$dst:" $L/b13_hf_a$a.log | tail -1)$(grep -o 'Private storage limit reached' $L/b13_hf_a$a.log | head -1)"
}

log "=== [b13] 시작 (job $PBS_JOBID, $(hostname), $GPU, XARM7_COLOR=$XARM7_COLOR XARM7_DYNAMICS=$XARM7_DYNAMICS, DART_ROOT -> $(readlink -f $DART_ROOT), free $(free_g)G) ==="

# ---- 준비 1. 복원 (restore_env.sh) 끝 대기·확인 ----
RS=$L/restore/status.txt
until grep -qE "ALL_DONE|중단" "$RS" 2>/dev/null; do sleep 30; done
for s in repo osmesa uv_sync libero_venv theta0 hf_ckpts; do
  grep -q " $s END rc=0 " "$RS" || stop "복원 단계 $s 실패: $(grep " $s END" "$RS" | tail -1)"
done
log "복원 완료: $(grep ALL_DONE "$RS" | tail -1 | sed 's/^[0-9-]* [0-9:]* //')"
NOK=0
for n in oneshot_src oneshot_tgt dart_a0.5; do
  l=$(grep "^$n → " $L/restore/hf_ckpts.log | tail -1); log "HF $l"; [[ $l == *"크기 일치 True"* ]] && NOK=$((NOK + 1))
done
for d in "$T0" "$CK/oneshot_src/pi05_libero_oneshotft/original/999" "$TGT" "$CK/dart_a0.5"; do
  [ -d "$d/params" ] || stop "체크포인트 없음 $d"; done
log "체크포인트 8개(약 40G)까지 늘어남 — 여유 $(free_g)G (필요 ~25G)"
[ "$(free_g)" -ge 40 ] || stop "여유 공간 부족 ($(free_g)G < 40G)"

# ---- STOP1. 대조군 30회 (합성보다 먼저), t4 × 초기 상태 0–9 ----
ev b13_s1_panda_t4  pi05_libero           "$T0"  Panda 4 10
ev b13_s1_theta0_t4 pi05_libero           "$T0"  XArm7 4 10
ev b13_s1_tgt_t4    pi05_libero_oneshotft "$TGT" XArm7 4 10
S1=$(/usr/bin/python3 - "$H/results" "$(pick b13_s1_panda_t4)" "$(pick b13_s1_theta0_t4)" "$(pick b13_s1_tgt_t4)" <<'PY'
import glob, json, sys
R = sys.argv[1]
def s(tag):
    return sum(json.loads(l)["success"] for f in glob.glob(f"{R}/{tag}/client*.jsonl") for l in open(f) if l.strip())
p, z, t = (s(x) for x in sys.argv[2:5])
print(f"Panda {p}/10 · xArm7 θ0 {z}/10 (Panda−θ0 {p - z}) · 원샷 FT {t}/10 [{' '.join(sys.argv[2:5])}]")
print("C_OK" if p >= 9 and p - z >= 3 and t <= 1 else "C_FAIL")
PY
)
log "STOP1: (a) HF 크기 일치 True $NOK/3 · (b) 세 태그 10행·errors 0 (ev 통과) · (c) $(echo "$S1" | head -1)"
[ "$NOK" = 3 ] || stop "STOP1 (a) 미달"
echo "$S1" | tail -1 | grep -q C_OK || stop "STOP1 (c) 미달"
log "STOP1 자동 통과 → 1단계 합성"
gitc "[13] Baseline v2 (b13): STOP1 대조군 t4 (잡 $PBS_JOBID, $(hostname))" $H/results/b13_s1_*

# ---- 1단계. 합성 5개 (순차) → 비례 확인 (0.4 vs HF 0.5) ----
for a in $ALPHAS; do merge $a; done
CMPF=$L/b13_compare_0.5_vs_0.4.txt
[ -s "$CMPF" ] || ( cd "$DART_REPO" && JAX_PLATFORMS=cpu .venv/bin/python $H/compare_merge.py $CK/dart_a0.5 $CK/dart_a0.4 $T0 ) > "$CMPF" 2>&1
CMP=$(/usr/bin/python3 - "$CMPF" <<'PY'
import math, re, sys
a2 = b2 = ab = 0.0; rows = []
for l in open(sys.argv[1]):
    m = re.match(r"\s+(vision|vlm|expert)\s+([\d.]+) \| ([\d.]+) \| ([-\d.]+) \| ([\d.eE+-]+) \|", l)
    if m:
        name, r, c, na = m.group(1), float(m.group(3)), float(m.group(4)), float(m.group(5))
        nb = r * na; a2 += na * na; b2 += nb * nb; ab += c * na * nb; rows.append(f"{name} 비 {r:.3f} cos {c:.3f}")
if len(rows) != 3:
    print(f"모듈 줄 {len(rows)}개 — 파싱 실패"); print("FAIL"); sys.exit()
ratio, cos = math.sqrt(b2 / a2), ab / math.sqrt(a2 * b2)
print(f"전체 ‖δ_0.4‖/‖δ_0.5‖={ratio:.4f} cos={cos:.4f} | " + " / ".join(rows))
print("OK" if 0.7 <= ratio <= 0.9 and cos >= 0.9 else "FAIL")
PY
)
log "비례 확인 (A=HF dart_a0.5, B=dart_a0.4): $(echo "$CMP" | head -1)"
echo "$CMP" | tail -1 | grep -q OK || stop "비례 확인 미달 (기대 0.7–0.9, cos ≥ 0.9) — 보정 평가 안 함"

# ---- 2단계. 보정 task 6개 → α* ----
ev b13_cal_theta0 pi05_libero "$T0" XArm7 "$CAL" 20 "α=0 기준 행"
for a in $ALPHAS; do ev b13_cal_a$a pi05_libero "$CK/dart_a$a" XArm7 "$CAL" 20; done
/usr/bin/python3 $H/b13_alpha.py > $L/b13_alpha.log 2>&1 || stop "b13_alpha.py 실패 (assert): $(tail -1 $L/b13_alpha.log)"
AS=$(sed -n 's/^alpha_star=//p' $H/results/b13_alpha_choice.txt)
case " $ALPHAS " in *" $AS "*) ;; *) stop "α* 읽기 실패 ('$AS')" ;; esac
[ -d "$CK/dart_a$AS/params" ] || stop "α* 체크포인트 없음 $CK/dart_a$AS"
log "α* = $AS 기록 ($H/results/b13_alpha_choice.txt) | $(sed -n 's/^means=//p' $H/results/b13_alpha_choice.txt) | t0 열만: $(sed -n 's/^t0_only_best=//p' $H/results/b13_alpha_choice.txt)"
gitc "[13] Baseline v2 (b13): 합성 5개·보정 task 6개, α*=$AS 기록 — 보고 task 평가 전 (잡 $PBS_JOBID, $(hostname))" \
  $H/results/b13_cal_* $H/results/b13_table2.md $H/results/b13_alpha_choice.txt
hf_star "$AS" & UPS=$!

# ---- 3단계. 본 평가 (같은 잡·노드에서 이어서) ----
JE=$DART_ROOT/eval/b13_job_end.txt
if [ -s "$JE" ]; then
  rem=$(( ( $(date -d "$(head -1 "$JE")" +%s) - $(date +%s) ) / 60 ))
  [ "$rem" -ge "$NEED_MIN" ] || stop "잡 남은 시간 ${rem}분 < 필요 ${NEED_MIN}분 (잡 끝 $(head -1 "$JE")) — 본 평가 시작 안 함"
  log "잡 남은 시간 ${rem}분 ≥ ${NEED_MIN}분 (잡 끝 $(head -1 "$JE")) → 본 평가"
else
  log "잡 끝 시각 미확인 ($JE 없음) — coss_agpu 기본 72h 로 보고 본 평가 진행"
fi
ev b13_panda     pi05_libero           "$T0"           Panda "$REP" 50
ev b13_theta0    pi05_libero           "$T0"           XArm7 "$REP" 50
ev b13_tgt       pi05_libero_oneshotft "$TGT"          XArm7 "$REP" 50
ev b13_dart_main pi05_libero           "$CK/dart_a$AS" XArm7 "$REP" 50 "alpha_star=$AS (b13_alpha_choice.txt)"
[ "$AS" != 0.4 ] && ev b13_dart_a0.4 pi05_libero "$CK/dart_a0.4" XArm7 "$REP" 50 "참고 행 (논문 α=0.4)"
/usr/bin/python3 $H/b13_summary.py > $L/b13_summary.log 2>&1 || stop "b13_summary.py 실패 (assert): $(tail -1 $L/b13_summary.log)"
log "집계: $(grep -E '^(H1|H2|대조군|\[12\])' $L/b13_summary.log | tr '\n' ' ')"
wait $UPS 2>/dev/null; UPS=""
up_videos
gitc "[13] Baseline v2 (b13): 본 평가 (α*=$AS) 표 1·짝 비교 (잡 $PBS_JOBID, $(hostname))" \
  $H/results/b13_panda* $H/results/b13_theta0* $H/results/b13_tgt* $H/results/b13_dart_* $H/results/b13_table1.md $H/results/b13_summary.json
log "=== [b13] ALL_DONE ==="
