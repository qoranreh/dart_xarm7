# 실험 6 조건표 — XARM7_COLOR / XARM7_DYNAMICS 를 정하는 유일한 곳. 데모 생성·평가 러너가 source 해서 쓴다.
#   source ~/hayeon/dart_xarm7/exp6_cond.sh && exp6_env v1   → XARM7_COLOR, XARM7_DYNAMICS export (미정의 조건이면 실패)
exp6_env() {
  case $1 in
    v0) XARM7_COLOR=white XARM7_DYNAMICS=panda ;;
    v1) XARM7_COLOR=dark  XARM7_DYNAMICS=panda ;;
    v2) XARM7_COLOR=white XARM7_DYNAMICS=robosuite ;;
    v3) XARM7_COLOR=dark  XARM7_DYNAMICS=robosuite ;;
    *) echo "COND=$1 미정의 (v0 | v1 | v2 | v3)" >&2; return 2 ;;
  esac
  export XARM7_COLOR XARM7_DYNAMICS
}
