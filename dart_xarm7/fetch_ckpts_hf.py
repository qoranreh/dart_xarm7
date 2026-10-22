"""HF 비공개 저장소(qoranreh/dart-xarm7-oneshot)에서 체크포인트 폴더를 받아 기존 스크립트가 기대하는 경로에 놓고,
파일 수·크기를 HF 메타데이터와 대조한다 (upload_folder_hf.py 의 검증과 같은 방식).

  cd $DART_REPO && HF_TOKEN_PATH=~/.cache/huggingface/token .venv/bin/python ~/hayeon/dart_xarm7/fetch_ckpts_hf.py <ckpts 폴더> <이름...>
  oneshot_*  → <ckpts>/<이름>/pi05_libero_oneshotft/original/999/   (run_oneshot_ft.sh 출력 = run_dart_merge.sh 입력 경로)
  그 밖(dart_a*, <cond>/... ) → <ckpts>/<이름>/
"""
import pathlib
import shutil
import sys

from huggingface_hub import HfApi, snapshot_download

REPO = "qoranreh/dart-xarm7-oneshot"
ck, names = pathlib.Path(sys.argv[1]), sys.argv[2:]
stage = ck / "_hf_stage"
snapshot_download(REPO, allow_patterns=[f"{n}/*" for n in names], local_dir=stage)
info = HfApi().model_info(REPO, files_metadata=True)
ok = True
for n in names:
    dst = ck / n / "pi05_libero_oneshotft/original/999" if n.startswith("oneshot_") else ck / n
    if dst.exists():
        print(f"{n}: {dst} 이미 있음 → 대조만")
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stage / n), str(dst))
    lf = {str(p.relative_to(dst)): p.stat().st_size for p in dst.rglob("*") if p.is_file()}
    rf = {s.rfilename[len(n) + 1:]: s.size for s in info.siblings if s.rfilename.startswith(n + "/")}
    ok &= lf == rf and len(rf) > 0
    print(f"{n} → {dst}: 로컬 {len(lf)} / HF {len(rf)} 파일, 크기 일치 {lf == rf}, {sum(lf.values()) / 2**30:.2f} GiB", flush=True)
shutil.rmtree(stage, ignore_errors=True)
sys.exit(0 if ok else 1)
