"""로컬 폴더 하나를 HF 비공개 저장소의 하위 폴더로 올리고 파일 수·크기를 검증한다.
  .venv/bin/python upload_folder_hf.py <로컬 폴더> <저장소 안 경로> [메모]"""
import pathlib, sys
from huggingface_hub import HfApi
REPO = "qoranreh/dart-xarm7-oneshot"
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
api = HfApi()
api.upload_folder(folder_path=str(src), path_in_repo=dst, repo_id=REPO, commit_message=f"add {dst} {sys.argv[3] if len(sys.argv) > 3 else ''}")
info = api.model_info(REPO, files_metadata=True)
lf = {str(p.relative_to(src)): p.stat().st_size for p in src.rglob("*") if p.is_file()}
rf = {s.rfilename[len(dst) + 1:]: s.size for s in info.siblings if s.rfilename.startswith(dst + "/")}
print(f"{dst}: 로컬 {len(lf)} / 원격 {len(rf)} 파일, 크기 일치 {lf == rf}, {sum(rf.values()) / 2**30:.2f} GiB")
