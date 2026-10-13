"""θ_src / θ_tgt 체크포인트를 HuggingFace 비공개 저장소에 올린다.  (cd $DART_REPO && .venv/bin/python 이파일 <ckpts 폴더>)"""
import pathlib, sys
from huggingface_hub import HfApi
REPO = "qoranreh/dart-xarm7-oneshot"
ck = pathlib.Path(sys.argv[1])
api = HfApi()
api.create_repo(REPO, private=True, exist_ok=True)
readme = """# DART Panda→xArm7 one-shot FT checkpoints (private)
θ0 = openpi pi05_libero 를 pi05_libero_oneshotft (batch 64, 1000 step, A100 1장) 로 full FT. params 는 bf16 (DART·서빙이 bf16 으로 읽음), train_state 없음.
- oneshot_src/: source = Panda, LIBERO libero_spatial task 0 demo_0 (98 frames). loss 0.0041→0.0017
- oneshot_tgt/: target = xArm7, MimicGen 방식 재타깃 + 회전 보정 제한 (98 frames). loss 0.1218→0.0026
사용: create_trained_policy(get_config("pi05_libero_oneshotft"), "<이 폴더>/oneshot_src") — 폴더에 params/, assets/ 가 있음.
학습: 2026-10-10, 잡 129459. 자세한 기록: ~/hayeon/dart_xarm7/ONESHOT_FT.md
"""
api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=REPO)
for name in ("oneshot_src", "oneshot_tgt"):
    step = ck / name / "pi05_libero_oneshotft/original/999"
    print("uploading", step, flush=True)
    api.upload_folder(folder_path=str(step), path_in_repo=name, repo_id=REPO, commit_message=f"add {name} (bf16 params + assets)")
    print("done", name, flush=True)
files = api.list_repo_files(REPO)
print(len(files), "files;", sum(1 for f in files if f.startswith("oneshot_src/")), "src /", sum(1 for f in files if f.startswith("oneshot_tgt/")), "tgt")
