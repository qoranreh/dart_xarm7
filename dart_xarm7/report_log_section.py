"""리포트의 '실험 기록' 섹션 (설정 과정 · 저장 위치 · 공식 코드 대비 변경 내용). build_report.py 가 log_section_html() 을 불러 붙인다.

내용의 원천: ENV_STATUS.md, XARM7_NOTES.md, DEMO_PAIR.md, ONESHOT_FT.md, EVAL_RESULTS.md, patches/dart_oneshot_ft.patch (모두 같은 폴더).
"""
import html
import pathlib

H = pathlib.Path.home() / "hayeon/dart_xarm7"

TOC = """
<nav class="toc" aria-label="목차">
  <a href="#demo">데모 쌍</a><a href="#loss">학습 loss</a><a href="#eval">평가 성공률</a><a href="#videos">대표 영상</a>
  <a href="#log-timeline">타임라인</a><a href="#log-env">환경 설정</a><a href="#log-storage">저장 위치</a><a href="#log-diff">공식 코드 대비 변경</a>
</nav>"""

PERM = '<span class="tag tag-perm">영구</span>'
VOL = '<span class="tag tag-vol">휘발성</span>'


def log_section_html() -> str:
    patch = html.escape((H / "patches/dart_oneshot_ft.patch").read_text())
    return f"""
<section id="log">
  <h2>실험 기록</h2>
  <p>LIBERO 에서 로봇을 Panda → xArm7 로 바꾼 DART cross-embodiment 실험의 전체 과정. 공식 코드는
  <a href="https://github.com/snumprlab/dart">snumprlab/dart</a> @<code>1b23c4f</code> (openpi/JAX), 논문 arXiv 2607.00666 (ECCV 2026).</p>

  <h3 id="log-timeline">타임라인</h3>
  <div class="tbl"><table class="log">
    <thead><tr><th>날짜</th><th>잡 / 노드</th><th>한 일</th><th>결과</th></tr></thead>
    <tbody>
      <tr><th>10-07</th><td>129138 / ece-agpu15</td><td>저장소 클론, uv 학습 venv, θ0(pi05_libero) 다운로드</td><td>임포트·GPU 인식 OK, θ0 11.58 GiB (fp32)</td></tr>
      <tr><th>10-08</th><td>129138</td><td>xArm7 등록 (robosuite v1.5.2 모델 → 1.4.1), 렌더링·액션 공간 확인</td><td>4개 suite 로딩, 액션 7차원 OSC 동일</td></tr>
      <tr><th>10-08</th><td>129138</td><td>데모 쌍: hdf5 demo_0 재생(Panda) + MimicGen 방식 재타깃·회전 제한(xArm7) → LeRobot</td><td>둘 다 태스크 성공, Panda 녹화 = 공식 DART 데이터 재현</td></tr>
      <tr><th>10-08~09</th><td>129138</td><td>one-shot FT 2회 → DART 합성(CPU) → 평가 (θ0 / θ_tgt / θ* α 0.5·0.8·1.0)</td><td>held-out: θ_tgt 0%, θ*0.5 73%, θ0 58%</td></tr>
      <tr><th>10-10</th><td>129442 / ece-agpu5</td><td>새 잡 — /var/tmp 6.6G 라 진행 불가</td><td>노드 이동</td></tr>
      <tr><th>10-10~11</th><td>129459 / ece-agpu6</td><td>환경 복원 (/var/tmp + /dev/shm), FT 2회 재실행, DART 3개, 평가 재실행, HF 백업</td><td>held-out: θ_tgt 0%, θ*0.5 73%, θ0 54% — 같은 결론</td></tr>
    </tbody></table></div>

  <h3 id="log-env">환경 설정 과정</h3>
  <ul class="findings">
    <li><b>클러스터.</b> 로그인 노드(ECE-util)는 코드 편집·git·qstat 만. 설치와 실행은 전부 <code>coss_agpu -g=1 -i 147.46.121.38:5000/pytorch:23.10-py3</code> 로 띄운
      GPU 노드 컨테이너(A100 80GB 1장, CPU 6, 잡 메모리 192GB, Singularity, root 없음) 안에서. 노드는 고를 수 없음 (옵션은 -g, -i 뿐).</li>
    <li><b>저장 공간 3종.</b> 홈(NFS, 모든 노드 공유, 쿼터 ≈28GB 중 27.1GiB 사용 — 70% 가 <code>~/.singularity/cache</code>) /
      <code>/var/tmp</code>(노드마다 별개 디스크 406G, 여유가 노드별로 6.6~24G) / <code>/dev/shm</code>(RAM 디스크, 잡 메모리로 계산될 가능성). 뒤의 둘은 잡이 끝나면 사라짐.</li>
    <li><b>파이썬 환경 2개.</b> 학습·서빙·DART: uv 독립 venv (Python 3.11, JAX 0.5.3, torch 2.7.1 — 이미지 torch 와 분리, 사용자 승인 예외).
      LIBERO 평가: 이미지 Python(<code>/usr/bin/python3</code>) <code>--system-site-packages</code> venv (robosuite 1.4.1, mujoco 3.2.3).
      설치 기록: <code>~/venvs/dart-openpi-requirements.txt</code>, <code>~/venvs/dart-libero-requirements.txt</code>.</li>
    <li><b>렌더링은 OSMesa.</b> 컨테이너에 GPU graphics capability 가 없어(<code>NVIDIA_DRIVER_CAPABILITIES=compute,utility,video</code>) EGL 불가 →
      conda-forge <code>mesalib&lt;25</code>, <code>MUJOCO_GL=osmesa</code>. 공식 평가 이미지와 픽셀이 약간 다름 (평균 2.3/255).</li>
    <li><b>새 잡 복원 순서</b> (약 20분 + 체크포인트는 HF 에서): <code>source env.sh</code> → 클론 + <code>git checkout 1b23c4f</code> → <code>git apply patches/dart_oneshot_ft.patch</code>
      → <code>uv sync --python 3.11</code> → opencv-headless 재설치 → θ0 다운로드 → 평가 venv + OSMesa (<code>client_env.sh</code>) → <code>snapshot_download</code>.</li>
  </ul>
  <details><summary>겪은 문제와 해결 (6건)</summary>
  <div class="tbl"><table class="log"><thead><tr><th>문제</th><th>원인</th><th>해결</th></tr></thead><tbody>
    <tr><th>학습이 첫 배치에서 무한 대기</th><td>uv venv 에 opencv GUI·headless 가 같이 깔려 cv2 가 libX11 요구 → import 실패가 sys.path 를 오염시켜 DataLoader 워커가 죽음</td><td><code>uv pip install --reinstall --no-deps opencv-python-headless==4.11.0.86</code></td></tr>
    <tr><th>홈 쿼터 위험</th><td><code>train.py</code> 가 JAX 컴파일 캐시를 <code>~/.cache/jax</code> 에 씀</td><td>환경 변수로 노드 로컬로 (아래 patch)</td></tr>
    <tr><th>체크포인트 1개 ≈35G</th><td>params + AdamW 상태(fp32) 함께 저장, 디스크 24G</td><td>params 만 저장 + 학습 후 bf16 변환 (DART·서빙이 bf16 으로 읽으므로 결과 비트 단위 동일, θ0 로 검증)</td></tr>
    <tr><th>DART 합성이 수십 배 느려짐</th><td>공식 dart.py 는 CPU 강제, 평가 클라이언트(OSMesa)와 6코어 경쟁</td><td>합성과 평가를 순서대로 (단독 α 당 ~21분, RAM ~45GiB)</td></tr>
    <tr><th>GPU 합성 결과가 다름</th><td>같은 main() 을 GPU 로 돌리면 δ 상대차 17%(TF32)~71%(fp32)</td><td>모든 θ* 는 공식 CPU 경로로</td></tr>
    <tr><th>디스크 부족 / 노드 이동</th><td>/var/tmp 는 노드마다 별개, agpu5 는 6.6G</td><td>agpu6 에서 θ0 캐시·ckpts 를 /dev/shm 에 두고 링크</td></tr>
  </tbody></table></div></details>

  <h3 id="log-storage">저장 위치</h3>
  <div class="tbl"><table class="log"><thead><tr><th>무엇</th><th>위치</th><th></th></tr></thead><tbody>
    <tr><th>θ_src · θ_tgt · θ* (α 0.5 / 0.8 / 1.0)</th><td>HuggingFace 비공개 <code>qoranreh/dart-xarm7-oneshot</code> — <code>oneshot_src/</code>, <code>oneshot_tgt/</code>, <code>dart_a0.5/</code>, <code>dart_a0.8/</code>, <code>dart_a1.0/</code> (각 bf16 4.9G, 파일 수·크기 검증)</td><td>{PERM}</td></tr>
    <tr><th>데모 데이터셋 (LeRobot)</th><td><code>~/hayeon/dart_xarm7/data/lerobot/</code> — <code>libero_spatial_t0_demo_0_src_panda</code>, <code>…_tgt_xarm7_retarget_rotclip</code> (학습에 사용), <code>…_tgt_xarm7_retarget</code>, <code>…_tgt_xarm7_replay</code> (비교용)</td><td>{PERM}</td></tr>
    <tr><th>평가 원자료</th><td><code>~/hayeon/dart_xarm7/results/</code> (10-10), <code>results_20261009_job129138/</code> (10-09) — <code>&lt;모델&gt;/client*.jsonl</code>, <code>summary.json</code></td><td>{PERM}</td></tr>
    <tr><th>평가 영상 50개</th><td><code>~/hayeon/dart_xarm7/results/videos/</code> (태스크별 episode 0)</td><td>{PERM}</td></tr>
    <tr><th>학습 로그</th><td><code>~/hayeon/dart_xarm7/logs/oneshot_ft/</code> (10-10), <code>logs/oneshot_ft_20261008_job129138/</code></td><td>{PERM}</td></tr>
    <tr><th>메모 문서</th><td><code>ENV_STATUS.md</code>, <code>XARM7_NOTES.md</code>, <code>DEMO_PAIR.md</code>, <code>ONESHOT_FT.md</code>, <code>EVAL_RESULTS.md</code> (같은 폴더)</td><td>{PERM}</td></tr>
    <tr><th>저장소 · venv · OSMesa</th><td><code>/var/tmp/pbs.129459.ECE-util1/dart/</code></td><td>{VOL}</td></tr>
    <tr><th>θ0 캐시 · 체크포인트 로컬 사본</th><td><code>/dev/shm/introai34_dart_129459.ECE-util1/</code> (<code>$DART_ROOT/openpi-cache</code>, <code>$DART_ROOT/ckpts</code> 가 링크)</td><td>{VOL}</td></tr>
  </tbody></table></div>
  <p class="note">체크포인트 복원: <code>huggingface_hub.snapshot_download("qoranreh/dart-xarm7-oneshot", local_dir=&lt;ckpts&gt;)</code> (토큰 <code>HF_TOKEN_PATH=~/.cache/huggingface/token</code>).
  받은 폴더는 <code>create_trained_policy</code> · DART 스크립트에 그대로 넣을 수 있다 (params/ + assets/).</p>
  <details><summary>스크립트 목록 (~/hayeon/dart_xarm7/)</summary>
  <div class="tbl"><table class="log"><tbody>
    <tr><th><code>env.sh</code>, <code>client_env.sh</code></th><td>학습/평가 환경 변수 (잡 ID 로 /var/tmp 경로 결정, OSMesa 설정)</td></tr>
    <tr><th><code>src/libero_xarm7/</code></th><td>xArm7 등록, IK, 초기 상태 이식, 그리퍼 state 환산</td></tr>
    <tr><th><code>make_demo_pair.py</code>, <code>make_retarget_demo.py</code></th><td>데모 재생 녹화 / MimicGen 방식 재타깃 (<code>--rot-clip quantile</code>)</td></tr>
    <tr><th><code>convert_to_lerobot.py</code></th><td>녹화 → LeRobot 데이터셋</td></tr>
    <tr><th><code>run_oneshot_ft.sh</code>, <code>convert_ckpt_bf16.py</code></th><td>FT 2회 순차 실행 (OOM 대응 포함) / bf16 변환·검증</td></tr>
    <tr><th><code>run_dart_merge.sh</code></th><td>공식 DART 합성 (CPU) + RAM 기록</td></tr>
    <tr><th><code>serve_ckpt.py</code>, <code>eval_xarm7.py</code>, <code>run_eval_model.sh</code>, <code>run_all_eval_v2.sh</code></th><td>Docker 없는 평가: 정책 서버 + LIBERO 클라이언트 5개</td></tr>
    <tr><th><code>summarize_eval.py</code>, <code>report_assets.py</code>, <code>build_report.py</code></th><td>결과 표·리포트 생성</td></tr>
    <tr><th><code>upload_ckpts_hf.py</code>, <code>upload_folder_hf.py</code></th><td>HF 백업</td></tr>
  </tbody></table></div></details>

  <h3 id="log-diff">공식 코드 대비 변경 내용</h3>
  <div class="tbl"><table class="log"><thead><tr><th>구분</th><th>내용</th></tr></thead><tbody>
    <tr><th>공식 그대로</th><td>DART 합성 <code>uv run -m domain_arithmetic.dart</code> (CPU), 학습 설정 <code>pi05_libero_oneshotft</code> (batch 64, 1000 step, lr 5e-5),
      평가 프로토콜 (seed 7, 대기 10스텝, 5스텝 재계획, max 220, 공식 초기 상태), 데이터 스키마 (<code>examples/libero-view/convert_libero_data_to_lerobot.py</code> 와 동일)</td></tr>
    <tr><th>저장소 수정 3곳<br><span class="muted">(기본 동작 유지)</span></th><td><code>scripts/train.py</code>: 컴파일 캐시 경로를 <code>JAX_COMPILATION_CACHE_DIR</code> 로 바꿀 수 있게<br>
      <code>src/openpi/training/checkpoints.py</code>: <code>OPENPI_SAVE_TRAIN_STATE=0</code> 이면 옵티마이저 상태 저장 생략<br>
      <code>exec/finetune.sh</code>: <code>EXTRA_ARGS</code> 전달 (로그 10스텝마다, <code>--overwrite</code>)</td></tr>
    <tr><th>환경 수정</th><td>opencv-python-headless 4.11.0.86 재설치 (lockfile 버전 그대로)</td></tr>
    <tr><th>저장소 밖 추가<br><span class="muted">(원본에 없음)</span></th><td>xArm7 로봇: robosuite v1.5.2 모델을 1.4.1 API 로 등록, Panda 그리퍼 포즈에 맞춘 IK 초기 자세, 동역학을 Panda 규약으로 (관절 마찰 0, 그리퍼 speed 0.01)<br>
      데모: hdf5 demo_0 을 HTTP 범위 요청으로 읽어 재생, xArm7 은 MimicGen 방식(명령 목표 포즈 추종) + 회전 액션을 학습 통계 q01~q99 로 제한<br>
      학습 후 bf16 변환 (비트 단위 검증), Docker 없는 평가 (서버 JIT 웜업, xArm7 은 Panda 초기 상태에서 물체만 옮기고 팔은 IK), CPU 합성·평가 순차 실행</td></tr>
  </tbody></table></div>
  <details><summary>저장소 patch 전체 (<code>patches/dart_oneshot_ft.patch</code>)</summary><pre class="patch">{patch}</pre></details>

  <h4>논문 설정과의 차이</h4>
  <div class="tbl"><table class="log"><thead><tr><th></th><th>논문 (MimicGen, 부록 C.3)</th><th>이 실험 (LIBERO)</th></tr></thead><tbody>
    <tr><th>로봇</th><td>Panda → UR5e</td><td>Panda → xArm7</td></tr>
    <tr><th>θ0 가 target 로봇을 본 적</th><td>있음 (Square·Threading 을 두 로봇 3,800 데모로 사전학습)</td><td>없음 (pi05_libero 는 Panda 만)</td></tr>
    <tr><th>norm stats</th><td>두 로봇 포함 사전학습 데이터</td><td>Panda LIBERO 통계 (체크포인트 assets 그대로)</td></tr>
    <tr><th>target 데모</th><td>MimicGen 파이프라인 첫 생성 궤적</td><td>MimicGen 방식 재타깃 + 회전 제한, source 와 같은 시작 장면</td></tr>
    <tr><th>α</th><td>0.4 (튜닝용 별도 seed)</td><td>0.5 / 0.8 / 1.0 스윕 (평가 결과로 비교, 0.5 최고)</td></tr>
  </tbody></table></div>
</section>"""
