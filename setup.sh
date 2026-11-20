#!/usr/bin/env bash
set -euo pipefail
workspace="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$workspace/real2sim-eval"
unset PYTHONPATH
bootstrap_python="${R2S_PYTHON:-python}"
"$bootstrap_python" -c 'import sys; assert sys.version_info[:2] == (3,11), "Python 3.11 환경을 활성화하세요."'
command -v nvcc >/dev/null || { echo 'CUDA 12.8 Toolkit의 nvcc가 필요합니다. README 설치 항목을 확인하세요.'; exit 1; }
if [[ ! -f /usr/include/glm/glm.hpp ]]; then
  echo '먼저 sudo apt install libglm-dev 를 실행하세요.'; exit 1
fi
if [[ ! -x .venv/bin/python ]]; then "$bootstrap_python" -m venv .venv; fi
.venv/bin/python -m pip install 'pip>=24' 'setuptools<81' wheel ninja
.venv/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-simulator.txt
.venv/bin/python -m pip install --no-build-isolation -e third-party/urdfpy-0.0.22
.venv/bin/python -m pip install --no-build-isolation -e third-party/diff-gaussian-rasterization-w-depth
.venv/bin/python -c 'import torch, warp, diff_gaussian_rasterization; assert torch.cuda.is_available(), "CUDA GPU를 찾을 수 없습니다."'
echo '설치 완료. 저장소 루트에서 bash run_sim.sh 로 실행하세요.'
