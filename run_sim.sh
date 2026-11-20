#!/usr/bin/env bash
# GELLO -> 시뮬레이터 UR5e (RealTable 로프 장면), 기존 방식(relative) 관절 미러링.
#   bash run_sim.sh                         기본 (로프 물리 원래 해상도)
#   bash run_sim.sh --substep-scale 0.5     더 빠름, 연습용
#   bash run_sim.sh --joint-sign 1,1,-1,1,1,1 --port /dev/ttyUSB1
#   bash run_sim.sh --fake                  GELLO 없이 가짜 입력으로 화면/속도 확인
# 옵션 전체: bash run_sim.sh --help
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$HERE/real2sim-eval/.venv/bin/python}"
# dialout 그룹에 이미 들어가 있으면 다시 로그인하지 않고 바로 적용
if ! id -nG | tr ' ' '\n' | grep -qx dialout && id -nG "$USER" | tr ' ' '\n' | grep -qx dialout; then
  printf -v command '%q ' bash "$HERE/run_sim.sh" "$@"
  exec sg dialout -c "$command"
fi
export R2S_WORKSPACE="$HERE"
unset PYTHONPATH
# 화면이 붙은 GPU에서 띄워야 창이 보인다 (보통 0번)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$PY" -u "$HERE/sysid_routing/code/gello_routing.py" "$@"
