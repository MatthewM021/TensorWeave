#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m tnlm_v2.benchmark \
  --config configs/reference_cpu.yaml \
  --output results/reference_cpu_reproduction \
  "$@"
python scripts/generate_reference_artifacts.py \
  --results results/reference_cpu_reproduction
