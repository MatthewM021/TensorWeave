#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m tnlm_v2.benchmark \
  --config configs/smoke.yaml \
  --output results/smoke_reproduction \
  "$@"
