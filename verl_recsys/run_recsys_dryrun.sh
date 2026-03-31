#!/bin/bash
set -euo pipefail

# Config-only dry-run validation for verl_recsys.
# This does not start Ray workers or training.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m verl_recsys.main_recsys \
  --config-name recsys_dryrun \
  --cfg job \
  --resolve

echo "[OK] recsys dry-run config resolved successfully."
