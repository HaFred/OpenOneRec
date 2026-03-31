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

python -m verl_recsys.smoke.recsys_smoke

echo "[OK] recsys dry-run config + smoke checks resolved successfully."
