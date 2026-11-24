#!/usr/bin/env bash
# Source-side check: does training on MELD alone work, with and without
# class-weighted CE? Uses only the EN-EN dev task (train / select / score on
# MELD; EN is never a Table 2 target), so no DE / FR / CN data is touched.
#   bash scripts/check_meld_training.sh 0 1 2 3
set -euo pipefail
GPUS=${@:-0}
[ -f data/manifests/EN.csv ] || { echo "data/manifests/EN.csv missing" >&2; exit 1; }
for cw in false true; do
  for seed in 0 1 2 3 4; do
    echo "python -m xlser.train --task EN-EN --system upper_bound --out runs_dev --tag cw_$cw --set class_weighted_ce=$cw seed=$seed --wandb --wandb-project zero-shot-xling-ser-dev"
  done
done | python scripts/launch.py --gpus $GPUS --logdir logs/check_meld
python scripts/collect_results.py runs_dev --dev
