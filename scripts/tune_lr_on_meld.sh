#!/usr/bin/env bash
# lr sweep on the EN-EN dev task (MELD only; EN is never a Table 2 target).
#   bash scripts/tune_lr_on_meld.sh 0 1 2 3
# Selection rule (fixed before looking at results, see CLAUDE.md):
#   score(lr) = ( mean val UAR over EN/CN/DE/FR-UR  +  val UAR on EN-EN ) / 2
set -euo pipefail
GPUS=${*:?"give GPU ids, e.g. 5 4"}
[ -f data/manifests/EN.csv ] || { echo "data/manifests/EN.csv missing" >&2; exit 1; }
for lr in 1e-4 3e-4 1e-3 3e-3; do
  for seed in 0 1 2 3 4; do
    echo "python -m xlser.train --task EN-EN --system upper_bound --out runs_dev --tag lr$lr --set lr=$lr seed=$seed --wandb --wandb-project zero-shot-xling-ser-dev"
  done
done | python scripts/launch.py --gpus $GPUS --logdir logs/tune_meld
python scripts/collect_results.py runs_dev --dev --lr-score
