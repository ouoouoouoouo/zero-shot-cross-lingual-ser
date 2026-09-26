#!/usr/bin/env bash
# Hyper-parameter selection on the URDU dev tasks only (UR is never a target).
#   bash scripts/tune_on_urdu.sh 0 1 2 3 4 5          # GPU ids
# Sweeps lr for every source backbone x 2 seeds, then prints the dev table.
# Pick ONE lr (best mean over backbones) and write it into configs/train.yaml.
set -euo pipefail
GPUS=${@:-0}
[ -f data/manifests/UR.csv ] || { echo "data/manifests/UR.csv missing: python -m xlser.prepare --lang UR --root <dir>" >&2; exit 1; }
for lr in 1e-4 3e-4 1e-3 3e-3; do
  for task in EN-UR CN-UR DE-UR FR-UR; do
    for seed in 0 1; do
      echo "python -m xlser.train --task $task --system upper_bound --out runs_dev --tag lr$lr --set lr=$lr seed=$seed --wandb --wandb-project zero-shot-xling-ser-dev"
    done
  done
done | python scripts/launch.py --gpus $GPUS --logdir logs/tune_urdu
python scripts/collect_results.py runs_dev --dev
