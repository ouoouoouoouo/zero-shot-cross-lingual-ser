#!/usr/bin/env bash
# All 9 tasks x 6 systems x 3 seeds (Table 2), one job per GPU.
#   bash scripts/run_all.sh 0 1 2 3 4 5
# Only run this after configs/train.yaml has been frozen from the URDU dev sweep.
set -euo pipefail
GPUS=${@:-0}
for l in EN CN DE FR UR; do
  [ -f data/manifests/$l.csv ] || { echo "data/manifests/$l.csv missing" >&2; exit 1; }
done
for seed in 0 1 2; do
  for task in EN-DE CN-DE FR-DE EN-FR CN-FR DE-FR EN-CN DE-CN FR-CN; do
    for system in baseline1 baseline2 proposed proposed_no_spk proposed_no_supcon upper_bound; do
      echo "python -m xlser.train --task $task --system $system --set seed=$seed --wandb"
    done
  done
done | python scripts/launch.py --gpus $GPUS --logdir logs/table2
python scripts/collect_results.py runs
