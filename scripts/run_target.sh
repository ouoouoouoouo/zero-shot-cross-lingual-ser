#!/usr/bin/env bash
# All systems x 5 seeds for the three tasks of ONE target language.
#   bash scripts/run_target.sh DE 0 1 2 3
# Only after configs/train.yaml is frozen (feature_layer + lr chosen on dev tasks).
set -euo pipefail
TARGET=${1:?target language: DE, FR or CN}; shift
GPUS=${*:?"give GPU ids, e.g. 5 4"}
for l in EN CN DE FR UR; do
  [ -f data/manifests/$l.csv ] || { echo "data/manifests/$l.csv missing" >&2; exit 1; }
done
TASKS=$(python -c "import yaml; t=yaml.safe_load(open('configs/protocol.yaml'))['tasks']; print(' '.join(k for k,v in t.items() if v['target']=='$TARGET'))")
[ -n "$TASKS" ] || { echo "no task with target $TARGET" >&2; exit 1; }
for seed in 0 1 2 3 4; do
  for task in $TASKS; do
    for system in baseline1 baseline2 proposed proposed_no_spk proposed_no_supcon upper_bound; do
      echo "python -m xlser.train --task $task --system $system --set seed=$seed --wandb"
    done
  done
done | python scripts/launch.py --gpus $GPUS --logdir logs/target_$TARGET
python scripts/collect_results.py runs
