#!/usr/bin/env bash
# All 9 tasks x 6 systems (Table 2). Needs all five manifests. Extra args go to xlser.train,
# e.g.  bash scripts/run_all.sh --set seed=1
set -euo pipefail
for task in EN-DE CN-DE FR-DE EN-FR CN-FR DE-FR EN-CN DE-CN FR-CN; do
  for system in baseline1 baseline2 proposed proposed_no_spk proposed_no_supcon upper_bound; do
    python -m xlser.train --task "$task" --system "$system" "$@"
  done
done
python scripts/collect_results.py runs
