#!/usr/bin/env bash
# Stage 1 of the reproduction: EMO-DB only.
#   bash scripts/run_emodb_pipeline.sh /path/to/emodb   (the folder containing wav/)
# 1. builds data/manifests/DE.csv (checks 266/35/38 against the paper)
# 2. trains the DE upper bound with each source backbone used by the *->DE tasks
#    (Table 2 "Upper Bound" column for EN->DE / CN->DE / FR->DE: 97.22 / 97.22 / 95.44 UAR)
# The zero-shot *->DE runs additionally need the EN, CN, FR and UR manifests.
set -euo pipefail
ROOT=${1:?usage: $0 /path/to/emodb}
python -m xlser.prepare --lang DE --root "$ROOT"
for task in EN-DE CN-DE FR-DE; do
  python -m xlser.train --task "$task" --system upper_bound "${@:2}"
done
