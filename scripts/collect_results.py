"""Collect runs/<task>/<system>_seed*/run.json into a Table-2-style markdown table
(mean over seeds). Refuses runs made under a different protocol file."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SYSTEMS = ["baseline1", "baseline2", "proposed", "proposed_no_spk", "proposed_no_supcon", "upper_bound"]
TASKS = ["EN-DE", "CN-DE", "FR-DE", "EN-FR", "CN-FR", "DE-FR", "EN-CN", "DE-CN", "FR-CN"]

runs = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
sha = hashlib.sha256(Path("configs/protocol.yaml").read_bytes()).hexdigest()
res = defaultdict(list)
for f in runs.glob("*/*/run.json"):
    r = json.loads(f.read_text())
    if r["protocol_sha256"] != sha:
        print(f"skip {f}: different protocol", file=sys.stderr)
        continue
    res[(r["task"], r["system"])].append((r["target_test"]["uar"], r["target_test"]["f1"]))

print("| Task | " + " | ".join(f"{s} UAR / F1" for s in SYSTEMS) + " |")
print("|---" * (len(SYSTEMS) + 1) + "|")
for t in TASKS:
    cells = []
    for s in SYSTEMS:
        v = res.get((t, s))
        cells.append(f"{np.mean([u for u, _ in v]):.2f} / {np.mean([f for _, f in v]):.2f} (n={len(v)})" if v else "-")
    print(f"| {t} | " + " | ".join(cells) + " |")
