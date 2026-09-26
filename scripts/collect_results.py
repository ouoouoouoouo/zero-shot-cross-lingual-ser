"""Collect run.json files into a table.

    python scripts/collect_results.py runs          # Table 2 (mean over seeds)
    python scripts/collect_results.py runs_dev --dev  # URDU dev sweep, by tag

Table 2 refuses runs made under a different protocol file and warns when the
runs were not all trained with the same hyper-parameters (hparams_sha256).
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SYSTEMS = ["baseline1", "baseline2", "proposed", "proposed_no_spk", "proposed_no_supcon", "upper_bound"]
TASKS = ["EN-DE", "CN-DE", "FR-DE", "EN-FR", "CN-FR", "DE-FR", "EN-CN", "DE-CN", "FR-CN"]
DEV = ["EN-UR", "CN-UR", "DE-UR", "FR-UR"]

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="?", default="runs")
ap.add_argument("--dev", action="store_true")
a = ap.parse_args()

sha = hashlib.sha256(Path("configs/protocol.yaml").read_bytes()).hexdigest()
res, hps = defaultdict(list), defaultdict(set)
for f in sorted(Path(a.runs).glob("*/*/run.json")):
    r = json.loads(f.read_text())
    if r["protocol_sha256"] != sha:
        print(f"skip {f}: different protocol", file=sys.stderr)
        continue
    if bool(r.get("dev")) != a.dev:
        continue
    key = (r["task"], r.get("tag", "") if a.dev else r["system"])
    res[key].append((r["target_test"]["uar"], r["target_test"]["f1"], r["best_val_mean_uar"]))
    hp = {k: v for k, v in r["hparams"].items() if k != "seed"}
    hps[hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()].add(f"{r['task']}/{r['system']}")


def cell(v):
    return f"{np.mean([x[0] for x in v]):.2f} / {np.mean([x[1] for x in v]):.2f} (n={len(v)})" if v else "-"


if a.dev:
    tags = sorted({t for _, t in res})
    print("URDU dev: each cell = val UAR (used to choose) ; test UAR / F1")
    print("| tag | " + " | ".join(DEV) + " | mean val UAR |")
    print("|---" * (len(DEV) + 2) + "|")
    for tag in tags:
        vals = [np.mean([x[2] for x in res[(t, tag)]]) for t in DEV if res.get((t, tag))]
        cells = [f"{np.mean([x[2] for x in res[(t, tag)]]):.1f} ; {cell(res[(t, tag)])}" if res.get((t, tag)) else "-"
                 for t in DEV]
        print(f"| {tag} | " + " | ".join(cells) + f" | {np.mean(vals):.2f} |")
    print("\nChoose by mean val UAR; test columns are shown only as a sanity check.")
else:
    print("| Task | " + " | ".join(f"{s} UAR / F1" for s in SYSTEMS) + " |")
    print("|---" * (len(SYSTEMS) + 1) + "|")
    for t in TASKS:
        print(f"| {t} | " + " | ".join(cell(res.get((t, s))) for s in SYSTEMS) + " |")
    if len(hps) > 1:
        print("\nWARNING: runs use different hyper-parameters:", file=sys.stderr)
        for h, runs in hps.items():
            print(f"  {h[:12]}: {sorted(runs)}", file=sys.stderr)
