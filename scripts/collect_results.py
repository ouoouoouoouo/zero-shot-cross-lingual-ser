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

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="?", default="runs")
ap.add_argument("--dev", action="store_true")
ap.add_argument("--all-target", action="store_true",
                help="zero-shot rows: score on the whole target corpus (target_all_supplementary) instead of its test split")
a = ap.parse_args()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xlser.protocol import load_protocol  # noqa: E402

protocol = load_protocol("configs/protocol.yaml")
sha = protocol["_sha256"]
DEV = list(protocol["dev_tasks"])
res, hps = defaultdict(list), defaultdict(set)
for f in sorted(Path(a.runs).glob("*/*/run.json")):
    r = json.loads(f.read_text())
    if r["protocol_sha256"] != sha and not r.get("dev"):
        print(f"skip {f}: different protocol", file=sys.stderr)
        continue
    if bool(r.get("dev")) != a.dev:
        continue
    key = (r["task"], r.get("tag", "") if a.dev else r["system"])
    m = r.get("target_all_supplementary") if a.all_target and not a.dev else None
    m = m or r["target_test"]  # upper bound has no supplementary score: always its test split
    res[key].append((m["uar"], m["f1"], r["best_val_mean_uar"]))
    hp = {k: v for k, v in r["hparams"].items() if k != "seed"}
    hps[hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()].add(f"{r['task']}/{r['system']}")


def cell(v):
    if not v:
        return "-"
    u, f = np.array([x[0] for x in v]), np.array([x[1] for x in v])
    sd = lambda x: f"±{x.std(ddof=1):.1f}" if len(x) > 1 else ""
    return f"{u.mean():.2f}{sd(u)} / {f.mean():.2f}{sd(f)} (n={len(v)})"


if a.dev:
    tags = sorted({t for _, t in res})
    print("dev tasks: each cell = val UAR (used to choose) ; test UAR / F1")
    DEV = [t for t in DEV if any(k[0] == t for k in res)]
    print("| tag | " + " | ".join(DEV) + " | mean val UAR |")
    print("|---" * (len(DEV) + 2) + "|")
    for tag in tags:
        vals = [np.mean([x[2] for x in res[(t, tag)]]) for t in DEV if res.get((t, tag))]
        cells = [f"{np.mean([x[2] for x in res[(t, tag)]]):.1f} ; {cell(res[(t, tag)])}" if res.get((t, tag)) else "-"
                 for t in DEV]
        print(f"| {tag} | " + " | ".join(cells) + f" | {np.mean(vals):.2f} |")
    print("\nChoose by mean val UAR; test columns are shown only as a sanity check.")
else:
    if a.all_target:
        print("zero-shot columns: whole target corpus (supplementary); upper_bound: target test split")
    print("| Task | " + " | ".join(f"{s} UAR / F1" for s in SYSTEMS) + " |")
    print("|---" * (len(SYSTEMS) + 1) + "|")
    for t in TASKS:
        print(f"| {t} | " + " | ".join(cell(res.get((t, s))) for s in SYSTEMS) + " |")
    if len(hps) > 1:
        print("\nWARNING: runs use different hyper-parameters:", file=sys.stderr)
        for h, runs in hps.items():
            print(f"  {h[:12]}: {sorted(runs)}", file=sys.stderr)
