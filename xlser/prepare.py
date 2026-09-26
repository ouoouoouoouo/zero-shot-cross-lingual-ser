"""Build the per-language manifest data/manifests/<LANG>.csv.

    python -m xlser.prepare --lang DE --root /data/EMO-DB

The split is a pure function of (sorted file list, split_seed, expected
counts) from configs/protocol.yaml, so it is identical on every machine.
Counts that differ from the paper's are a hard error unless --allow-count-mismatch.
"""
import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .corpora import SCANNERS
from .protocol import MANIFEST_FIELDS, load_protocol, manifest_path


def _largest_remainder(sizes, total):
    """Split `total` over strata proportionally to `sizes`; returns {key: count}."""
    n = sum(sizes.values())
    quota = {k: total * s / n for k, s in sizes.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    order = sorted(sizes, key=lambda k: (-(quota[k] - alloc[k]), k))
    for k in order[: total - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def stratified_split(rows, n_val, n_test, strata, seed):
    """Assign rows["split"] in place with exactly n_val / n_test / rest."""
    groups = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["utt_id"]):
        groups[tuple(r[k] for k in strata)].append(r)
    sizes = {k: len(v) for k, v in groups.items()}
    val = _largest_remainder(sizes, n_val)
    test = _largest_remainder(sizes, n_test)
    rng = np.random.default_rng(seed)
    for key in sorted(groups):
        g = groups[key]
        if val[key] + test[key] >= len(g):
            raise ValueError(f"stratum {key} too small ({len(g)}) for val={val[key]} test={test[key]}")
        perm = rng.permutation(len(g))
        for rank, i in enumerate(perm):
            g[i]["split"] = "val" if rank < val[key] else "test" if rank < val[key] + test[key] else "train"


def build_manifest(lang, root, protocol, allow_count_mismatch=False):
    spec = protocol["corpora"][lang]
    rows = SCANNERS[spec["corpus"]](root)
    if not rows:
        raise SystemExit(f"no 4-class utterances found under {root} for {spec['corpus']}")
    exp = spec["expected"]
    if spec["split"] == "official":
        for r in rows:
            r["split"] = r["official_split"]
    else:
        # exact paper counts; on a corpus copy of a different size keep the paper's ratios
        scale = len(rows) / sum(exp.values())
        n_val, n_test = (exp["val"], exp["test"]) if scale == 1 else (round(exp["val"] * scale), round(exp["test"] * scale))
        stratified_split(rows, n_val, n_test, spec["strata"], protocol["split_seed"])
    for r in rows:
        r.update(corpus=spec["corpus"], lang=lang)

    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    msg = f"[{lang}/{spec['corpus']}] {counts} (paper: {exp})"
    if counts != exp:
        if not allow_count_mismatch:
            raise SystemExit(msg + "\ncounts differ from the paper; fix the corpus copy "
                             "or pass --allow-count-mismatch (run will not match Table 1)")
        print("WARNING " + msg, file=sys.stderr)
    else:
        print(msg)
    return sorted(rows, key=lambda r: r["utt_id"])


def write_manifest(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    # split fingerprint independent of absolute paths: commit this file
    digest = hashlib.sha256("".join(f"{r['utt_id']},{r['split']}\n" for r in rows).encode()).hexdigest()
    path.with_suffix(".split.sha256").write_text(digest + "\n")
    print(f"wrote {path} ({len(rows)} rows, split sha256 {digest[:12]})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, choices=["EN", "CN", "DE", "FR", "UR"])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--manifest-dir", default="data/manifests", type=Path)
    ap.add_argument("--allow-count-mismatch", action="store_true")
    a = ap.parse_args(argv)
    protocol = load_protocol(a.protocol)
    rows = build_manifest(a.lang, a.root, protocol, a.allow_count_mismatch)
    write_manifest(rows, manifest_path(a.manifest_dir, a.lang))


if __name__ == "__main__":
    main()
