"""Layer-wise analysis on dev data only, to choose `feature_layer` per backbone.

    python scripts/layerwise_probe.py                       # all 4 backbones, dev corpora UR + EN
    python scripts/layerwise_probe.py --backbones EN CN

Follows the layer-wise analysis of the baseline reference [14] (Han et al.
2025): the backbone is frozen, each utterance is mean-pooled at every
transformer layer, and a linear classifier is trained per layer.

Dev data (neither language is ever a Table 2 target):
  UR (URDU): train on train (300), score on val + test (100; val alone is 20 utterances)
  EN (MELD): train on train (8,244), score on val (896)
Probe: standardised features, multinomial logistic regression with balanced
class weights; metric UAR.

Selection rule (fixed before running): for each backbone, the layer with the
highest mean UAR over the two dev corpora (each corpus weighted 1/2); ties go
to the lower layer. Layer k = output of the k-th transformer layer (1..12).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xlser.data import SR, load_audio, normalize  # noqa: E402
from xlser.model import load_backbone  # noqa: E402
from xlser.protocol import load_protocol, read_manifest  # noqa: E402

DEV = {"UR": {"fit": ("train",), "score": ("val", "test")},
       "EN": {"fit": ("train",), "score": ("val",)}}


@torch.no_grad()
def extract(model, rows, device, max_sec):
    feats = []
    for i, r in enumerate(rows):
        wav = torch.from_numpy(normalize(load_audio(r["path"])[: int(max_sec * SR)]))[None].to(device)
        hs = model(wav, output_hidden_states=True).hidden_states       # (n_layers + 1) x (1, T, d)
        feats.append(torch.stack([h[0].mean(0) for h in hs]).cpu().numpy())
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{len(rows)}", flush=True)
    return np.stack(feats)                                              # (N, n_layers + 1, d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+", default=["EN", "CN", "DE", "FR"])
    ap.add_argument("--corpora", nargs="+", default=["UR", "EN"], choices=list(DEV))
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--out", default="runs_dev/layerwise", type=Path)
    ap.add_argument("--max-sec", type=float, default=15.0)
    ap.add_argument("--backbone-override", help="e.g. tiny-random (tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    protocol = load_protocol()
    targets = {t["target"] for t in protocol["tasks"].values()}
    assert not targets & set(a.corpora), "dev corpora must never be a Table 2 target"
    labels = protocol["labels"]
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    results = {}
    for bb in a.backbones:
        name = a.backbone_override or protocol["backbones"][bb]
        print(f"[{bb}] {name}", flush=True)
        model = load_backbone(name).to(a.device).eval()
        per_corpus = {}
        for corpus in a.corpora:
            rows = read_manifest(a.manifest_dir, corpus)
            cache = a.out / f"{bb}_{corpus}.npz"
            if cache.exists() and not a.backbone_override:
                z = np.load(cache)
                X, y, split = z["X"], z["y"], z["split"]
            else:
                print(f"  extracting {corpus} ({len(rows)} utterances)", flush=True)
                X = extract(model, rows, a.device, a.max_sec)
                y = np.array([labels.index(r["label"]) for r in rows])
                split = np.array([r["split"] for r in rows])
                if not a.backbone_override:
                    np.savez(cache, X=X, y=y, split=split)
            fit = np.isin(split, DEV[corpus]["fit"])
            score = np.isin(split, DEV[corpus]["score"])
            uars = []
            for k in range(1, X.shape[1]):
                sc = StandardScaler().fit(X[fit, k])
                clf = LogisticRegression(max_iter=3000, class_weight="balanced")
                clf.fit(sc.transform(X[fit, k]), y[fit])
                pred = clf.predict(sc.transform(X[score, k]))
                uars.append(100 * recall_score(y[score], pred, average="macro"))
            per_corpus[corpus] = uars
        del model
        torch.cuda.empty_cache()
        mean = np.mean([per_corpus[c] for c in a.corpora], axis=0)
        best = int(np.argmax(mean)) + 1                                  # argmax takes the first (lowest) on ties
        results[bb] = {"uar": per_corpus, "mean": mean.tolist(), "best_layer": best}

        print(f"\n  layer | " + " | ".join(f"{c:>6s}" for c in a.corpora) + " |  mean")
        for k in range(len(mean)):
            mark = "  <- best" if k + 1 == best else ""
            print(f"  {k + 1:5d} | " + " | ".join(f"{per_corpus[c][k]:6.1f}" for c in a.corpora)
                  + f" | {mean[k]:5.1f}{mark}")
        print(flush=True)

    (a.out / "selection.json").write_text(json.dumps(results, indent=2))
    print("feature_layer (paste into configs/train.yaml):")
    print(yaml.safe_dump({"feature_layer": {bb: r["best_layer"] for bb, r in results.items()}}, sort_keys=False))


if __name__ == "__main__":
    main()
