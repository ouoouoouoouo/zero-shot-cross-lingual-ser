"""How much do mean-pooled last-layer features differ between utterances?

    python scripts/inspect_backbones.py --lang UR --n 64
Uses unlabeled audio of one language (default UR, never a target). If the mean
pairwise cosine similarity is ~1, all utterances look alike to the classifier
head, which then just predicts the majority class.
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xlser.data import load_audio, normalize  # noqa: E402
from xlser.model import load_backbone  # noqa: E402
from xlser.protocol import read_manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lang", default="UR")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--manifest-dir", default="data/manifests")
a = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
rows = [r for r in read_manifest(a.manifest_dir, a.lang) if r["split"] == "train"][: a.n]
wavs = [torch.from_numpy(normalize(load_audio(r["path"])[: 16000 * 6])) for r in rows]

print(f"{'backbone':45s} {'norm':>8s} {'max|h|':>8s} {'cos(mean)':>9s} {'cos(min)':>8s} {'layer':>5s}")
for lang, name in yaml.safe_load(open("configs/protocol.yaml"))["backbones"].items():
    m = load_backbone(name).to(device).eval()
    with torch.no_grad():
        layers = [torch.stack(x) for x in zip(*[
            [h[0].mean(0) for h in m(w[None].to(device), output_hidden_states=True).hidden_states] for w in wavs])]
    for li in (len(layers) // 2, len(layers) - 1):
        h = layers[li]
        z = torch.nn.functional.normalize(h, dim=-1)
        cos = (z @ z.t())[~torch.eye(len(z), dtype=torch.bool, device=z.device)]
        print(f"{name:45s} {h.norm(dim=-1).mean():8.2f} {h.abs().max():8.2f} {cos.mean():9.4f} {cos.min():8.4f} {li:5d}")
    del m
