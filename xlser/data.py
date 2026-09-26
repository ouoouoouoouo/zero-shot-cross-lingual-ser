"""Audio dataset. Normalisation is per utterance (zero mean, unit variance, as
the wav2vec 2.0 feature extractor does), so no statistics are ever pooled
across a corpus — in particular never across the target language."""
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

SR = 16000


def load_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
    return wav


def normalize(wav):
    return (wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)


class SERDataset(Dataset):
    def __init__(self, rows, labels, speakers=(), max_sec=6.0, train=False, seed=0):
        self.rows = list(rows)
        self.label_idx = {l: i for i, l in enumerate(labels)}
        self.spk_idx = {s: i for i, s in enumerate(speakers)}
        self.max_len = int(max_sec * SR)
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        wav = load_audio(r["path"])
        if len(wav) > self.max_len:
            start = self.rng.integers(0, len(wav) - self.max_len + 1) if self.train else 0
            wav = wav[start:start + self.max_len]
        return {
            "wav": torch.from_numpy(normalize(wav)),
            "label": self.label_idx[r["label"]],
            "speaker": self.spk_idx.get(f"{r['lang']}:{r['speaker']}", -100),
            "lang": r["lang"],
            "index": i,
        }


def collate(items):
    lengths = torch.tensor([len(x["wav"]) for x in items])
    wav = torch.zeros(len(items), int(lengths.max()))
    for j, x in enumerate(items):
        wav[j, : len(x["wav"])] = x["wav"]
    return {
        "wav": wav,
        "lengths": lengths,
        "label": torch.tensor([x["label"] for x in items]),
        "speaker": torch.tensor([x["speaker"] for x in items]),
        "lang": [x["lang"] for x in items],
        "index": torch.tensor([x["index"] for x in items]),
    }
