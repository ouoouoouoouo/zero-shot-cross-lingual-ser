"""Export one corpus from a CAMEO parquet file to 16 kHz mono wav + metadata.

    python scripts/cameo_to_wav.py ~/dataset/CAMEO/CAMEO/data/cafe-00000-of-00001.parquet ~/dataset/CaFE_cameo

Writes <out>/<emotion>/<file_id>.wav and <out>/cameo_metadata.csv
(utt_id, speaker, emotion, transcription, path). The corpus scanner
(xlser/corpora/cafe.py) reads that CSV when it is present, because CAMEO
replaces the original file names (which encode speaker / emotion) by hashes.
Prints per-emotion counts so they can be checked against the original corpus.
"""
import argparse
import csv
import io
from collections import Counter
from math import gcd
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("parquet", type=Path)
ap.add_argument("out", type=Path)
a = ap.parse_args()

pf = pq.ParquetFile(a.parquet)
a.out.mkdir(parents=True, exist_ok=True)
meta, emotions, speakers, rates = [], Counter(), Counter(), Counter()
for i in range(pf.num_row_groups):
    for r in pf.read_row_group(i, columns=["file_id", "audio", "emotion", "speaker_id", "transcription"]).to_pylist():
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        rates[sr] += 1
        if sr != SR:
            g = gcd(sr, SR)
            wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
        utt = Path(r["file_id"]).stem
        rel = Path(r["emotion"]) / f"{utt}.wav"
        (a.out / rel.parent).mkdir(exist_ok=True)
        sf.write(a.out / rel, wav, SR)
        meta.append({"utt_id": utt, "speaker": r["speaker_id"], "emotion": r["emotion"],
                     "transcription": r["transcription"], "path": str(rel)})
        emotions[r["emotion"]] += 1
        speakers[r["speaker_id"]] += 1

with open(a.out / "cameo_metadata.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(meta[0]))
    w.writeheader()
    w.writerows(sorted(meta, key=lambda m: m["utt_id"]))
print(f"wrote {len(meta)} wav to {a.out}")
print("source sample rates:", dict(rates))
print("emotions:", dict(sorted(emotions.items())))
print(f"speakers ({len(speakers)}):", dict(sorted(speakers.items())))
