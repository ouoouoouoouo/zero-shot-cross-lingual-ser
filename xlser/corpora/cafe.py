"""CaFE, Canadian French. Files like <root>/Colère/Faible/01-C-1-01.wav and
<root>/Neutre/01-N-01.wav, i.e. <actor>-<emotion>[-<intensity>]-<sentence>.

Emotion codes: C=colère(angry) D=dégoût J=joie(happy) N=neutre P=peur
S=surprise T=tristesse(sad). 4-class subset: 72 neutral + 3 x 144 = 504.

The CAMEO release replaces file names by hashes; export it with
scripts/cameo_to_wav.py, which writes <root>/cameo_metadata.csv, read here instead.
"""
import csv
from pathlib import Path

EMOTION_MAP = {"C": "angry", "J": "happy", "T": "sad", "N": "neutral"}
CAMEO_EMOTION_MAP = {"anger": "angry", "angry": "angry", "happiness": "happy", "happy": "happy", "joy": "happy",
                     "sadness": "sad", "sad": "sad", "neutral": "neutral"}


def scan_cameo(root):
    rows = []
    with open(Path(root) / "cameo_metadata.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = CAMEO_EMOTION_MAP.get(r["emotion"].strip().lower())
            if label is None:
                continue
            rows.append({"utt_id": r["utt_id"], "speaker": r["speaker"].removeprefix("cafe_"), "label": label,
                         "path": str((Path(root) / r["path"]).resolve()), "official_split": ""})
    return rows


def scan(root):
    if (Path(root) / "cameo_metadata.csv").exists():
        return scan_cameo(root)
    rows = []
    for wav in sorted(Path(root).rglob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) not in (3, 4):
            continue
        label = EMOTION_MAP.get(parts[1])
        if label is None:
            continue
        rows.append({"utt_id": wav.stem, "speaker": parts[0], "label": label,
                     "path": str(wav.resolve()), "official_split": ""})
    return rows
