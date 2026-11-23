"""MELD (English), audio only, official train/dev/test split.

Expected layout (wav extracted from the MELD.Raw mp4s, see scripts/meld_mp4_to_wav.sh):
  <root>/train_sent_emo.csv   <root>/train/dia0_utt0.wav
  <root>/dev_sent_emo.csv     <root>/dev/dia0_utt0.wav
  <root>/test_sent_emo.csv    <root>/test/dia0_utt0.wav
4-class subset: joy->happy, anger->angry, sadness->sad, neutral.
Rows whose audio is missing come back with missing=True and no path; prepare.py
drops them and checks them against `known_missing` in configs/protocol.yaml.
"""
import csv
from pathlib import Path

EMOTION_MAP = {"joy": "happy", "anger": "angry", "sadness": "sad", "neutral": "neutral"}
SPLITS = {"train": "train", "dev": "val", "test": "test"}


def scan(root):
    root = Path(root)
    rows = []
    for name, split in SPLITS.items():
        with open(root / f"{name}_sent_emo.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                label = EMOTION_MAP.get(r["Emotion"].strip().lower())
                if label is None:
                    continue
                stem = f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}"
                wav = root / name / f"{stem}.wav"
                ok = wav.exists() and wav.stat().st_size > 0
                rows.append({"utt_id": f"{name}/{stem}", "speaker": r["Speaker"].strip(), "label": label,
                             "path": str(wav.resolve()) if ok else "", "official_split": split,
                             "missing": not ok})
    return rows
