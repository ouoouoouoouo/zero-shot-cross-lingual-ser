"""URDU dataset (Latif et al. 2018). Files: <root>/{Angry,Happy,Neutral,Sad}/SM1_F10_A01.wav
The first underscore-separated token is the speaker (38 speakers, 100 utts per class).
"""
from pathlib import Path

EMOTION_MAP = {"angry": "angry", "happy": "happy", "sad": "sad", "neutral": "neutral"}


def scan(root):
    rows = []
    for wav in sorted(Path(root).rglob("*.wav")):
        label = EMOTION_MAP.get(wav.parent.name.lower())
        if label is None:
            continue
        rows.append({"utt_id": f"{wav.parent.name}/{wav.stem}", "speaker": wav.stem.split("_")[0],
                     "label": label, "path": str(wav.resolve()), "official_split": ""})
    return rows
