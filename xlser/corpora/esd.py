"""ESD, Mandarin subset only (speakers 0001-0010).
Files: <root>/0001/Angry/[train|evaluation|test/]0001_000351.wav

The paper uses 11200/1400/1400 (= 280/35/35 per speaker x emotion), not the
official 300/20/30, so the official sub-folders are ignored and the protocol
re-splits stratified by (speaker, label). Surprise is dropped.
"""
from pathlib import Path

EMOTION_MAP = {"angry": "angry", "happy": "happy", "sad": "sad", "neutral": "neutral"}
MANDARIN_SPEAKERS = {f"{i:04d}" for i in range(1, 11)}


def scan(root):
    rows = []
    for wav in sorted(Path(root).rglob("*.wav")):
        speaker = wav.stem.split("_")[0]
        if speaker not in MANDARIN_SPEAKERS:
            continue
        # emotion folder is the first ancestor whose name is an emotion
        label = next((EMOTION_MAP[p.name.lower()] for p in wav.parents
                      if p.name.lower() in EMOTION_MAP), None)
        if label is None:
            continue
        rows.append({"utt_id": wav.stem, "speaker": speaker, "label": label,
                     "path": str(wav.resolve()), "official_split": ""})
    return rows
