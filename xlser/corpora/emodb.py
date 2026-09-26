"""Berlin EMO-DB (German). Files: <root>/wav/03a01Fa.wav

chars 0-1 speaker, 2-4 text id, 5 emotion, 6 version. Emotion codes:
W=Ärger(anger) L=Langeweile(boredom) E=Ekel(disgust) A=Angst(fear)
F=Freude(happiness) T=Trauer(sadness) N=neutral.
4-class subset: W 127, F 71, T 62, N 79 -> 339 utterances.
"""
from pathlib import Path

EMOTION_MAP = {"W": "angry", "F": "happy", "T": "sad", "N": "neutral"}


def scan(root):
    rows = []
    for wav in sorted(Path(root).rglob("*.wav")):
        name = wav.stem
        if len(name) != 7:
            continue
        label = EMOTION_MAP.get(name[5])
        if label is None:
            continue
        rows.append({"utt_id": name, "speaker": name[:2], "label": label,
                     "path": str(wav.resolve()), "official_split": ""})
    return rows
