"""CaFE, Canadian French. Files like <root>/Colère/Faible/01-C-1-01.wav and
<root>/Neutre/01-N-01.wav, i.e. <actor>-<emotion>[-<intensity>]-<sentence>.

Emotion codes: C=colère(angry) D=dégoût J=joie(happy) N=neutre P=peur
S=surprise T=tristesse(sad). 4-class subset: 72 neutral + 3 x 144 = 504.
"""
from pathlib import Path

EMOTION_MAP = {"C": "angry", "J": "happy", "T": "sad", "N": "neutral"}


def scan(root):
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
