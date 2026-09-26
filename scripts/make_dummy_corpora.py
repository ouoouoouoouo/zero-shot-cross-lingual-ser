"""Write tiny synthetic corpora with the real directory / file-name conventions
of all five datasets, for smoke-testing the pipeline without the real audio.

EMO-DB gets the real 4-class counts (127 W / 71 F / 62 T / 79 N, plus some
other emotions that must be dropped), so its manifest passes the exact-count
check; the others are small and need --allow-count-mismatch.
Each emotion is a different tone frequency so a model can actually learn it.

    python scripts/make_dummy_corpora.py data/dummy
"""
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

FREQ = {"neutral": 220, "happy": 440, "angry": 660, "sad": 330, "other": 550}
rng = np.random.default_rng(0)


def tone(path, emotion, sr=16000, sec=0.6):
    t = np.arange(int(sr * sec * rng.uniform(0.7, 1.3))) / sr
    x = 0.3 * np.sin(2 * np.pi * FREQ[emotion] * rng.uniform(0.95, 1.05) * t) + 0.05 * rng.standard_normal(len(t))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x.astype(np.float32), sr)


def emodb(root):
    counts = {"W": ("angry", 127), "F": ("happy", 71), "T": ("sad", 62), "N": ("neutral", 79),
              "L": ("other", 10), "A": ("other", 10)}
    speakers = ["03", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
    texts = ["a01", "a02", "a04", "a05", "a07", "b01", "b02", "b03", "b09", "b10"]
    for code, (emo, n) in counts.items():
        for k in range(n):
            spk, txt, ver = speakers[k % 10], texts[(k // 10) % 10], "abcdefg"[k // 100]
            tone(root / "wav" / f"{spk}{txt}{code}{ver}.wav", emo)


def cafe(root, per=8):
    folder = {"C": "Colère", "J": "Joie", "T": "Tristesse", "N": "Neutre", "P": "Peur"}
    emo = {"C": "angry", "J": "happy", "T": "sad", "N": "neutral", "P": "other"}
    for code in folder:
        for k in range(per):
            actor, sent = f"{k % 4 + 1:02d}", f"{k // 4 + 1:02d}"
            name = f"{actor}-N-{sent}.wav" if code == "N" else f"{actor}-{code}-1-{sent}.wav"
            sub = folder[code] if code == "N" else f"{folder[code]}/Faible"
            tone(root / sub / name, emo[code])


def esd(root, per=5):
    for spk in ["0001", "0002", "0011"]:  # 0011 is English and must be ignored
        for i, emo in enumerate(["Neutral", "Angry", "Happy", "Sad", "Surprise"]):
            for k in range(per):
                e = emo.lower() if emo != "Surprise" else "other"
                tone(root / spk / emo / "train" / f"{spk}_{i * 350 + k + 1:06d}.wav", e)


def urdu(root, per=6):
    for emo in ["Angry", "Happy", "Neutral", "Sad"]:
        for k in range(per):
            tone(root / emo / f"S{'MF'[k % 2]}{k % 3 + 1}_F{k + 1}_{emo[0]}{k + 1:02d}.wav", emo.lower())


def meld(root, per=6):
    emos = {"neutral": "neutral", "joy": "happy", "anger": "angry", "sadness": "sad", "surprise": "other"}
    for split in ["train", "dev", "test"]:
        with open(root / f"{split}_sent_emo.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Sr No.", "Utterance", "Speaker", "Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID"])
            u = 0
            for raw, emo in emos.items():
                for k in range(per):
                    w.writerow([u, "...", ["Ross", "Rachel", "Joey"][k % 3], raw, "x", k, u])
                    tone(root / split / f"dia{k}_utt{u}.wav", emo)
                    u += 1


if __name__ == "__main__":
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "data/dummy")
    for name, fn in [("emodb", emodb), ("cafe", cafe), ("esd", esd), ("urdu", urdu), ("meld", meld)]:
        (base / name).mkdir(parents=True, exist_ok=True)
        fn(base / name)
        print(f"wrote {base / name}")
