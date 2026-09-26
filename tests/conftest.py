import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xlser.protocol import MANIFEST_FIELDS, load_protocol  # noqa: E402

CORPUS = {"EN": "meld", "CN": "esd", "DE": "emodb", "FR": "cafe", "UR": "urdu"}


@pytest.fixture
def protocol():
    return load_protocol(ROOT / "configs/protocol.yaml")


def write_fake_manifests(d, per=3):
    """Audio-less manifests: per language x label x split, 2 speakers."""
    d.mkdir(parents=True, exist_ok=True)
    for lang, corpus in CORPUS.items():
        with open(d / f"{lang}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            for split in ("train", "val", "test"):
                for label in ("neutral", "happy", "angry", "sad"):
                    for k in range(per):
                        uid = f"{split}_{label}_{k}"
                        w.writerow({"utt_id": uid, "corpus": corpus, "lang": lang, "speaker": f"s{k % 2}",
                                    "label": label, "split": split, "path": f"/{corpus}/{uid}.wav"})
    return d


@pytest.fixture
def manifests(tmp_path):
    return write_fake_manifests(tmp_path / "manifests")
