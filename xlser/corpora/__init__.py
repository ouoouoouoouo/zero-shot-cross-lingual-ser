"""Corpus scanners.

Each scanner walks a corpus root and returns one dict per utterance that falls
into the shared 4-class label space (neutral / happy / angry / sad); every
other emotion is dropped here, so nothing downstream can see it.

    {"utt_id": str, "speaker": str, "label": str, "path": str, "official_split": str}

`official_split` is "" unless the corpus ships its own partition (MELD).
"""
from . import cafe, emodb, esd, meld, urdu

SCANNERS = {
    "emodb": emodb.scan,
    "cafe": cafe.scan,
    "esd": esd.scan,
    "meld": meld.scan,
    "urdu": urdu.scan,
}
