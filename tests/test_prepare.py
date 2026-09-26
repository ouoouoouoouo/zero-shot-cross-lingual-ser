from collections import Counter

from xlser.corpora import cafe, emodb
from xlser.prepare import stratified_split


def emodb_like():
    n = {"angry": 127, "happy": 71, "sad": 62, "neutral": 79}
    return [{"utt_id": f"{l}{i:03d}", "label": l} for l, k in n.items() for i in range(k)]


def test_emodb_exact_paper_counts_and_deterministic():
    a, b = emodb_like(), emodb_like()
    stratified_split(a, 35, 38, ["label"], seed=1)
    stratified_split(b, 35, 38, ["label"], seed=1)
    assert Counter(r["split"] for r in a) == {"train": 266, "val": 35, "test": 38}
    assert [r["split"] for r in a] == [r["split"] for r in b]
    for split in ("val", "test"):  # every class present
        assert set(r["label"] for r in a if r["split"] == split) == {"angry", "happy", "sad", "neutral"}


def test_emodb_scanner(tmp_path):
    for name in ["03a01Wa", "03a01Fa", "08b02Tc", "16a05Nb", "03a01La", "03a01Ea", "11a02Ab"]:
        (tmp_path / f"{name}.wav").touch()
    rows = {r["utt_id"]: r for r in emodb.scan(tmp_path)}
    assert {k: v["label"] for k, v in rows.items()} == {
        "03a01Wa": "angry", "03a01Fa": "happy", "08b02Tc": "sad", "16a05Nb": "neutral"}
    assert rows["08b02Tc"]["speaker"] == "08"


def test_cafe_scanner(tmp_path):
    (tmp_path / "Neutre").mkdir()
    (tmp_path / "Neutre/01-N-01.wav").touch()
    (tmp_path / "Joie/Fort").mkdir(parents=True)
    (tmp_path / "Joie/Fort/12-J-2-06.wav").touch()
    (tmp_path / "Joie/Fort/12-P-2-06.wav").touch()
    assert sorted((r["speaker"], r["label"]) for r in cafe.scan(tmp_path)) == [("01", "neutral"), ("12", "happy")]
