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


def _fake_meld(root, have_audio):
    import csv as _csv
    rows = {"train": [(0, 0, "joy"), (125, 3, "neutral"), (1, 0, "anger")],
            "dev": [(110, 7, "sadness"), (0, 0, "neutral")],
            "test": [(0, 0, "joy")]}
    for split, utts in rows.items():
        (root / split).mkdir(parents=True)
        with open(root / f"{split}_sent_emo.csv", "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["Sr No.", "Utterance", "Speaker", "Emotion", "Sentiment", "Dialogue_ID", "Utterance_ID"])
            for d, u, e in utts:
                w.writerow([0, "x", "Ross", e, "x", d, u])
                if f"{split}/dia{d}_utt{u}" in have_audio:
                    (root / split / f"dia{d}_utt{u}.wav").write_bytes(b"RIFF")


def _meld_protocol(protocol):
    protocol["corpora"]["EN"]["expected"] = {"train": 3, "val": 2, "test": 1}
    return protocol


def test_meld_known_missing_accepted_and_counted(tmp_path, protocol):
    import pytest
    from xlser.prepare import build_manifest
    _fake_meld(tmp_path, {"train/dia0_utt0", "train/dia1_utt0", "dev/dia0_utt0", "test/dia0_utt0"})
    rows = build_manifest("EN", tmp_path, _meld_protocol(protocol))
    assert sorted(r["utt_id"] for r in rows) == ["dev/dia0_utt0", "test/dia0_utt0", "train/dia0_utt0", "train/dia1_utt0"]
    assert all(r["path"] for r in rows)


def test_meld_unexpected_missing_is_rejected(tmp_path, protocol):
    import pytest
    from xlser.prepare import build_manifest
    # dia1_utt0 also missing -> not in known_missing -> hard error
    _fake_meld(tmp_path, {"train/dia0_utt0", "dev/dia0_utt0", "test/dia0_utt0"})
    with pytest.raises(SystemExit, match="known_missing"):
        build_manifest("EN", tmp_path, _meld_protocol(protocol))
