"""Full pipeline on synthetic corpora, recording every audio file that is read.
Before the final evaluation, no German (EMO-DB) file may have been opened."""
import importlib.util
import json

import pytest

from conftest import ROOT
from xlser import data, prepare, train
from xlser.protocol import load_protocol, manifest_path


@pytest.fixture(scope="module")
def dummy(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("mk", ROOT / "scripts/make_dummy_corpora.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    base = tmp_path_factory.mktemp("corpora")
    protocol = load_protocol(ROOT / "configs/protocol.yaml")
    for lang, name in [("EN", "meld"), ("CN", "esd"), ("DE", "emodb"), ("FR", "cafe"), ("UR", "urdu")]:
        (base / name).mkdir()
        getattr(mk, name)(base / name)
        rows = prepare.build_manifest(lang, base / name, protocol, allow_count_mismatch=True)
        prepare.write_manifest(rows, manifest_path(base / "manifests", lang))
    return base


def test_no_target_audio_read_before_final_evaluation(dummy, monkeypatch, tmp_path):
    reads, phase = [], {"now": "fit"}
    real_load = data.load_audio
    monkeypatch.setattr(data, "load_audio", lambda p: reads.append((phase["now"], p)) or real_load(p))
    real_fit = train.fit

    def fit_then_mark(*a, **k):
        out = real_fit(*a, **k)
        phase["now"] = "eval"
        return out

    monkeypatch.setattr(train, "fit", fit_then_mark)
    train.main(["--task", "EN-DE", "--system", "proposed", "--manifest-dir", str(dummy / "manifests"),
                "--out", str(tmp_path), "--backbone", "tiny-random",
                "--set", "max_steps=6", "eval_every=3", "num_workers=0"])
    fit_reads = [p for ph, p in reads if ph == "fit"]
    assert fit_reads and not any("/emodb/" in p for p in fit_reads)
    assert any("/emodb/" in p for ph, p in reads if ph == "eval")

    run = json.loads((tmp_path / "EN-DE/proposed_seed0/run.json").read_text())
    assert run["audit"]["early_stopping"] == ["CN", "EN", "FR", "UR"]
    assert run["audit"]["test"] == ["DE"] and run["target_test"]["n"] == 38
    assert run["target_all_supplementary"]["n"] == 339
