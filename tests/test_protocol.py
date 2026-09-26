import csv

import pytest

from xlser.protocol import LeakageError, build_plan
from xlser.sampler import HierarchicalBatchSampler

ZERO_SHOT = ["baseline1", "baseline2", "proposed", "proposed_no_spk", "proposed_no_supcon"]


@pytest.mark.parametrize("system", ZERO_SHOT)
def test_target_never_reaches_training_stages(protocol, manifests, system):
    for task in protocol["tasks"]:
        plan = build_plan(protocol, task, system, manifests, backbone="tiny-random")
        tgt = plan.target
        assert tgt not in plan.train_langs
        assert all(r["lang"] != tgt for r in plan.train + plan.val)
        assert not any(s.startswith(tgt + ":") for s in plan.speakers)
        assert {r["lang"] for r in plan.test} == {tgt}
        assert all(r["split"] == "test" for r in plan.test)
        audit = plan.audit()
        for stage in ("train", "sampler", "speaker_classifier", "early_stopping"):
            assert tgt not in audit[stage], (task, system, stage)


def test_train_language_sets(protocol, manifests):
    p = build_plan(protocol, "EN-DE", "baseline1", manifests)
    assert p.train_langs == ("EN",)
    p = build_plan(protocol, "EN-DE", "proposed", manifests)
    assert set(p.train_langs) == {"EN", "CN", "FR", "UR"} and p.source == "EN"
    assert p.backbone == "facebook/wav2vec2-base-960h"   # source-language backbone, never the target's


def test_upper_bound_is_target_only(protocol, manifests):
    p = build_plan(protocol, "EN-DE", "upper_bound", manifests)
    assert not p.zero_shot and set(p.train_langs) == {"DE"}
    assert {r["lang"] for r in p.train + p.val + p.test} == {"DE"}
    assert not {r["path"] for r in p.train + p.val} & {r["path"] for r in p.test}


def test_smuggled_target_row_is_rejected(protocol, manifests):
    # a German EMO-DB row hidden inside the French manifest
    with open(manifests / "FR.csv", "a", newline="") as f:
        csv.writer(f).writerow(["x", "emodb", "DE", "03", "angry", "train", "/emodb/x.wav"])
    with pytest.raises(LeakageError):
        build_plan(protocol, "EN-DE", "proposed", manifests)


def test_target_corpus_relabelled_as_other_language_is_rejected(protocol, manifests):
    with open(manifests / "FR.csv", "a", newline="") as f:
        csv.writer(f).writerow(["x", "emodb", "FR", "03", "angry", "train", "/emodb/x.wav"])
    with pytest.raises(LeakageError):
        build_plan(protocol, "EN-DE", "proposed", manifests)


def test_target_file_reused_under_other_corpus_is_rejected(protocol, manifests):
    with open(manifests / "FR.csv", "a", newline="") as f:
        csv.writer(f).writerow(["x", "cafe", "FR", "03", "angry", "train", "/emodb/test_sad_0.wav"])
    with pytest.raises(LeakageError):
        build_plan(protocol, "EN-DE", "proposed", manifests)


def test_hierarchical_sampler_on_plan(protocol, manifests):
    plan = build_plan(protocol, "EN-DE", "proposed", manifests)
    s = HierarchicalBatchSampler(plan.train, 3, 4, 3, num_batches=20)
    for batch in s:
        rows = [plan.train[i] for i in batch]
        assert len(batch) == 36
        assert len({r["lang"] for r in rows}) == 3
        assert all(r["lang"] != "DE" for r in rows)
        for lang in {r["lang"] for r in rows}:
            for label in plan.labels:
                assert sum(r["lang"] == lang and r["label"] == label for r in rows) == 3
