"""The zero-shot protocol: which rows each stage of a run is allowed to see.

`build_plan` is the only place that reads manifests. It returns a RunPlan whose
`train` / `val` rows are drawn exclusively from the system's training languages
and whose `test` rows are the target language's test split. `check_isolation`
re-verifies this from the rows themselves and raises on any violation, and the
trainer only ever builds its sampler, speaker map and early-stopping set from
`plan.train` / `plan.val`.
"""
import csv
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MANIFEST_FIELDS = ["utt_id", "corpus", "lang", "speaker", "label", "split", "path"]
LANGS = ("EN", "CN", "DE", "FR", "UR")


class LeakageError(RuntimeError):
    """Target-language data reached a stage it must not see."""


def load_protocol(path="configs/protocol.yaml"):
    raw = Path(path).read_bytes()
    protocol = yaml.safe_load(raw)
    protocol["_sha256"] = hashlib.sha256(raw).hexdigest()
    return protocol


def manifest_path(manifest_dir, lang):
    return Path(manifest_dir) / f"{lang}.csv"


def read_manifest(manifest_dir, lang):
    path = manifest_path(manifest_dir, lang)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m xlser.prepare --lang {lang} --root ...` first")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bad = {r["lang"] for r in rows} - {lang}
    if bad:
        raise LeakageError(f"{path} contains rows of other languages: {bad}")
    return rows


@dataclass(frozen=True)
class RunPlan:
    task: str
    system: str
    dev: bool
    source: str
    target: str
    non_target: tuple
    train_langs: tuple
    zero_shot: bool
    backbone: str
    sampler: str
    supcon: bool
    spkadv: bool
    labels: tuple
    train: tuple = field(repr=False)
    val: tuple = field(repr=False)
    test: tuple = field(repr=False)          # target test split: the reported number
    target_all: tuple = field(repr=False)    # all target rows (supplementary, zero-shot only)
    speakers: tuple = field(repr=False)      # speaker-classifier classes, from `train` only
    protocol_sha256: str = ""

    def audit(self):
        """Human-readable record of what each stage saw (written to run.json)."""
        langs = lambda rows: sorted({r["lang"] for r in rows})
        return {
            "dev": self.dev,
            "zero_shot": self.zero_shot,
            "target": self.target,
            "train": langs(self.train),
            "sampler": langs(self.train),
            "speaker_classifier": sorted({s.split(":")[0] for s in self.speakers}),
            "normalization": "per-utterance zero-mean/unit-variance; no corpus statistics",
            "early_stopping": langs(self.val),
            "test": langs(self.test),
            "n_train": len(self.train), "n_val": len(self.val), "n_test": len(self.test),
            "n_speakers": len(self.speakers),
        }


def speaker_key(row):
    return f"{row['lang']}:{row['speaker']}"


def build_plan(protocol, task, system, manifest_dir="data/manifests", backbone=None):
    dev = task in protocol["dev_tasks"]
    if dev and system != "upper_bound":
        raise ValueError(f"dev task {task} is for hyper-parameter selection and only runs upper_bound")
    if not dev and task not in protocol["tasks"]:
        raise KeyError(f"unknown task {task}; choose from {list(protocol['tasks']) + list(protocol['dev_tasks'])}")
    t = protocol["dev_tasks" if dev else "tasks"][task]
    sysc = protocol["systems"][system]
    source, target = t["source"], t["target"]
    non_target = tuple(l for l in protocol["corpora"] if l not in (source, target))
    roles = {"source": (source,), "non_target": non_target, "target": (target,)}
    train_langs = tuple(l for role in sysc["train_langs"] for l in roles[role])
    zero_shot = sysc.get("zero_shot", True)
    if zero_shot and target in train_langs:
        raise LeakageError(f"system {system} lists the target {target} as a training language")

    train, val = [], []
    for lang in train_langs:
        rows = read_manifest(manifest_dir, lang)
        train += [r for r in rows if r["split"] == "train"]
        val += [r for r in rows if r["split"] == "val"]
    target_rows = read_manifest(manifest_dir, target)
    test = [r for r in target_rows if r["split"] == "test"]

    plan = RunPlan(
        task=task, system=system, dev=dev, source=source, target=target, non_target=non_target,
        train_langs=train_langs, zero_shot=zero_shot,
        backbone=backbone or protocol["backbones"][source],
        sampler=sysc["sampler"], supcon=sysc["supcon"], spkadv=sysc["spkadv"],
        labels=tuple(protocol["labels"]),
        train=tuple(train), val=tuple(val), test=tuple(test),
        target_all=tuple(target_rows) if zero_shot else (),
        speakers=tuple(sorted({speaker_key(r) for r in train})),
        protocol_sha256=protocol["_sha256"],
    )
    check_isolation(plan, protocol)
    return plan


def check_isolation(plan, protocol):
    labels = set(plan.labels)
    for name, rows in (("train", plan.train), ("val", plan.val), ("test", plan.test)):
        if not rows:
            raise ValueError(f"{name} split is empty")
        if {r["label"] for r in rows} - labels:
            raise ValueError(f"{name} has labels outside {sorted(labels)}")
    if any(r["split"] != "train" for r in plan.train) or any(r["split"] != "val" for r in plan.val):
        raise LeakageError("train/val rows carry the wrong split tag")
    if any(r["lang"] != plan.target or r["split"] != "test" for r in plan.test):
        raise LeakageError("test rows must be the target language's test split")

    if plan.zero_shot:
        tcorpus = protocol["corpora"][plan.target]["corpus"]
        for name, rows in (("train", plan.train), ("val", plan.val)):
            hit = [r for r in rows if r["lang"] == plan.target or r["corpus"] == tcorpus]
            if hit:
                raise LeakageError(f"{len(hit)} target ({plan.target}/{tcorpus}) rows in {name}")
        if any(s.startswith(plan.target + ":") for s in plan.speakers):
            raise LeakageError("target speakers in the speaker classifier")
        seen = {r["path"] for r in plan.train + plan.val}
        if seen & {r["path"] for r in plan.target_all}:
            raise LeakageError("target audio files appear in train/val")
    else:
        if set(plan.train_langs) != {plan.target}:
            raise ValueError("non-zero-shot systems (upper bound) train on the target only")
        if {r["path"] for r in plan.train + plan.val} & {r["path"] for r in plan.test}:
            raise LeakageError("upper bound: test files overlap train/val")

    # Table 1 reproduction check (source + non-target systems only)
    t = protocol["tasks"].get(plan.task, {})
    if plan.zero_shot and len(plan.train_langs) == len(protocol["corpora"]) - 1:
        got = (len(plan.train), len(plan.speakers))
        exp = (t["expected_train"], t["expected_speakers"])
        if got != exp:
            print(f"WARNING {plan.task}: #samples/#spk = {got}, paper Table 1 = {exp}", file=sys.stderr)
