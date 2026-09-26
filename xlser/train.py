"""Train one (task, system) and report on the target test split.

    python -m xlser.train --task EN-DE --system proposed

Order of operations (the only order the target test rows are touched in):
  1. build_plan()   -> train / val rows from the training languages only
  2. fit()          -> sampler, speaker map, early stopping use train / val only
  3. restore best checkpoint, then evaluate on plan.test (target) exactly once
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .data import SERDataset, collate
from .losses import language_aware_supcon
from .metrics import per_language, uar_f1
from .model import SERModel
from .protocol import build_plan, load_protocol
from .sampler import HierarchicalBatchSampler, RandomBatchSampler


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, rows, plan, hp, device):
    ds = SERDataset(rows, plan.labels, max_sec=hp["max_eval_sec"], train=False)
    order = sorted(range(len(ds)), key=lambda i: rows[i]["path"])
    dl = DataLoader(torch.utils.data.Subset(ds, order), batch_size=hp["eval_batch_size"],
                    collate_fn=collate, num_workers=hp["num_workers"])
    model.eval()
    y_true, y_pred = np.zeros(len(ds), int), np.zeros(len(ds), int)
    for b in dl:
        logits = model(b["wav"].to(device), b["lengths"].to(device))["emotion"]
        y_pred[b["index"].numpy()] = logits.argmax(-1).cpu().numpy()
        y_true[b["index"].numpy()] = b["label"].numpy()
    return y_true, y_pred


def fit(model, plan, hp, device, log):
    """Uses plan.train and plan.val only."""
    train_rows = list(plan.train)
    ds = SERDataset(train_rows, plan.labels, plan.speakers, hp["max_train_sec"], train=True, seed=hp["seed"])
    bs = hp["n_lang"] * hp["n_cls"] * hp["n_sam"]
    if plan.sampler == "hierarchical":
        sampler = HierarchicalBatchSampler(train_rows, hp["n_lang"], hp["n_cls"], hp["n_sam"],
                                           num_batches=max(1, len(train_rows) // bs), seed=hp["seed"])
    else:
        sampler = RandomBatchSampler(len(train_rows), bs, seed=hp["seed"])
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=hp["num_workers"])
    lang_id = {l: i for i, l in enumerate(plan.train_langs)}

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    best, best_state, bad, history = -1.0, None, 0, []
    val_rows = list(plan.val)
    for epoch in range(hp["epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        t0, sums, steps = time.time(), {"ce": 0.0, "supcon": 0.0, "spk": 0.0}, 0
        for b in dl:
            out = model(b["wav"].to(device), b["lengths"].to(device))
            y = b["label"].to(device)
            loss = ce = F.cross_entropy(out["emotion"], y)
            sums["ce"] += ce.item()
            if plan.supcon:
                langs = torch.tensor([lang_id[l] for l in b["lang"]], device=device)
                sc = language_aware_supcon(out["h"], y, langs, hp["temperature"], hp["lambda_xling"])
                loss = loss + hp["alpha"] * sc
                sums["supcon"] += sc.item()
            if plan.spkadv:
                spk = F.cross_entropy(out["speaker"], b["speaker"].to(device))  # GRL inside the model
                loss = loss + hp["beta"] * spk
                sums["spk"] += spk.item()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, hp["grad_clip"])
            opt.step()
            steps += 1
        y_true, y_pred = predict(model, val_rows, plan, hp, device)
        val = per_language(y_true, y_pred, [r["lang"] for r in val_rows], len(plan.labels))
        rec = {"epoch": epoch, **{k: v / max(steps, 1) for k, v in sums.items()},
               "val_mean_uar": val["mean_uar"],
               "val_uar": {l: round(m["uar"], 2) for l, m in val.items() if l != "mean_uar"},
               "sec": round(time.time() - t0, 1)}
        history.append(rec)
        log(json.dumps(rec))
        if val["mean_uar"] > best:
            best, best_state, bad = val["mean_uar"], model.trainable_state_dict(), 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                log(f"early stop at epoch {epoch}")
                break
    model.load_state_dict(best_state, strict=False)
    return {"best_val_mean_uar": best, "history": history}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--system", required=True,
                    choices=["baseline1", "baseline2", "proposed", "proposed_no_spk", "proposed_no_supcon", "upper_bound"])
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--hparams", default="configs/train.yaml")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--backbone", help="override the protocol's source-language backbone (e.g. tiny-random)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override hparams, e.g. epochs=2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    protocol = load_protocol(a.protocol)
    hp = yaml.safe_load(Path(a.hparams).read_text())
    for kv in a.set:
        k, v = kv.split("=", 1)
        v = yaml.safe_load(v)
        if isinstance(v, str):  # YAML 1.1 reads "1e-3" as a string
            try:
                v = float(v)
            except ValueError:
                pass
        hp[k] = v
    seed_all(hp["seed"])

    plan = build_plan(protocol, a.task, a.system, a.manifest_dir, backbone=a.backbone)
    out = Path(a.out) / a.task / f"{a.system}_seed{hp['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    audit = plan.audit()
    log(json.dumps({"task": a.task, "system": a.system, "backbone": plan.backbone, "audit": audit}))

    model = SERModel(plan.backbone, len(plan.labels), len(plan.speakers) if plan.spkadv else 0,
                     hp["lora_rank"], hp["lora_alpha"], hp["adapter_dim"], hp["spk_hidden"],
                     hp["spk_dropout"], hp["grl_coeff"]).to(a.device)
    train_info = fit(model, plan, hp, a.device, log)

    # ---- target evaluation: first and only access to target audio ----
    n = len(plan.labels)
    y_true, y_pred = predict(model, list(plan.test), plan, hp, a.device)
    result = {"target_test": uar_f1(y_true, y_pred, n)}
    with open(out / "target_test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "label", "pred"])
        for r, p in zip(plan.test, y_pred):
            w.writerow([r["utt_id"], r["label"], plan.labels[p]])
    if plan.zero_shot:  # supplementary: whole target corpus (never trained on, so also unseen)
        yt, yp = predict(model, list(plan.target_all), plan, hp, a.device)
        result["target_all_supplementary"] = uar_f1(yt, yp, n)

    summary = {"task": a.task, "system": a.system, "backbone": plan.backbone,
               "protocol_sha256": plan.protocol_sha256, "hparams": hp, "audit": audit,
               **train_info, **result}
    (out / "run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    torch.save(model.trainable_state_dict(), out / "trainable_params.pt")
    log(f"[{a.task} {a.system}] target test UAR {result['target_test']['uar']:.2f}  "
        f"F1 {result['target_test']['f1']:.2f}  -> {out / 'run.json'}")


if __name__ == "__main__":
    main()
