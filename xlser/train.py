"""Train one (task, system) and report on the target test split.

    python -m xlser.train --task EN-DE --system proposed

Order of operations (the only order the target test rows are touched in):
  1. build_plan()   -> train / val rows from the training languages only
  2. fit()          -> sampler, speaker map, early stopping use train / val only
  3. restore best checkpoint, then evaluate on plan.test (target) exactly once
"""
import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
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


def hparams_sha256(hp):
    """Fingerprint of everything that should be identical across a results table (seed excluded)."""
    return hashlib.sha256(json.dumps({k: v for k, v in hp.items() if k != "seed"}, sort_keys=True).encode()).hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        return ""


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, rows, plan, hp, device, batch_size=None):
    """batch_size=1 makes every prediction independent of the other utterances:
    group-norm wav2vec 2.0 checkpoints run without an attention mask, so zero
    padding (i.e. the other utterances' lengths) would otherwise leak into each
    output. Used for all target evaluation; val batches are length-sorted."""
    ds = SERDataset(rows, plan.labels, max_sec=hp["max_eval_sec"], train=False)
    order = sorted(range(len(ds)), key=lambda i: (sf.info(rows[i]["path"]).frames / sf.info(rows[i]["path"]).samplerate,
                                                  rows[i]["path"]))
    dl = DataLoader(torch.utils.data.Subset(ds, order), batch_size=batch_size or hp["eval_batch_size"],
                    collate_fn=collate, num_workers=hp["num_workers"])
    model.eval()
    y_true, y_pred = np.zeros(len(ds), int), np.zeros(len(ds), int)
    for b in dl:
        logits = model(b["wav"].to(device), b["lengths"].to(device))["emotion"]
        y_pred[b["index"].numpy()] = logits.argmax(-1).cpu().numpy()
        y_true[b["index"].numpy()] = b["label"].numpy()
    return y_true, y_pred


def lr_lambda(hp):
    """Linear warm-up, then cosine decay to 0 at max_steps."""
    warm, total = hp["warmup_steps"], hp["max_steps"]

    def f(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))
    return f


def batches(dl, sampler):
    """Endless stream of training batches; reshuffles every pass."""
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        yield from dl
        epoch += 1


def fit(model, plan, hp, device, log, tracker=None):
    """Uses plan.train and plan.val only. `tracker` is an optional wandb run.

    Budget is counted in optimizer steps (not epochs), so every system gets the
    same number of updates regardless of how much training data it has. The val
    set is scored every `eval_every` steps; training stops after `patience`
    evaluations without improvement or at `max_steps`."""
    train_rows = list(plan.train)
    ds = SERDataset(train_rows, plan.labels, plan.speakers, hp["max_train_sec"], train=True, seed=hp["seed"])
    bs = hp["n_lang"] * hp["n_cls"] * hp["n_sam"]
    if plan.sampler == "hierarchical":
        sampler = HierarchicalBatchSampler(train_rows, hp["n_lang"], hp["n_cls"], hp["n_sam"],
                                           num_batches=max(1, len(train_rows) // bs), seed=hp["seed"])
    else:
        sampler = RandomBatchSampler(len(train_rows), bs, seed=hp["seed"])
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=hp["num_workers"],
                    persistent_workers=hp["num_workers"] > 0, worker_init_fn=ds.init_worker)
    lang_id = {l: i for i, l in enumerate(plan.train_langs)}

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(hp))
    losses = ["ce"] + (["supcon"] if plan.supcon else []) + (["spk"] if plan.spkadv else [])
    best, best_step, best_state, bad, history = -1.0, 0, None, 0, []
    val_rows = list(plan.val)
    sums, n, t0 = dict.fromkeys(losses + ["grad_norm", "h_norm"], 0.0), 0, time.time()
    model.train()
    for step, b in enumerate(batches(dl, sampler), start=1):
        out = model(b["wav"].to(device), b["lengths"].to(device))
        y = b["label"].to(device)
        terms = {"ce": F.cross_entropy(out["emotion"], y)}
        loss = terms["ce"]
        if plan.supcon:
            langs = torch.tensor([lang_id[l] for l in b["lang"]], device=device)
            terms["supcon"] = language_aware_supcon(out["h"], y, langs, hp["temperature"], hp["lambda_xling"])
            loss = loss + hp["alpha"] * terms["supcon"]
        if plan.spkadv:
            terms["spk"] = F.cross_entropy(out["speaker"], b["speaker"].to(device))  # GRL inside the model
            loss = loss + hp["beta"] * terms["spk"]
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, hp["grad_clip"])
        opt.step()
        sched.step()
        for k, v in terms.items():
            sums[k] += v.item()
        sums["grad_norm"] += gn.item()
        sums["h_norm"] += out["h"].detach().norm(dim=-1).mean().item()
        n += 1
        if step % hp["eval_every"] and step != hp["max_steps"]:
            continue

        y_true, y_pred = predict(model, val_rows, plan, hp, device)
        model.train()
        val = per_language(y_true, y_pred, [r["lang"] for r in val_rows], len(plan.labels))
        rec = {"step": step, **{k: round(v / n, 4) for k, v in sums.items()}, "lr": sched.get_last_lr()[0],
               "val_mean_uar": round(val["mean_uar"], 2),
               "val_uar": {l: round(m["uar"], 2) for l, m in val.items() if l != "mean_uar"},
               "sec": round(time.time() - t0, 1)}
        history.append(rec)
        log(json.dumps(rec))
        if tracker is not None:
            tracker.log({**{f"train/{k}": v / n for k, v in sums.items()}, "train/lr": rec["lr"],
                         "val/mean_uar": val["mean_uar"],
                         **{f"val/uar_{l}": m["uar"] for l, m in val.items() if l != "mean_uar"}}, step=step)
        sums, n, t0 = dict.fromkeys(sums, 0.0), 0, time.time()
        if val["mean_uar"] > best:
            best, best_step, best_state, bad = val["mean_uar"], step, model.trainable_state_dict(), 0
        else:
            bad += 1
        if bad >= hp["patience"]:
            log(f"early stop at step {step} (best step {best_step})")
            break
        if step >= hp["max_steps"]:
            break
    model.load_state_dict(best_state, strict=False)
    return {"best_val_mean_uar": best, "best_step": best_step, "history": history}


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
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override hparams, e.g. max_steps=200")
    ap.add_argument("--tag", default="", help="suffix for the run directory / wandb name (e.g. an lr sweep point)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases (off by default)")
    ap.add_argument("--wandb-project", default="zero-shot-xling-ser")
    ap.add_argument("--wandb-entity", default=None)
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
    hp_sha = hparams_sha256(hp)

    plan = build_plan(protocol, a.task, a.system, a.manifest_dir, backbone=a.backbone)
    run_name = f"{a.system}_seed{hp['seed']}" + (f"_{a.tag}" if a.tag else "")
    out = Path(a.out) / a.task / run_name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    audit = plan.audit()
    log(json.dumps({"task": a.task, "system": a.system, "backbone": plan.backbone, "audit": audit}))

    tracker = None
    if a.wandb:
        import wandb
        tracker = wandb.init(
            project=a.wandb_project, entity=a.wandb_entity, group=a.task, job_type=a.system,
            name=f"{a.task}_{run_name}", dir=str(out),
            tags=[a.task, a.system, "dev" if plan.dev else "zero-shot" if plan.zero_shot else "upper-bound"],
            config={"task": a.task, "system": a.system, "backbone": plan.backbone, "tag": a.tag,
                    "protocol_sha256": plan.protocol_sha256, "hparams_sha256": hp_sha, "audit": audit, **hp})

    model = SERModel(plan.backbone, len(plan.labels), len(plan.speakers) if plan.spkadv else 0,
                     hp["lora_rank"], hp["lora_alpha"], hp["adapter_dim"], hp["spk_hidden"],
                     hp["spk_dropout"], hp["grl_coeff"]).to(a.device)
    train_info = fit(model, plan, hp, a.device, log, tracker)

    # ---- target evaluation: first and only access to target audio ----
    n = len(plan.labels)
    y_true, y_pred = predict(model, list(plan.test), plan, hp, a.device, batch_size=1)
    result = {"target_test": uar_f1(y_true, y_pred, n)}
    with open(out / "target_test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "label", "pred"])
        for r, p in zip(plan.test, y_pred):
            w.writerow([r["utt_id"], r["label"], plan.labels[p]])
    if plan.zero_shot:  # supplementary: whole target corpus (never trained on, so also unseen)
        yt, yp = predict(model, list(plan.target_all), plan, hp, a.device, batch_size=1)
        result["target_all_supplementary"] = uar_f1(yt, yp, n)

    summary = {"task": a.task, "system": a.system, "backbone": plan.backbone, "tag": a.tag, "dev": plan.dev,
               "protocol_sha256": plan.protocol_sha256, "hparams_sha256": hp_sha, "git_commit": git_commit(), "hparams": hp, "audit": audit,
               **train_info, **result}
    (out / "run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if tracker is not None:
        # target numbers go to the run summary only, after training has finished
        tracker.summary.update({"best_val_mean_uar": train_info["best_val_mean_uar"],
                                **{f"{k}/{m}": v[m] for k, v in result.items() for m in ("uar", "f1")}})
        tracker.finish()
    torch.save(model.trainable_state_dict(), out / "trainable_params.pt")
    log(f"[{a.task} {a.system}] target test UAR {result['target_test']['uar']:.2f}  "
        f"F1 {result['target_test']['f1']:.2f}  -> {out / 'run.json'}")


if __name__ == "__main__":
    main()
