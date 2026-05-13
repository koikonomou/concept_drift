"""
Scientific questions:
  Pool A: Does fine-tuning on DATA DRIFT change concept drift? 
  Pool B: Does fine-tuning on CONCEPT DRIFT improve margins and accuracy?

POOL A
  Train  : Pure Data Drift, ranked by data_drift_score descending
  Eval   : ALL Concept Drifted (Pure Concept Drift + Full Drift = 672 images)
           Always held out — Pool A never trains on concept drift samples.

POOL B
  Train  : Concept Drifted (Pure Concept + Full Drift), ranked by has_margin ascending
  Eval   : Held-out Concept Drifted samples not used for fine-tuning
           At 100%, no held-out exists → reported as diagnostic

Both pools also report:
  - Source landscape test accuracy (catastrophic forgetting check)
  - Stable custom accuracy (In-Distribution + Pure Data Drift)
  - Full custom margin before/after (HAS objective recovery)

"""

import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import (
    TEST_ROOT, WEIGHT_DIR, RESULT_DIR, DEVICE,
    HAS_MARGIN, HAS_SCALE, BATCH_SIZE, ALPHA,
    LANDSCAPE_CLASSES, ensure_dirs,
)
from models import HASModel, FolderDataset, STANDARD_TRANSFORM, TRAIN_AUGMENT


RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class PathLabelDataset(Dataset):
    def __init__(self, records, transform):
        self.records   = list(records)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r   = self.records[idx]
        img = Image.open(r["file_path"]).convert("RGB")
        return self.transform(img), int(r["label"])


# ─────────────────────────────────────────────────────────────────────────────
# CSV loading and filtering
# ─────────────────────────────────────────────────────────────────────────────

def _cls_to_idx():
    return {name: i for i, name in enumerate(LANDSCAPE_CLASSES)}


def load_taxonomy(csv_path):
    """Load drift_has.csv into a list of record dicts.

    Each record has: file_path, label, true_class, drift_type,
    data_drift_score, has_margin, concept_severity, data_severity.
    Only records with existing file_path are included.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing: {csv_path} — run detect.py first.")
    df      = pd.read_csv(csv_path)
    required = ["file_path", "true_class", "has_drift_type_margin",
                "data_drift_score", "has_margin"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}\n"
                         f"  Re-run detect.py with the updated version.")
    mapper  = _cls_to_idx()
    records = []
    for _, r in df.iterrows():
        fp  = str(r["file_path"])
        cls = str(r["true_class"])
        if not os.path.exists(fp) or cls not in mapper:
            continue
        m  = float(r["has_margin"])
        dd = float(r["data_drift_score"])
        records.append({
            "file_path":         fp,
            "label":             int(mapper[cls]),
            "true_class":        cls,
            "drift_type":        str(r["has_drift_type_margin"]),
            "data_drift_score":  dd,
            "has_margin":        m,
            "concept_severity":  1.0 - m,
            "data_severity":     dd,
        })
    return records


def filter_records(records, drift_types):
    dt = set(drift_types)
    return [r for r in records if r["drift_type"] in dt]


def sort_pool_a(records):
    """Most severe data drift first."""
    return sorted(records, key=lambda r: r["data_drift_score"], reverse=True)


def sort_pool_b(records):
    """Most boundary-confused first (smallest margin = closest to boundary)."""
    return sorted(records, key=lambda r: r["has_margin"])


def split_pct(records, pct):
    """Split records into (selected top pct%, held-out rest)."""
    if not records:
        return [], []
    n = max(1, int(round(len(records) * pct)))
    return records[:n], records[n:]


def record_stats(records, prefix):
    """Summary statistics for a set of records."""
    if not records:
        return {f"{prefix}_n": 0,
                f"{prefix}_data_drift_score_mean": None,
                f"{prefix}_has_margin_mean": None,
                f"{prefix}_concept_severity_mean": None}
    data   = np.array([r["data_drift_score"] for r in records])
    margin = np.array([r["has_margin"]        for r in records])
    csev   = np.array([r["concept_severity"]  for r in records])
    return {
        f"{prefix}_n":                      len(records),
        f"{prefix}_data_drift_score_mean":  float(data.mean()),
        f"{prefix}_data_drift_score_min":   float(data.min()),
        f"{prefix}_data_drift_score_max":   float(data.max()),
        f"{prefix}_has_margin_mean":        float(margin.mean()),
        f"{prefix}_has_margin_min":         float(margin.min()),
        f"{prefix}_has_margin_max":         float(margin.max()),
        f"{prefix}_concept_severity_mean":  float(csev.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(weight_path, dropout=0.1):
    """Load HAS model. Accepts plain state_dict or wrapped checkpoints."""
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weights not found: {weight_path}")
    model = HASModel(n_classes=len(LANDSCAPE_CLASSES),
                     margin=HAS_MARGIN, scale=HAS_SCALE,
                     dropout=dropout).to(DEVICE)
    state = torch.load(weight_path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_records(model, records, batch_size):
    """Accuracy and error rate on a list of records."""
    if not records:
        return {"n": 0, "acc": None, "err": None}
    model.eval()
    ds     = PathLabelDataset(records, STANDARD_TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    total = correct = 0
    for imgs, labels in loader:
        imgs    = imgs.to(DEVICE)
        labels  = labels.long().to(DEVICE)
        logits, _, _ = model(imgs)
        correct += logits.argmax(1).eq(labels).sum().item()
        total   += len(labels)
    acc = 100.0 * correct / max(total, 1)
    return {"n": total, "acc": acc, "err": 100.0 - acc}


@torch.no_grad()
def margin_records(model, records, batch_size):
    """Mean angular margin (cos_best - cos_second) on a list of records."""
    if not records:
        return {"n": 0, "mean": None, "std": None}
    model.eval()
    ds     = PathLabelDataset(records, STANDARD_TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    nw      = model.get_normed_weights()
    margins = []
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        _, _, latent = model(imgs)
        cos_t = latent @ nw
        cos_sorted, _ = cos_t.sort(dim=1, descending=True)
        margins.extend(
            (cos_sorted[:, 0] - cos_sorted[:, 1]).detach().cpu().numpy().tolist())
    return {"n": len(margins),
            "mean": float(np.mean(margins)),
            "std":  float(np.std(margins))}


@torch.no_grad()
def evaluate_source(model, test_root, batch_size):
    """Accuracy + margin on the original source test set."""
    if not test_root or not os.path.isdir(test_root):
        return {"n": 0, "acc": None, "err": None, "margin": None}
    ds = FolderDataset(test_root, class_names=LANDSCAPE_CLASSES,
                       transform=STANDARD_TRANSFORM)
    if len(ds) == 0:
        return {"n": 0, "acc": None, "err": None, "margin": None}
    loader  = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    model.eval()
    nw      = model.get_normed_weights()
    total   = correct = 0
    margins = []
    for imgs, labels, _ in loader:
        imgs   = imgs.to(DEVICE)
        labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
        labels = labels.long().to(DEVICE)
        logits, _, latent = model(imgs)
        correct += logits.argmax(1).eq(labels).sum().item()
        total   += len(labels)
        cos_t    = latent @ nw
        cos_sorted, _ = cos_t.sort(dim=1, descending=True)
        margins.extend(
            (cos_sorted[:, 0] - cos_sorted[:, 1]).detach().cpu().numpy().tolist())
    acc = 100.0 * correct / max(total, 1)
    return {"n": total, "acc": acc, "err": 100.0 - acc,
            "margin": float(np.mean(margins)) if margins else None}


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def set_trainable(model, scope):
    """Freeze everything, then unfreeze the specified scope.

    Scopes:
      head              has_layer only                  (320 params)
      embedder-fc-head  embedder.fc + has_layer         (~2000 params)

    Returns (modules_list, trainable_params_list).
    """
    for p in model.parameters():
        p.requires_grad = False

    if scope == "head":
        modules = [("has_layer", model.has_layer)]
    elif scope == "embedder-fc-head":
        modules = [
            ("embedder.fc", model.embedder.fc),
            ("has_layer",   model.has_layer),
        ]
    else:
        raise ValueError(f"Unknown ft_scope: {scope}")

    params = []
    for _, m in modules:
        for p in m.parameters():
            p.requires_grad = True
            params.append(p)

    return modules, params


def finetune_model(model, train_records, args):
    """Fine-tune the model on train_records.

    Critical design:
      model.eval() sets ALL BatchNorm to eval mode (uses running statistics).
      Then only the trainable modules are set to .train().
      This prevents small-batch BatchNorm statistics from corrupting embeddings —
      the root cause of Loss≈3.0 and Train-Acc≈35% we saw earlier.

    Loss = ALPHA * NLL + ft_beta * HAS_penalty
      ft_beta=2.0 enforces margin constraints more strongly during fine-tuning.
      This is the HAS-native signal: push embeddings away from boundaries.

    Optimizer: AdamW (more stable than SGD for small data fine-tuning).
    Gradient clipping: prevents instability with small batches.
    """
    if not train_records:
        print("  No training records — skipping.")
        return model

    transform = TRAIN_AUGMENT if args.augment else STANDARD_TRANSFORM
    ds        = PathLabelDataset(train_records, transform)
    loader    = DataLoader(ds,
                           batch_size=min(args.batch_size, len(ds)),
                           shuffle=True, num_workers=4, drop_last=False)

    modules, params = set_trainable(model, args.ft_scope)
    print(f"  Scope      : {args.ft_scope}")
    print(f"  Trainable  : {sum(p.numel() for p in params):,} params "
          f"({'+'.join(n for n, _ in modules)})")
    print(f"  Loss       : {ALPHA} * NLL + {args.ft_beta} * HAS_penalty")
    print(f"  Optimizer  : AdamW(lr={args.ft_lr}, wd={args.weight_decay})")
    print(f"  BatchNorm  : eval mode (running statistics, not batch stats)")

    opt = torch.optim.AdamW(params, lr=args.ft_lr, weight_decay=args.weight_decay)
    nll = nn.NLLLoss()

    # ── KEY: model.eval() keeps BatchNorm in eval mode ───────────────────────
    # Then .train() only on the modules being optimised.
    # BatchNorm layers are never put in train mode → always use running stats.
    model.eval()
    for _, m in modules:
        m.train()

    for ep in range(1, args.ft_epochs + 1):
        total = correct = 0
        loss_sum = nll_sum = pen_sum = 0.0

        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.long().to(DEVICE)

            opt.zero_grad(set_to_none=True)

            # Full HAS forward — labels provided so penalty is computed
            logits, penalty, _ = model(imgs, labels=labels)
            loss_nll = nll(logits, labels)
            loss     = ALPHA * loss_nll + args.ft_beta * penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()

            correct  += logits.argmax(1).eq(labels).sum().item()
            total    += len(labels)
            loss_sum += loss.item()
            nll_sum  += loss_nll.item()
            pen_sum  += penalty.item()

        # Log at first epoch, last epoch, and every log_every epochs
        if (ep == 1 or ep == args.ft_epochs
                or (args.log_every > 0 and ep % args.log_every == 0)):
            with torch.no_grad():
                nw     = model.get_normed_weights()
                cos_ww = (nw.T @ nw).detach().cpu()
                mask   = ~torch.eye(cos_ww.shape[0], dtype=torch.bool)
                w_cos  = cos_ww[mask].mean().item()
            n = max(len(loader), 1)
            print(f"    EP {ep:3d}/{args.ft_epochs} | "
                  f"Loss {loss_sum/n:.4f} | "
                  f"NLL {nll_sum/n:.4f} | "
                  f"Pen {pen_sum/n:.4f} | "
                  f"Train-Acc {100.0*correct/max(total,1):.1f}% | "
                  f"w_cos={w_cos:.3f}")

    # Restore all gradients and set model back to eval
    for p in model.parameters():
        p.requires_grad = True
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CSV row helpers
# ─────────────────────────────────────────────────────────────────────────────

def add_acc(row, prefix, before, after):
    row[f"{prefix}_n"] = before.get("n", after.get("n"))
    for k in ["acc", "err"]:
        b = before.get(k)
        a = after.get(k)
        row[f"{prefix}_{k}_before"] = b
        row[f"{prefix}_{k}_after"]  = a
        row[f"{prefix}_{k}_delta"]  = (a - b
                                        if a is not None and b is not None
                                        else None)


def add_margin(row, prefix, before, after):
    row[f"{prefix}_margin_n"]          = before.get("n", after.get("n"))
    row[f"{prefix}_margin_before"]     = before.get("mean")
    row[f"{prefix}_margin_after"]      = after.get("mean")
    row[f"{prefix}_margin_delta"]      = (
        after.get("mean") - before.get("mean")
        if after.get("mean") is not None and before.get("mean") is not None
        else None)
    row[f"{prefix}_margin_std_before"] = before.get("std")
    row[f"{prefix}_margin_std_after"]  = after.get("std")


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment run
# ─────────────────────────────────────────────────────────────────────────────
def fmt_pct(x):
    return "—" if x is None else f"{x:.2f}%"

def fmt_num(x):
    return "—" if x is None else f"{x:.4f}"

def run_condition(pool_name, pct, fixed_n,  train_pool, concept_all, stable,
                  full_custom, args, has_weights, ckpt_dir):
    """Run one (pool, ft_pct) combination.

    Pool A: train on Pure Data Drift → evaluate on ALL concept drifted
            (orthogonality test: should show near-zero concept change)
i
    Pool B: train on Concept Drifted → evaluate on held-out concept drifted
            (recovery test: margin should increase, errors should decrease)
    """
    
    #selected, heldout = split_pct(train_pool, pct)
    selected = train_pool[:fixed_n]
    
    if pool_name == "A":
        concept_eval    = concept_all
        eval_protocol   = "heldout_cross_pool"
        train_subset    = "Pure Data Drift"
        ranked_by       = "data_drift_score descending"
    else:
        heldout = train_pool[fixed_n:]
        concept_eval    = heldout if len(heldout) > 0 else selected
        eval_protocol   = ("heldout_concept" if len(heldout) > 0
                           else "concept_train_diagnostic_no_heldout")
        train_subset    = "Pure Concept Drift + Full Drift (both)"
        ranked_by       = "has_margin ascending"

    print("\n" + "─" * 72)
    print(f"POOL {pool_name} | ft={pct*100:.0f}% | "
          f"train={len(selected)} | concept_eval={len(concept_eval)} | "
          f"{eval_protocol}")
    print("─" * 72)

    # Fresh model for every run
    model = load_model(has_weights, args.dropout)

    # ── Evaluate BEFORE ───────────────────────────────────────────────────────
    before_concept        = evaluate_records(model, concept_eval, args.batch_size)
    before_concept_margin = margin_records(model, concept_eval, args.batch_size)
    before_stable         = evaluate_records(model, stable, args.batch_size)
    before_stable_margin  = margin_records(model, stable, args.batch_size)
    before_full_margin    = margin_records(model, full_custom, args.batch_size)
    before_source         = evaluate_source(model, args.test_root, args.batch_size)
    
    print(f"  BEFORE concept eval: acc={fmt_pct(before_concept['acc'])}  "
            f"err={fmt_pct(before_concept['err'])}  "
            f"margin_μ={fmt_num(before_concept_margin['mean'])}")
    print(f"  BEFORE stable      : acc={fmt_pct(before_stable['acc'])}%  "
          f"margin_μ={fmt_num(before_stable_margin['mean'])}")
    print(f"  BEFORE source test : acc={fmt_pct(before_source['acc'])}%")
    print(f"  BEFORE full margin : μ={fmt_num(before_full_margin['mean'])}")

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    finetune_model(model, selected, args)

    # ── Evaluate AFTER ────────────────────────────────────────────────────────
    after_concept        = evaluate_records(model, concept_eval, args.batch_size)
    after_concept_margin = margin_records(model, concept_eval, args.batch_size)
    after_stable         = evaluate_records(model, stable, args.batch_size)
    after_stable_margin  = margin_records(model, stable, args.batch_size)
    after_full_margin    = margin_records(model, full_custom, args.batch_size)
    after_source         = evaluate_source(model, args.test_root, args.batch_size)

    # Print Δ values
    def delta_str(b, a, key):
        bv, av = b.get(key), a.get(key)
        if bv is None or av is None: return "—"
        d = av - bv
        return f"{d:+.2f}" if key in ("acc","err") else f"{d:+.4f}"

    print(f"  AFTER  concept eval: acc={fmt_pct(after_concept['acc'])}%  "
          f"err={fmt_pct(after_concept['err'])}%  "
          f"margin_μ={fmt_num(after_concept_margin['mean'])}  "
          f"[Δacc={delta_str(before_concept,after_concept,'acc')}  "
          f"Δmargin={delta_str(before_concept_margin,after_concept_margin,'mean')}]")
    print(f"  AFTER  stable      : acc={fmt_pct(after_stable['acc'])}%  "
          f"[Δacc={delta_str(before_stable,after_stable,'acc')}]")
    print(f"  AFTER  source test : acc={fmt_pct(after_source['acc'])}%  "
          f"[Δacc={delta_str(before_source,after_source,'acc')}]")
    print(f"  AFTER  full margin : μ={fmt_num(after_full_margin['mean'])}  "
          f"[Δ={delta_str(before_full_margin,after_full_margin,'mean')}]")

    # Save checkpoint
    ckpt_path = ckpt_dir / (f"has_pool_{pool_name.lower()}_"
                            f"{int(pct*100)}pct_{args.ft_scope}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  ✓ {ckpt_path}")

    # ── Build CSV row ─────────────────────────────────────────────────────────
    row = {
        "pool":                  pool_name,
        "ft_pct":                pct,
        "ft_scope":              args.ft_scope,
        "n_finetune":            len(selected),
        "n_pool_total":          len(train_pool),
        "train_subset":          train_subset,
        "ranked_by":             ranked_by,
        "concept_eval_protocol": eval_protocol,
        "n_concept_eval":        len(concept_eval),
        "checkpoint":            str(ckpt_path),
    }
    row.update(record_stats(selected,     "train"))
    row.update(record_stats(concept_eval, "concept_eval"))

    add_acc(row,    "concept_custom",  before_concept,        after_concept)
    add_margin(row, "concept_custom",  before_concept_margin, after_concept_margin)
    add_acc(row,    "stable_custom",   before_stable,         after_stable)
    add_margin(row, "stable_custom",   before_stable_margin,  after_stable_margin)
    add_acc(row,    "source_test",     before_source,         after_source)
    row["source_test_margin_before"] = before_source.get("margin")
    row["source_test_margin_after"]  = after_source.get("margin")
    row["source_test_margin_delta"]  = (
        after_source.get("margin") - before_source.get("margin")
        if after_source.get("margin") is not None
        and before_source.get("margin") is not None else None)
    add_margin(row, "full_custom",     before_full_margin,    after_full_margin)

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Two-pool HAS fine-tuning experiment.")

    # Paths
    parser.add_argument("--weight-dir",  default=None, help="Override weights directory")
    parser.add_argument("--result-dir",  default=None, help="Override results directory")
    parser.add_argument("--output-dir",  default=None, help="Output directory for CSV and checkpoints")
    parser.add_argument("--input-csv",   default=None, help="Path to drift_has.csv (default: results/drift_has.csv)")
    parser.add_argument("--has-weights", default=None, help="Path to has_model.pth (auto-detected if not given)")
    parser.add_argument("--test-root",   default=TEST_ROOT, help="Source test set root for catastrophic forgetting check")

    # Experiment
    parser.add_argument("--pool", choices=["data", "concept", "both"], default="both")
    parser.add_argument("--ft-pcts", default="25,50,100", help="Comma-separated %% of each pool to use")

    # Fine-tuning
    parser.add_argument("--ft-scope",choices=["head", "embedder-fc-head"], default="embedder-fc-head",
                        help="Which parameters to train: "
                             "head=has_layer only, "
                             "embedder-fc-head=embedder.fc+has_layer")
    parser.add_argument("--ft-lr",       type=float, default=5e-5)
    parser.add_argument("--ft-beta",     type=float, default=2.0, help="HAS penalty weight during fine-tuning (default 2.0)")
    parser.add_argument("--ft-epochs",   type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--weight-decay",type=float, default=1e-4)
    parser.add_argument("--grad-clip",   type=float, default=5.0)
    parser.add_argument("--augment",     action="store_true", help="Use training augmentation (default: eval transform)")
    parser.add_argument("--log-every",   type=int,   default=10, help="Print training log every N epochs")
    parser.add_argument("--seed",        type=int,   default=RANDOM_SEED)

    args = parser.parse_args()
    ensure_dirs()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    result_dir = Path(args.result_dir) if args.result_dir else Path(RESULT_DIR)
    weight_dir = Path(args.weight_dir) if args.weight_dir else Path(WEIGHT_DIR)
    output_dir = Path(args.output_dir) if args.output_dir else Path(RESULT_DIR) / "finetune"
    ckpt_dir   = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    input_csv  = Path(args.input_csv) if args.input_csv else result_dir / "drift_has.csv"

    has_weights = args.has_weights
    if has_weights is None:
        for cand in [weight_dir / "has_best.pth", weight_dir / "has_model.pth"]:
            if cand.exists():
                has_weights = str(cand)
                break
    if has_weights is None:
        raise FileNotFoundError("No HAS weights found. Run train.py first.")

    # ── Load taxonomy ─────────────────────────────────────────────────────────
    all_records = load_taxonomy(str(input_csv))
    
    pool_a_full      = sort_pool_a(filter_records(all_records, ["Pure Data Drift"]))
    pool_b_full      = sort_pool_b(filter_records(all_records,
                                              ["Pure Concept Drift", "Full Drift (both)"]))
    pool_r_full = list(all_records) # "All Samples"
    pool_a = sort_pool_a(pool_a_full)
    pool_b = sort_pool_b(pool_b_full)
    pool_r = list(pool_r_full)
    import random
    random.seed(args.seed)
    random.shuffle(pool_r)
    concept_all = sort_pool_b(filter_records(all_records,
                                              ["Pure Concept Drift", "Full Drift (both)"]))
    stable      = filter_records(all_records, ["In-Distribution", "Pure Data Drift"])
    full_custom = list(all_records)

    pcts = [float(x) / 100.0 if float(x) > 1 else float(x) for x in args.ft_pcts.split(",")]
    
    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 72)
    print("TWO-POOL FINE-TUNING EXPERIMENT")
    print("=" * 72)
    print(f"  input_csv   : {input_csv}")
    print(f"  weights     : {has_weights}")
    print(f"  output_dir  : {output_dir}")
    print(f"  Pool A      : {len(pool_a)} Pure Data Drift images")
    print(f"  Pool B      : {len(pool_b)} Concept Drifted images "
          f"({len(filter_records(all_records,['Pure Concept Drift']))} Pure + "
          f"{len(filter_records(all_records,['Full Drift (both)']))} Full)")
    print(f"  Concept eval: {len(concept_all)} images (Pool A cross-pool eval)")
    print(f"  Stable eval : {len(stable)} images (In-Dist + Pure Data Drift)")
    print(f"  Full custom : {len(full_custom)} images (margin measurement)")
    print(f"  ft_pcts     : {[f'{p*100:.0f}%' for p in pcts]}")
    print(f"  ft_scope    : {args.ft_scope}")
    print(f"  ft_epochs   : {args.ft_epochs}  ft_lr: {args.ft_lr}  "
          f"ft_beta: {args.ft_beta}")

    # ── Run experiments ───────────────────────────────────────────────────────
    rows = []

    if args.pool in {"data", "both"}:
        print("\n" + "═" * 72)
        print("POOL A — Train Pure Data Drift → Evaluate Concept Drift")
        print("Expected: near-zero concept Δ (orthogonality claim)")
        print("═" * 72)
        for pct in pcts:
            rows.append(run_condition(
                pool_name="A", pct=pct, fixed_n = ref_n,
                train_pool=pool_a, concept_all=concept_all,
                stable=stable, full_custom=full_custom,
                args=args, has_weights=has_weights, ckpt_dir=ckpt_dir))

    if args.pool in {"concept", "both"}:
        print("\n" + "═" * 72)
        print("POOL B — Train Concept Drift → Margin Recovery + Eval Held-out Concept")
        print("Expected: margin increases, concept errors decrease")
        print("═" * 72)
        for pct in pcts:
            rows.append(run_condition(
                pool_name="B", pct=pct, fixed_n = ref_n,
                train_pool=pool_b, concept_all=concept_all,
                stable=stable, full_custom=full_custom,
                args=args, has_weights=has_weights, ckpt_dir=ckpt_dir))

    if args.pool in {"all", "both"}:
        print("\n" + "═" * 72)
        print("POOL ALL")
        print("═" * 72)
        for pct in pcts:
            rows.append(run_condition(
                pool_name="R", pct=pct, fixed_n = ref_n,
                train_pool=pool_b, concept_all=concept_all,
                stable=stable, full_custom=full_custom,
                args=args, has_weights=has_weights, ckpt_dir=ckpt_dir))

    # ── Save outputs ──────────────────────────────────────────────────────────
    summary = pd.DataFrame(rows)
    summary_path = output_dir / "final_finetune_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n  ✓ {summary_path}")

    # Long-format CSV for easy plotting
    long_rows = []
    for _, r in summary.iterrows():
        for group in ["concept_custom", "stable_custom", "source_test", "full_custom"]:
            for metric in ["acc", "err", "margin"]:
                b_key = f"{group}_{metric}_before"
                a_key = f"{group}_{metric}_after"
                d_key = f"{group}_{metric}_delta"
                if b_key in summary.columns or a_key in summary.columns:
                    long_rows.append({
                        "pool":        r["pool"],
                        "ft_pct":      r["ft_pct"],
                        "ft_scope":    r["ft_scope"],
                        "n_finetune":  r["n_finetune"],
                        "train_subset":r["train_subset"],
                        "ranked_by":   r["ranked_by"],
                        "eval_protocol":r["concept_eval_protocol"],
                        "metric_group":group,
                        "metric":      metric,
                        "before":      r.get(b_key),
                        "after":       r.get(a_key),
                        "delta":       r.get(d_key),
                    })
    long_path = output_dir / "final_finetune_summary_long.csv"
    pd.DataFrame(long_rows).to_csv(long_path, index=False)
    print(f"  ✓ {long_path}")

    # Config JSON
    config = vars(args).copy()
    config.update({"input_csv": str(input_csv), "has_weights": str(has_weights),
                   "output_dir": str(output_dir)})
    (output_dir / "finetune_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True))
    print(f"  ✓ {output_dir / 'finetune_config.json'}")

    # Preview table
    preview_cols = [
        "pool", "ft_pct", "n_finetune", "train_subset", "concept_eval_protocol",
        "concept_custom_acc_before",  "concept_custom_acc_after",  "concept_custom_acc_delta",
        "concept_custom_margin_before","concept_custom_margin_after","concept_custom_margin_delta",
        "stable_custom_acc_before",   "stable_custom_acc_after",   "stable_custom_acc_delta",
        "source_test_acc_before",     "source_test_acc_after",     "source_test_acc_delta",
        "full_custom_margin_before",  "full_custom_margin_after",  "full_custom_margin_delta",
    ]
    preview_cols = [c for c in preview_cols if c in summary.columns]
    print("\nPREVIEW:")
    print(summary[preview_cols].to_string(index=False))

    print("\nfinetune.py complete.")


if __name__ == "__main__":
    main()
