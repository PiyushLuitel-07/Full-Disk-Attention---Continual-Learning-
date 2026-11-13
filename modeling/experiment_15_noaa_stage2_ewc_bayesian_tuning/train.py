"""Run one compact Stage 2 EWC-only Bayesian-sweep trial."""
import argparse, csv, math, random, sys, time
from datetime import datetime
from pathlib import Path
import torch, wandb
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP13 = ROOT / "modeling/experiment_13_noaa_stage2_ewc"
sys.path[:0] = [str(HERE), str(EXP13), str(ROOT)]
from config import (ANCHOR, CONFIG, IMAGE_DIR, RESULTS_DIR, STAGE1_HOLDOUT_CSV, STAGE1_TRAIN_CSV,  # noqa: E402
    STAGE2_HOLDOUT_CSV, STAGE2_TRAIN_CSV, VALIDATION_CSV, WANDB_ENTITY, WANDB_PROJECT, WANDB_TAGS)
from dataloader import build_stage2_loaders, class_counts  # noqa: E402
from ewc import ewc_penalty, move_ewc_anchor_to_device  # noqa: E402
from evaluation import calculate_classification_metrics  # noqa: E402
from model import build_model  # noqa: E402


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor", type=Path, default=ANCHOR); p.add_argument("--run-name")
    return p.parse_args()


def rates(m):
    nf, fl = m["tn"] + m["fp"], m["tp"] + m["fn"]
    m.update(false_positive_rate=m["fp"] / nf if nf else 0., false_negative_rate=m["fn"] / fl if fl else 0.,
             predicted_fl_rate=(m["tp"] + m["fp"]) / (nf + fl) if nf + fl else 0.)
    return m


def evaluate(model, loader, loss_fn, device):
    model.eval(); loss, count, pred, truth = 0., 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            z = model(x); loss += loss_fn(z, y).item() * len(x); count += len(x)
            pred += z.argmax(1).cpu().tolist(); truth += y.cpu().tolist()
    m = calculate_classification_metrics(pred, truth); m["loss"] = loss / count
    return rates(m)


def optimizer_for(model, c):
    groups = {(part, decay): [] for part in ("backbone", "classifier") for decay in (0, 1)}
    for name, p in model.named_parameters():
        part = "classifier" if name.startswith("classifier.") else "backbone"
        groups[part, int(not (name.endswith(".bias") or p.ndim == 1))].append(p)
    lr = {"backbone": c["backbone_learning_rate"], "classifier": c["classifier_learning_rate"]}
    return AdamW([{"params": groups[part, decay], "lr": lr[part],
                   "weight_decay": c["weight_decay"] if decay else 0.}
                  for part in ("backbone", "classifier") for decay in (1, 0)])


def train_epoch(model, loader, loss_fn, optimizer, device, anchors, c):
    model.train()
    if not c["adapt_batchnorm_running_stats"]:
        for module in model.modules():
            if isinstance(module, _BatchNorm): module.eval()
    sums = dict(total=0., classification=0., ewc=0.); count = 0; pred, truth = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True); z = model(x); classification = loss_fn(z, y)
        penalty = ewc_penalty(model, anchors); regularization = .5 * c["ewc_lambda"] * penalty
        total = classification + regularization
        if not torch.isfinite(total): raise FloatingPointError("Non-finite Stage 2 loss")
        total.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c["max_gradient_norm"])
        if not torch.isfinite(norm): raise FloatingPointError("Non-finite gradient norm")
        optimizer.step(); n = len(x); count += n
        for key, value in (("total", total), ("classification", classification), ("ewc", regularization)):
            sums[key] += value.detach().item() * n
        pred += z.argmax(1).detach().cpu().tolist(); truth += y.cpu().tolist()
    m = calculate_classification_metrics(pred, truth)
    m.update(loss=sums["total"] / count, classification_loss=sums["classification"] / count,
             ewc_loss=sums["ewc"] / count)
    return rates(m)


def prefixed(name, metrics): return {f"{name}/{k}": v for k, v in metrics.items()}
def joint(a, b): return math.sqrt(max(0., a["css"]) * max(0., b["css"]))


def write_csv(path, data, wide=False):
    with path.open("w", newline="", encoding="utf-8") as f:
        if wide:
            w = csv.DictWriter(f, fieldnames=data[0]); w.writeheader(); w.writerows(data)
        else:
            w = csv.writer(f); w.writerow(["metric", "value"]); w.writerows(data.items())


def main():
    a = args(); stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"mobilenet_stage2_ewc_tuning_{stamp}"; out.mkdir(parents=True, exist_ok=False)
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, config=CONFIG, dir=str(out),
        name=a.run_name or f"noaa_stage2_experiment_15_ewc_{stamp}", tags=WANDB_TAGS,
        job_type="stage2-ewc-bayesian-sweep-trial")
    c = dict(run.config)
    c["classifier_learning_rate"] = float(c["backbone_learning_rate"]) * float(c["classifier_lr_multiplier"])
    c["freeze_batchnorm_running_stats"] = not bool(c["adapt_batchnorm_running_stats"])
    c["batchnorm_running_stats_mode"] = "adapt_on_stage2" if c["adapt_batchnorm_running_stats"] else "frozen_at_stage1"
    run.config.update({"classifier_learning_rate": c["classifier_learning_rate"],
                       "freeze_batchnorm_running_stats": c["freeze_batchnorm_running_stats"],
                       "batchnorm_running_stats_mode": c["batchnorm_running_stats_mode"]}, allow_val_change=True)
    if c["ewc_lambda"] <= 0: raise ValueError("Experiment 15 requires ewc_lambda > 0")
    if not a.anchor.is_file(): raise FileNotFoundError(f"Missing EWC anchor: {a.anchor}")
    random.seed(c["random_seed"]); torch.manual_seed(c["random_seed"]); torch.cuda.manual_seed_all(c["random_seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.cuda.set_device(0)
    anchor = torch.load(a.anchor, map_location="cpu", weights_only=False)
    if len(anchor["ewc_history"]) != 1: raise ValueError("Expected exactly one Stage 1 EWC anchor")
    model = build_model(pretrained=False, hidden_dim=c["classifier_hidden_dim"], dropout=0.).to(device)
    model.load_state_dict(anchor["model_state_dict"], strict=True)
    anchors = [move_ewc_anchor_to_device(anchor["ewc_history"][0], device)]
    paths = {"image_directory": IMAGE_DIR, "stage1_train_csv": STAGE1_TRAIN_CSV,
        "stage1_holdout_csv": STAGE1_HOLDOUT_CSV, "stage2_train_csv": STAGE2_TRAIN_CSV,
        "stage2_holdout_csv": STAGE2_HOLDOUT_CSV, "future_validation_csv": VALIDATION_CSV}
    loaders, sampling = build_stage2_loaders(c, paths)
    counts = class_counts(loaders["stage2_train"].dataset)
    weights = torch.tensor([c["nf_class_weight"], c["fl_class_weight"]], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(c["label_smoothing"]))
    optimizer = optimizer_for(model, c)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=c["scheduler_factor"],
        patience=c["scheduler_patience"], threshold=c["early_stopping_min_delta"], threshold_mode="abs",
        min_lr=[c["min_backbone_learning_rate"]] * 2 + [c["min_classifier_learning_rate"]] * 2)
    run.config.update({"source_ewc_anchor": str(a.anchor.resolve()), "stage2_train_nf": counts[0],
        "stage2_train_fl": counts[1], "sampling_metadata": sampling, "device": str(device)}, allow_val_change=True)
    base1, base2 = evaluate(model, loaders["stage1_holdout"], loss_fn, device), evaluate(model, loaders["stage2_holdout"], loss_fn, device)
    print(f"Results: {out}\nDevice: {device}\nStage 2 data: NF={counts[0]:,}, FL={counts[1]:,}")
    print(f"lambda={c['ewc_lambda']:.3g} | backbone LR={c['backbone_learning_rate']:.2e} | classifier LR={c['classifier_learning_rate']:.2e}")
    print(f"weight decay={c['weight_decay']:.4f} | smoothing={c['label_smoothing']:.3f} | BN={c['batchnorm_running_stats_mode']}")
    print(f"Epoch 0 | Stage 1 CSS={base1['css']:.4f} | Stage 2 CSS={base2['css']:.4f}")
    run.define_metric("epoch")
    for group in ("stage2_train", "stage1_holdout", "stage2_holdout"): run.define_metric(f"{group}/*", step_metric="epoch")
    initial = {"epoch": 0, **prefixed("stage1_holdout", base1), **prefixed("stage2_holdout", base2)}
    run.log(initial); write_csv(out / "initial_metrics.csv", {k: v for k, v in initial.items() if k != "epoch"})
    best, best_epoch, stale, rows = float("-inf"), None, 0, []; checkpoint = out / "stage2_best.pt"
    for epoch in range(1, int(c["epochs"]) + 1):
        started = time.perf_counter(); train = train_epoch(model, loaders["stage2_train"], loss_fn, optimizer, device, anchors, c)
        stage1, stage2 = evaluate(model, loaders["stage1_holdout"], loss_fn, device), evaluate(model, loaders["stage2_holdout"], loss_fn, device)
        score, gap, old = stage2["css"], train["css"] - stage2["css"], best; scheduler.step(score)
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"epoch": epoch, "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": c, "stage2_holdout": stage2, "stage1_retention": stage1,
                "stage2_train": train, "ewc_history": anchor["ewc_history"],
                "stage1_fisher_metadata": anchor["fisher_metadata"]}, checkpoint)
        stale = 0 if score > old + c["early_stopping_min_delta"] else stale + 1
        logged = {"epoch": epoch, "selection/stage2_css": score, "diagnostics/train_holdout_css_gap": gap,
            "diagnostics/joint_css": joint(stage1, stage2), "diagnostics/stage1_retention_ratio": stage1["css"] / base1["css"],
            "best/epoch": best_epoch, "best/stage2_css": best, "epochs_without_improvement": stale,
            "timing/epoch_seconds": time.perf_counter() - started, **prefixed("stage2_train", train),
            **prefixed("stage1_holdout", stage1), **prefixed("stage2_holdout", stage2)}
        run.log(logged); rows.append({k.replace("/", "_"): v for k, v in logged.items()}); write_csv(out / "epoch_metrics.csv", rows, True)
        print(f"Epoch {epoch:02d} | train CSS={train['css']:.4f} | holdout CSS={score:.4f} | recall={stage2['recall']:.4f} | FPR={stage2['false_positive_rate']:.4f} | gap={gap:.4f} | best={best:.4f}")
        if stale >= c["early_stopping_patience"]: break
    selected = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(selected["model_state_dict"])
    stage1, stage2 = evaluate(model, loaders["stage1_holdout"], loss_fn, device), evaluate(model, loaders["stage2_holdout"], loss_fn, device)
    validation = evaluate(model, loaders["future_validation"], loss_fn, device)
    final = {"selected_epoch": selected["epoch"], "selected_stage2_train_css": selected["stage2_train"]["css"],
        "selected_stage2_train_holdout_gap": selected["stage2_train"]["css"] - stage2["css"],
        "stage2_css_delta_from_epoch0": stage2["css"] - base2["css"], "stage1_css_retention_ratio": stage1["css"] / base1["css"],
        **prefixed("stage1_retention", stage1), **prefixed("stage2_holdout", stage2), **prefixed("validation", validation)}
    run.log(final)
    for k, v in final.items(): run.summary[k] = v
    write_csv(out / "final_metrics.csv", final)
    artifact = wandb.Artifact(f"noaa-stage2-ewc-tuning-{run.id}", type="model", metadata={"selected_epoch": selected["epoch"]})
    for path in (checkpoint, out / "initial_metrics.csv", out / "epoch_metrics.csv", out / "final_metrics.csv"): artifact.add_file(str(path))
    run.log_artifact(artifact)
    print(f"Selected epoch={selected['epoch']} | Stage 2 CSS={stage2['css']:.4f} | Stage 1 CSS={stage1['css']:.4f} | validation CSS={validation['css']:.4f}")
    run.finish()


if __name__ == "__main__": main()
