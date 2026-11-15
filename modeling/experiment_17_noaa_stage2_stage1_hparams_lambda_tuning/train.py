"""Train Stage 2 with the selected Stage 1 settings; vary only EWC lambda."""
import argparse, csv, json, math, random, sys, time
from datetime import datetime
from pathlib import Path
import torch, wandb
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from config import (ANCHOR, CONFIG, IMAGE_DIR, RESULTS_DIR, STAGE1_HOLDOUT_CSV, STAGE1_KEYS,  # noqa: E402
    STAGE1_TRAIN_CSV, STAGE2_HOLDOUT_CSV, STAGE2_TRAIN_CSV, VALIDATION_CSV,
    WANDB_ENTITY, WANDB_PROJECT, WANDB_TAGS)
from dataloader import build_loaders, class_counts  # noqa: E402
from ewc import ewc_penalty, move_ewc_anchor_to_device  # noqa: E402
from metrics import classification_metrics  # noqa: E402
from model import build_model  # noqa: E402


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor", type=Path, default=ANCHOR)
    p.add_argument("--ewc-lambda", type=float)
    p.add_argument("--run-name")
    return p.parse_args()


def evaluate(model, loader, loss_fn, device):
    model.eval(); loss, count, pred, truth = 0., 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            z = model(x); loss += loss_fn(z, y).item() * len(x); count += len(x)
            pred += z.argmax(1).cpu().tolist(); truth += y.cpu().tolist()
    m = classification_metrics(pred, truth); m["loss"] = loss / count
    return m


def optimizer_for(model, c):
    groups = {(part, decay): [] for part in ("backbone", "classifier") for decay in (0, 1)}
    for name, p in model.named_parameters():
        part = "classifier" if name.startswith("classifier.") else "backbone"
        groups[part, int(not (name.endswith(".bias") or p.ndim == 1))].append(p)
    empty = [name for name, values in groups.items() if not values]
    if empty: raise ValueError(f"Empty optimizer parameter groups: {empty}")
    lr = {"backbone": c["backbone_learning_rate"], "classifier": c["classifier_learning_rate"]}
    return AdamW([{"params": groups[part, decay], "lr": lr[part],
                   "weight_decay": c["weight_decay"] if decay else 0.}
                  for part in ("backbone", "classifier") for decay in (1, 0)])


def train_epoch(model, loader, loss_fn, optimizer, device, anchors, c):
    model.train(); sums = dict(total=0., classification=0., ewc=0., raw_ewc=0., gradient_norm=0.)
    count = batches = 0; maximum_gradient_norm = 0.; pred, truth = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True); z = model(x); classification = loss_fn(z, y)
        raw_penalty = ewc_penalty(model, anchors); regularization = .5 * c["ewc_lambda"] * raw_penalty
        total = classification + regularization
        if not torch.isfinite(total): raise FloatingPointError("Non-finite Stage 2 loss")
        total.backward()
        gradient_norms = [parameter.grad.detach().norm(2) for parameter in model.parameters()
                          if parameter.grad is not None]
        norm = torch.stack(gradient_norms).norm(2) if gradient_norms else total.new_zeros(())
        if not torch.isfinite(norm): raise FloatingPointError("Non-finite gradient norm")
        optimizer.step(); n = len(x); count += n; batches += 1
        maximum_gradient_norm = max(maximum_gradient_norm, float(norm))
        for key, value in (("total", total), ("classification", classification), ("ewc", regularization),
                           ("raw_ewc", raw_penalty)):
            sums[key] += value.detach().item() * n
        sums["gradient_norm"] += float(norm)
        pred += z.argmax(1).detach().cpu().tolist(); truth += y.cpu().tolist()
    m = classification_metrics(pred, truth)
    m.update(loss=sums["total"] / count, classification_loss=sums["classification"] / count,
             ewc_loss=sums["ewc"] / count, raw_ewc_penalty=sums["raw_ewc"] / count,
             mean_gradient_norm=sums["gradient_norm"] / batches, max_gradient_norm=maximum_gradient_norm,
             optimizer_steps=batches, examples=count)
    return m


def prefixed(name, values): return {f"{name}/{k}": v for k, v in values.items()}
def joint(a, b): return math.sqrt(max(0., a["css"]) * max(0., b["css"]))


def write_csv(path, data, wide=False):
    with path.open("w", newline="", encoding="utf-8") as f:
        if wide:
            w = csv.DictWriter(f, fieldnames=data[0]); w.writeheader(); w.writerows(data)
        else:
            w = csv.writer(f); w.writerow(["metric", "value"]); w.writerows(data.items())


def log_learning_curves(run, history):
    """Create explicit combined plots in every W&B trial."""
    x = history["epoch"]
    charts = {
        "css": ([history["stage2_train/css"], history["stage2_holdout/css"], history["stage1_holdout/css"]],
                ["Stage 2 training", "Stage 2 holdout", "Stage 1 retention"]),
        "tss": ([history["stage2_train/tss"], history["stage2_holdout/tss"], history["stage1_holdout/tss"]],
                ["Stage 2 training", "Stage 2 holdout", "Stage 1 retention"]),
        "hss": ([history["stage2_train/hss"], history["stage2_holdout/hss"], history["stage1_holdout/hss"]],
                ["Stage 2 training", "Stage 2 holdout", "Stage 1 retention"]),
        "loss": ([history["stage2_train/loss"], history["stage2_train/classification_loss"],
                  history["stage2_train/ewc_loss"], history["stage2_holdout/loss"], history["stage1_holdout/loss"]],
                 ["Stage 2 total", "Stage 2 classification", "EWC contribution", "Stage 2 holdout", "Stage 1 holdout"]),
        "recall": ([history["stage2_train/recall"], history["stage2_holdout/recall"], history["stage1_holdout/recall"]],
                   ["Stage 2 training", "Stage 2 holdout", "Stage 1 retention"]),
        "precision": ([history["stage2_train/precision"], history["stage2_holdout/precision"],
                       history["stage1_holdout/precision"]],
                      ["Stage 2 training", "Stage 2 holdout", "Stage 1 retention"]),
        "f1": ([history["stage2_train/f1"], history["stage2_holdout/f1"], history["stage1_holdout/f1"]],
               ["Stage 2 training", "Stage 2 holdout", "Stage 1 retention"]),
        "accuracy": ([history["stage2_train/accuracy"], history["stage2_holdout/accuracy"],
                      history["stage1_holdout/accuracy"]],
                     ["Stage 2 training", "Stage 2 holdout", "Stage 1 retention"]),
        "false_positive_rate": ([history["stage2_train/false_positive_rate"],
                                 history["stage2_holdout/false_positive_rate"],
                                 history["stage1_holdout/false_positive_rate"]],
                                ["Stage 2 training", "Stage 2 holdout", "Stage 1 retention"]),
        "ewc": ([history["stage2_train/raw_ewc_penalty"], history["stage2_train/ewc_loss"]],
                ["Raw EWC penalty", "Weighted EWC loss contribution"]),
        "learning_rate": ([history["learning_rate/backbone"], history["learning_rate/classifier"]],
                          ["Backbone", "Classifier"]),
        "adaptation_diagnostics": ([history["diagnostics/train_holdout_css_gap"],
                                    history["diagnostics/stage1_retention_ratio"]],
                                   ["Stage 2 train-holdout CSS gap", "Stage 1 CSS retention ratio"]),
    }
    for metric, (values, names) in charts.items():
        run.log({f"learning_curves/{metric}": wandb.plot.line_series(
            xs=x, ys=values, keys=names, title=f"Training, holdout and retention {metric.replace('_', ' ').upper()}",
            xname="Epoch")})


def resolve_config(run, anchor, cli_lambda):
    c, source = dict(CONFIG), anchor.get("source_stage1_config")
    if not isinstance(source, dict): raise ValueError("EWC anchor lacks source_stage1_config")
    missing = [key for key in STAGE1_KEYS if key not in source]
    if missing: raise ValueError(f"Stage 1 configuration is missing: {missing}")
    mismatched = {key: {"documented": CONFIG[key], "anchor": source[key]}
                  for key in STAGE1_KEYS if CONFIG[key] != source[key]}
    if mismatched: raise ValueError(f"Documented Stage 1 settings disagree with the EWC anchor: {mismatched}")
    c.update({key: source[key] for key in STAGE1_KEYS})
    c["ewc_lambda"] = float(cli_lambda if cli_lambda is not None else run.config["ewc_lambda"])
    c["classifier_learning_rate"] = float(c["backbone_learning_rate"]) * float(c["classifier_lr_multiplier"])
    if c["ewc_lambda"] < 0: raise ValueError("ewc_lambda must be non-negative")
    run.config.update(c, allow_val_change=True)
    return c


def main():
    a = arguments(); stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"mobilenet_stage2_lambda_{stamp}"; out.mkdir(parents=True, exist_ok=False)
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, config={"ewc_lambda": CONFIG["ewc_lambda"]},
        dir=str(out), name=a.run_name or f"noaa_stage2_experiment_17_lambda_{stamp}", tags=WANDB_TAGS,
        job_type="stage2-ewc-lambda-ablation")
    if not a.anchor.is_file(): raise FileNotFoundError(f"Missing EWC anchor: {a.anchor}")
    anchor = torch.load(a.anchor, map_location="cpu", weights_only=False)
    if anchor.get("completed_stage") != 1 or anchor.get("architecture") != "mobilenet_v3_large":
        raise ValueError("Expected a completed Stage 1 MobileNetV3-Large EWC anchor")
    c = resolve_config(run, anchor, a.ewc_lambda)
    with (out / "resolved_config.json").open("w", encoding="utf-8") as f: json.dump(c, f, indent=2, sort_keys=True)
    random.seed(c["random_seed"]); torch.manual_seed(c["random_seed"]); torch.cuda.manual_seed_all(c["random_seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.cuda.set_device(0)
    if len(anchor.get("ewc_history", [])) != 1: raise ValueError("Expected exactly one Stage 1 EWC anchor")
    model = build_model(pretrained=False, hidden_dim=c["classifier_hidden_dim"], dropout=c["dropout"]).to(device)
    model.load_state_dict(anchor["model_state_dict"], strict=True)
    anchors = [move_ewc_anchor_to_device(anchor["ewc_history"][0], device)]
    with torch.no_grad(): initial_penalty = ewc_penalty(model, anchors).item()
    if anchors[0]["stage"] != 1 or not math.isfinite(initial_penalty) or abs(initial_penalty) > 1e-10:
        raise ValueError(f"Invalid Stage 1 EWC anchor; initial penalty={initial_penalty}")
    paths = {"image_directory": IMAGE_DIR, "stage1_train_csv": STAGE1_TRAIN_CSV,
        "stage1_holdout_csv": STAGE1_HOLDOUT_CSV, "stage2_train_csv": STAGE2_TRAIN_CSV,
        "stage2_holdout_csv": STAGE2_HOLDOUT_CSV, "future_validation_csv": VALIDATION_CSV}
    loaders, sampling = build_loaders(c, paths); counts = class_counts(loaders["stage2_train"].dataset)
    weights = torch.tensor([c["nf_class_weight"], c["fl_class_weight"]], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=c["label_smoothing"])
    optimizer = optimizer_for(model, c)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=c["scheduler_factor"],
        patience=c["scheduler_patience"], threshold=c["early_stopping_min_delta"], threshold_mode="abs",
        min_lr=[c["min_backbone_learning_rate"]] * 2 + [c["min_classifier_learning_rate"]] * 2)
    run.config.update({"source_ewc_anchor": str(a.anchor.resolve()), "sampling_metadata": sampling,
        "stage2_train_nf": counts[0], "stage2_train_fl": counts[1], "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "pytorch_version": str(torch.__version__), "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "stage1_holdout_samples": len(loaders["stage1_holdout"].dataset),
        "stage2_holdout_samples": len(loaders["stage2_holdout"].dataset),
        "validation_samples": len(loaders["future_validation"].dataset),
        "initial_ewc_penalty": initial_penalty, "only_tuned_parameter": "ewc_lambda"}, allow_val_change=True)
    base1 = evaluate(model, loaders["stage1_holdout"], loss_fn, device)
    base2 = evaluate(model, loaders["stage2_holdout"], loss_fn, device)
    initial = {"epoch": 0, **prefixed("stage1_holdout", base1), **prefixed("stage2_holdout", base2)}
    run.define_metric("epoch")
    for group in ("stage2_train", "stage1_holdout", "stage2_holdout", "selection", "diagnostics",
                  "learning_rate", "best", "timing"):
        run.define_metric(f"{group}/*", step_metric="epoch")
    run.log(initial); write_csv(out / "initial_metrics.csv", {k: v for k, v in initial.items() if k != "epoch"})
    print(f"Results: {out}\nDevice: {device} | lambda={c['ewc_lambda']:g} | only lambda varies")
    print(f"Stage 1 settings | LR={c['backbone_learning_rate']:.3g} x {c['classifier_lr_multiplier']:g} | "
          f"FL weight={c['fl_class_weight']:.4f} | decay={c['weight_decay']:.4f} | batch={c['batch_size']}")
    print(f"Epoch 0 | Stage 1 CSS={base1['css']:.4f} | Stage 2 CSS={base2['css']:.4f}")
    curve_metrics = ("css", "tss", "hss", "loss", "accuracy", "precision", "recall", "f1",
                     "false_positive_rate")
    history = {"epoch": []}
    for group in ("stage2_train", "stage1_holdout", "stage2_holdout"):
        for metric in curve_metrics: history[f"{group}/{metric}"] = []
    history["stage2_train/classification_loss"], history["stage2_train/ewc_loss"] = [], []
    history["stage2_train/raw_ewc_penalty"] = []
    history["learning_rate/backbone"], history["learning_rate/classifier"] = [], []
    history["diagnostics/train_holdout_css_gap"], history["diagnostics/stage1_retention_ratio"] = [], []
    best, best_epoch, stale, rows = float("-inf"), None, 0, []; checkpoint = out / "stage2_best.pt"
    for epoch in range(1, int(c["epochs"]) + 1):
        used_backbone_lr, used_classifier_lr = optimizer.param_groups[0]["lr"], optimizer.param_groups[2]["lr"]
        started = time.perf_counter(); train = train_epoch(model, loaders["stage2_train"], loss_fn, optimizer, device, anchors, c)
        stage1 = evaluate(model, loaders["stage1_holdout"], loss_fn, device)
        stage2 = evaluate(model, loaders["stage2_holdout"], loss_fn, device)
        score, old = stage2["css"], best; scheduler.step(score)
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"epoch": epoch, "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": c, "stage2_train": train, "stage2_holdout": stage2, "stage1_retention": stage1,
                "ewc_history": anchor["ewc_history"], "stage1_fisher_metadata": anchor["fisher_metadata"]}, checkpoint)
        stale = 0 if score > old + c["early_stopping_min_delta"] else stale + 1
        epoch_seconds = time.perf_counter() - started
        logged = {"epoch": epoch, "selection/stage2_css": score,
            "diagnostics/train_holdout_css_gap": train["css"] - score,
            "diagnostics/joint_css": joint(stage1, stage2),
            "diagnostics/stage1_retention_ratio": stage1["css"] / base1["css"],
            "diagnostics/stage1_css_change": stage1["css"] - base1["css"],
            "diagnostics/stage2_css_change": stage2["css"] - base2["css"],
            "diagnostics/ewc_to_classification_loss_ratio": (
                train["ewc_loss"] / train["classification_loss"] if train["classification_loss"] else 0.),
            "best/epoch": best_epoch, "best/stage2_css": best, "epochs_without_improvement": stale,
            "learning_rate/backbone": used_backbone_lr,
            "learning_rate/classifier": used_classifier_lr,
            "timing/epoch_seconds": epoch_seconds,
            "timing/examples_per_second": train["examples"] / epoch_seconds,
            **prefixed("stage2_train", train),
            **prefixed("stage1_holdout", stage1), **prefixed("stage2_holdout", stage2)}
        run.log(logged); rows.append({k.replace("/", "_"): v for k, v in logged.items()})
        history["epoch"].append(epoch)
        for key in history:
            if key != "epoch": history[key].append(logged[key])
        write_csv(out / "epoch_metrics.csv", rows, True)
        print(f"Epoch {epoch:02d} | loss={train['loss']:.4f} (class={train['classification_loss']:.4f}, "
              f"EWC={train['ewc_loss']:.4f}) | train CSS={train['css']:.4f} | Stage2 CSS={score:.4f} "
              f"TSS={stage2['tss']:.4f} HSS={stage2['hss']:.4f} recall={stage2['recall']:.4f} "
              f"FPR={stage2['false_positive_rate']:.4f} | Stage1 CSS={stage1['css']:.4f}")
        if stale >= c["early_stopping_patience"]: break
    log_learning_curves(run, history)
    run.log({"tables/epoch_metrics": wandb.Table(columns=list(rows[0]),
        data=[[row[column] for column in rows[0]] for row in rows])})
    selected = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(selected["model_state_dict"])
    stage1 = evaluate(model, loaders["stage1_holdout"], loss_fn, device)
    stage2 = evaluate(model, loaders["stage2_holdout"], loss_fn, device)
    validation = evaluate(model, loaders["future_validation"], loss_fn, device)
    final = {"ewc_lambda": c["ewc_lambda"], "selected_epoch": selected["epoch"],
        "selected_stage2_train_css": selected["stage2_train"]["css"],
        "selected_stage2_train_holdout_gap": selected["stage2_train"]["css"] - stage2["css"],
        "stage2_css_delta_from_epoch0": stage2["css"] - base2["css"],
        "stage1_css_retention_ratio": stage1["css"] / base1["css"],
        **prefixed("stage1_retention", stage1), **prefixed("stage2_holdout", stage2),
        **prefixed("validation", validation)}
    run.log(final)
    for key, value in final.items(): run.summary[key] = value
    write_csv(out / "final_metrics.csv", final)
    artifact = wandb.Artifact(f"noaa-stage2-lambda-{run.id}", type="model",
        metadata={"ewc_lambda": c["ewc_lambda"], "selected_epoch": selected["epoch"]})
    for path in (checkpoint, out / "resolved_config.json", out / "initial_metrics.csv",
                 out / "epoch_metrics.csv", out / "final_metrics.csv"): artifact.add_file(str(path))
    run.log_artifact(artifact)
    print(f"Selected epoch={selected['epoch']} | Stage2={stage2['css']:.4f} | "
          f"Stage1={stage1['css']:.4f} | validation={validation['css']:.4f}")
    run.finish()


if __name__ == "__main__": main()
