"""Train NOAA Stage 2 with corrected EWC and optional Stage 1 FL-only replay."""
import argparse, csv, math, random, sys, time
from datetime import datetime
from pathlib import Path
import torch, wandb
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from config import (CONFIG, DEFAULT_EWC_ANCHOR, FUTURE_VALIDATION_CSV, IMAGE_DIR, RESULTS_DIR,  # noqa: E402
    STAGE1_FL_REPLAY_CSV, STAGE1_HOLDOUT_CSV, STAGE1_TRAIN_CSV, STAGE2_HOLDOUT_CSV,
    STAGE2_TRAIN_CSV, WANDB_ENTITY, WANDB_PROJECT, WANDB_TAGS)
from dataloader import build_experiment_loaders, cycle_batches
from ewc import ewc_penalty, move_ewc_anchor_to_device
from evaluation import calculate_classification_metrics
from model import build_model  # noqa: E402

GROUPS = ("stage2_train", "stage1_holdout", "stage2_holdout")
CURVES = ("loss", "css", "tss", "hss")


def arguments():
    p, add = argparse.ArgumentParser(description=__doc__), None
    add = p.add_argument
    add("--anchor", type=Path, default=DEFAULT_EWC_ANCHOR)
    add("--replay-csv", type=Path, default=STAGE1_FL_REPLAY_CSV)
    for x in ("ewc-lambda", "replay-loss-weight", "nf-class-weight", "fl-class-weight",
              "backbone-learning-rate", "classifier-lr-multiplier", "weight-decay"):
        add(f"--{x}", type=float)
    for x in ("total-batch-size", "replay-batch-size", "epochs", "num-workers",
              "random-seed", "early-stopping-patience"):
        add(f"--{x}", type=int)
    bn = p.add_mutually_exclusive_group()
    bn.add_argument("--adapt-batchnorm-running-stats", dest="freeze_batchnorm_running_stats", action="store_false")
    bn.add_argument("--freeze-batchnorm-running-stats", dest="freeze_batchnorm_running_stats", action="store_true")
    p.set_defaults(freeze_batchnorm_running_stats=None)
    add("--run-name")
    return p.parse_args()


def configuration(args):
    c = dict(CONFIG)
    keys = ("ewc_lambda", "replay_loss_weight", "nf_class_weight", "fl_class_weight",
            "backbone_learning_rate", "classifier_lr_multiplier", "weight_decay", "epochs",
            "num_workers", "random_seed", "early_stopping_patience", "total_batch_size", "replay_batch_size")
    c.update({k: getattr(args, k) for k in keys if getattr(args, k) is not None})
    for k in ("ewc_lambda", "replay_batch_size", "num_workers", "weight_decay"):
        if c[k] < 0: raise ValueError(f"{k} must be non-negative")
    for k in ("total_batch_size", "replay_loss_weight", "epochs", "early_stopping_patience",
              "nf_class_weight", "fl_class_weight", "backbone_learning_rate", "classifier_lr_multiplier"):
        if c[k] <= 0: raise ValueError(f"{k} must be positive")
    if c["replay_batch_size"] >= c["total_batch_size"]:
        raise ValueError("replay_batch_size must be smaller than total_batch_size")
    c["stage2_batch_size"] = c["total_batch_size"] - c["replay_batch_size"]
    c["classifier_learning_rate"] = c["backbone_learning_rate"] * c["classifier_lr_multiplier"]
    if args.freeze_batchnorm_running_stats is not None:
        c["freeze_batchnorm_running_stats"] = args.freeze_batchnorm_running_stats
    c["batchnorm_running_stats_mode"] = ("frozen_at_stage1_anchor" if c["freeze_batchnorm_running_stats"]
                                          else "adapt_on_mixed_stage2_replay_batches")
    c["replay_enabled"] = c["replay_batch_size"] > 0
    return c


def rates(m):
    nf, fl = m["tn"] + m["fp"], m["tp"] + m["fn"]
    m.update(false_positive_rate=m["fp"] / nf if nf else 0., false_negative_rate=m["fn"] / fl if fl else 0.,
             predicted_fl_rate=(m["tp"] + m["fp"]) / (nf + fl) if nf + fl else 0.)
    return m


def evaluate(model, loader, criterion, device):
    model.eval(); loss, count, predictions, targets = 0., 0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images); n = len(images)
            loss += criterion(logits, labels).item() * n; count += n
            predictions += logits.argmax(1).cpu().tolist(); targets += labels.cpu().tolist()
    if not count: raise ValueError("Cannot evaluate an empty loader")
    m = calculate_classification_metrics(predictions, targets); m["loss"] = loss / count
    return rates(m)


def optimizer_for(model, backbone_lr, classifier_lr, decay):
    buckets = {(part, wd): [] for part in ("backbone", "classifier") for wd in (0, 1)}
    for name, p in model.named_parameters():
        if p.requires_grad:
            part = "classifier" if name.startswith("classifier.") else "backbone"
            buckets[part, int(not (name.endswith(".bias") or p.ndim == 1))].append(p)
    lrs = {"backbone": backbone_lr, "classifier": classifier_lr}
    return AdamW([{"params": buckets[part, wd], "lr": lrs[part], "weight_decay": decay if wd else 0.}
                  for part in ("backbone", "classifier") for wd in (1, 0)])


def ce(logits, targets, weights):
    values = F.cross_entropy(logits, targets, weight=weights, reduction="none")
    num, den = values.sum(), weights[targets].sum()
    return num / den, num, den


def train_epoch(model, loader, replay_loader, criterion, optimizer, device, anchors, c):
    model.train()
    if c["freeze_batchnorm_running_stats"]:
        for module in model.modules():
            if isinstance(module, _BatchNorm): module.eval()
    replay = cycle_batches(replay_loader) if replay_loader is not None else None
    names = ("total", "mixed_num", "mixed_den", "s2_num", "s2_den", "replay_num", "replay_den", "ewc")
    sums, s2_seen, replay_seen, predictions, targets = dict.fromkeys(names, 0.), 0, 0, [], []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        n, rn = len(images), 0
        if replay is not None:
            replay_images, replay_labels = next(replay)
            replay_images, replay_labels = replay_images.to(device, non_blocking=True), replay_labels.to(device, non_blocking=True)
            rn, mixed_images = len(replay_images), torch.cat((images, replay_images))
        else: mixed_images = images
        optimizer.zero_grad(set_to_none=True); logits = model(mixed_images); s2_logits = logits[:n]
        s2_loss, s2_num, s2_den = ce(s2_logits, labels, criterion.weight)
        if rn:
            replay_loss, replay_num, replay_den = ce(logits[n:], replay_labels, criterion.weight)
            mixed_num = s2_num + c["replay_loss_weight"] * replay_num
            mixed_den = s2_den + c["replay_loss_weight"] * replay_den
            class_loss = mixed_num / mixed_den
        else:
            replay_loss = replay_num = replay_den = s2_loss.new_zeros(())
            mixed_num, mixed_den, class_loss = s2_num, s2_den, s2_loss
        raw_ewc = ewc_penalty(model, anchors) if anchors and c["ewc_lambda"] > 0 else class_loss.new_zeros(())
        ewc_loss = .5 * c["ewc_lambda"] * raw_ewc; total = class_loss + ewc_loss
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite loss: stage2={s2_loss.item()}, replay={replay_loss.item()}, mixed={class_loss.item()}, raw_ewc={raw_ewc.item()}, lambda={c['ewc_lambda']}, ewc={ewc_loss.item()}")
        total.backward()
        if c["max_gradient_norm"] is not None:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c["max_gradient_norm"])
            if not torch.isfinite(norm): raise FloatingPointError("Non-finite gradient norm")
        optimizer.step()
        values = (total.item() * n, mixed_num.item(), mixed_den.item(), s2_num.item(), s2_den.item(),
                  replay_num.item(), replay_den.item(), ewc_loss.item() * n)
        sums = {k: sums[k] + v for k, v in zip(names, values)}
        s2_seen += n; replay_seen += rn
        predictions += s2_logits.argmax(1).detach().cpu().tolist(); targets += labels.cpu().tolist()
    if s2_seen != len(loader.dataset): raise RuntimeError("Stage 2 sample count changed during epoch")
    m = calculate_classification_metrics(predictions, targets)
    m.update(loss=sums["total"] / s2_seen, classification_loss=sums["mixed_num"] / sums["mixed_den"],
             stage2_classification_loss=sums["s2_num"] / sums["s2_den"],
             replay_classification_loss=sums["replay_num"] / sums["replay_den"] if replay_seen else 0.,
             ewc_loss=sums["ewc"] / s2_seen, stage2_images_seen=s2_seen, replay_images_seen=replay_seen)
    return rates(m)


def joint_css(a, b): return math.sqrt(max(0., a["css"]) * max(0., b["css"]))
def prefix(name, m): return {f"{name}/{k}": v for k, v in m.items()}


def write_csv(path, data, wide=False):
    with path.open("w", newline="", encoding="utf-8") as f:
        if wide:
            writer = csv.DictWriter(f, fieldnames=data[0]); writer.writeheader(); writer.writerows(data)
        else:
            writer = csv.writer(f); writer.writerow(["metric", "value"]); writer.writerows(data.items())


def save(path, model, optimizer, scheduler, epoch, c, metrics, anchor, sampling):
    torch.save({"completed_stage": 2, "stage2_adapted": True, "architecture": "mobilenet_v3_large",
        "experiment": "stage2_ewc_stage1_fl_replay", "epoch": epoch,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "selection_metrics": metrics, "config": c, "sampling_metadata": sampling,
        "ewc_history": anchor["ewc_history"], "stage1_fisher_metadata": anchor["fisher_metadata"]}, path)


def main():
    args = arguments(); c = configuration(args)
    if not args.anchor.is_file(): raise FileNotFoundError(f"EWC anchor not found: {args.anchor}")
    if c["replay_enabled"] and not args.replay_csv.is_file(): raise FileNotFoundError(f"Replay CSV not found: {args.replay_csv}")
    random.seed(c["random_seed"]); torch.manual_seed(c["random_seed"]); torch.cuda.manual_seed_all(c["random_seed"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.cuda.set_device(0)
    anchor = torch.load(args.anchor, map_location="cpu", weights_only=False)
    missing = {"model_state_dict", "ewc_history", "fisher_metadata"} - anchor.keys()
    if missing: raise ValueError(f"Invalid EWC anchor; missing {sorted(missing)}")
    if len(anchor["ewc_history"]) != 1 or anchor["ewc_history"][0].get("stage") != 1:
        raise ValueError("Stage 2 requires exactly one Stage 1 EWC anchor")
    anchors = [move_ewc_anchor_to_device(anchor["ewc_history"][0], device)]
    model = build_model(pretrained=False, hidden_dim=c["classifier_hidden_dim"], dropout=c["dropout"]).to(device)
    model.load_state_dict(anchor["model_state_dict"], strict=True)
    paths = {"image_directory": IMAGE_DIR, "stage1_train_csv": STAGE1_TRAIN_CSV,
             "stage1_holdout_csv": STAGE1_HOLDOUT_CSV, "stage2_train_csv": STAGE2_TRAIN_CSV,
             "stage2_holdout_csv": STAGE2_HOLDOUT_CSV, "future_validation_csv": FUTURE_VALIDATION_CSV,
             "replay_csv": args.replay_csv}
    loaders, sampling = build_experiment_loaders(c, paths)
    weights = torch.tensor([c["nf_class_weight"], c["fl_class_weight"]], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optimizer_for(model, c["backbone_learning_rate"], c["classifier_learning_rate"], c["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=c["scheduler_factor"], patience=c["scheduler_patience"],
        threshold=c["early_stopping_min_delta"], threshold_mode="abs",
        min_lr=[c["min_backbone_learning_rate"]] * 2 + [c["min_classifier_learning_rate"]] * 2)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S"); out = RESULTS_DIR / f"mobilenet_stage2_ewc_fl_replay_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    replay_tags = ["replay", "stage1-fl-only-replay"] if c["replay_enabled"] else ["no-replay-control"]
    bn_tag = "batchnorm-stage1-frozen" if c["freeze_batchnorm_running_stats"] else "batchnorm-mixed-batch-adaptation"
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
        name=args.run_name or f"noaa_stage2_ewc_fl_replay_{stamp}", tags=WANDB_TAGS + replay_tags + [bn_tag],
        job_type="stage2-ewc-replay-training", dir=str(out), config={**c,
        "source_ewc_anchor": str(args.anchor.resolve()),
        "source_replay_csv": str(args.replay_csv.resolve()) if c["replay_enabled"] else None,
        "source_stage1_selected_epoch": anchor["fisher_metadata"]["source_selected_epoch"],
        "stage1_fisher_statistics": anchor["fisher_metadata"]["statistics"], "sampling_metadata": sampling,
        "device": str(device), "gpu_count_used": int(device.type == "cuda"), "multi_gpu_method": "none"})
    c = dict(run.config); base1 = evaluate(model, loaders["stage1_holdout"], criterion, device)
    base2 = evaluate(model, loaders["stage2_holdout"], criterion, device); base_joint = joint_css(base1, base2)
    print(f"Device: {device} | GPUs={int(device.type == 'cuda')}\nResults: {out}\nEWC anchor: {args.anchor}")
    print(f"EWC lambda: {c['ewc_lambda']:.6g}\nReplay enabled: {c['replay_enabled']}")
    if c["replay_enabled"]: print(f"Replay CSV: {args.replay_csv}")
    print(f"Mixed batch: Stage 2={c['stage2_batch_size']} | replay={c['replay_batch_size']} | maximum total={c['total_batch_size']}")
    print(f"Replay loss weight: {c['replay_loss_weight']:.4f}\nBatchNorm running statistics: {c['batchnorm_running_stats_mode']}")
    print(f"Sampling metadata: {sampling}\nStage 1 anchor holdout CSS: {base1['css']:.4f}")
    print(f"Before Stage 2 training | Stage 2 holdout CSS: {base2['css']:.4f}\nEpoch 0 joint CSS: {base_joint:.4f}")
    print("Epoch 0 is reference-only; checkpoint selection starts at epoch 1.")
    run.define_metric("epoch")
    for group in GROUPS: run.define_metric(f"{group}/*", step_metric="epoch")
    run.define_metric("best/*"); run.define_metric("future_validation/*")
    initial = {"epoch": 0, "selection/stage2_css": base2["css"], "selection/joint_css": base_joint,
        "selection/stage1_css_retention_ratio": 1., **prefix("stage1_holdout", base1), **prefix("stage2_holdout", base2),
        **prefix("stage1_anchor", base1), **prefix("stage2_before_training", base2)}
    run.log(initial); initial_path = out / "initial_metrics.csv"
    write_csv(initial_path, {k: v for k, v in initial.items() if k != "epoch"})
    best, best_recall, best_joint, best_epoch, stale = float("-inf"), float("-inf"), float("nan"), None, 0
    checkpoint_path, rows = out / "stage2_best.pt", []
    history = {"epoch": [], **{f"{g}/{m}": [] for g in GROUPS for m in CURVES}}
    for epoch in range(1, int(c["epochs"]) + 1):
        start = time.perf_counter(); train = train_epoch(model, loaders["stage2_train"], loaders["stage1_replay"],
            criterion, optimizer, device, anchors, c)
        stage1 = evaluate(model, loaders["stage1_holdout"], criterion, device)
        stage2 = evaluate(model, loaders["stage2_holdout"], criterion, device)
        score, gap, diagnostic = stage2["css"], train["css"] - stage2["css"], joint_css(stage1, stage2)
        retention = stage1["css"] / base1["css"] if base1["css"] > 0 else 0.; scheduler.step(score); old_best = best
        if score > best:
            best, best_recall, best_joint, best_epoch = score, stage2["recall"], diagnostic, epoch
            selected = {"selection_metric": "stage2_holdout_css", "stage2_holdout_css": score,
                "stage2_holdout_recall": stage2["recall"], "stage2_train_css": train["css"],
                "stage2_train_holdout_css_gap": gap, "joint_css": diagnostic,
                "stage1_css_retention_ratio": retention, "stage1_holdout": stage1, "stage2_holdout": stage2}
            save(checkpoint_path, model, optimizer, scheduler, epoch, c, selected, anchor, sampling)
        stale = 0 if score > old_best + c["early_stopping_min_delta"] else stale + 1
        elapsed = time.perf_counter() - start
        logged = {"epoch": epoch, "selection/stage2_css": score, "selection/joint_css": diagnostic,
            "diagnostics/stage2_train_holdout_css_gap": gap, "selection/stage1_css_retention_ratio": retention,
            "learning_rate/backbone": optimizer.param_groups[0]["lr"], "learning_rate/classifier": optimizer.param_groups[2]["lr"],
            "best/stage2_css": best, "best/stage2_recall": best_recall, "best/joint_css_at_selected_epoch": best_joint,
            "best/epoch": best_epoch, "epochs_without_improvement": stale, "timing/epoch_seconds": elapsed,
            **prefix("stage2_train", train), **prefix("stage1_holdout", stage1), **prefix("stage2_holdout", stage2)}
        run.log(logged); rows.append({k.replace("/", "_"): v for k, v in logged.items()})
        write_csv(out / "epoch_metrics.csv", rows, True); history["epoch"].append(epoch)
        for group, metrics in (("stage2_train", train), ("stage1_holdout", stage1), ("stage2_holdout", stage2)):
            for metric in CURVES: history[f"{group}/{metric}"].append(metrics[metric])
        print(f"Epoch {epoch:02d}/{c['epochs']}\n  Stage 2 train | loss={train['loss']:.4f} | mixed class={train['classification_loss']:.4f} | Stage 2 class={train['stage2_classification_loss']:.4f} | replay class={train['replay_classification_loss']:.4f} | EWC={train['ewc_loss']:.4f} | CSS={train['css']:.4f}")
        print(f"  Images seen | Stage 2={train['stage2_images_seen']:,} | replay={train['replay_images_seen']:,}")
        print(f"  Stage 1 retention | CSS={stage1['css']:.4f} | recall={stage1['recall']:.4f} | FPR={stage1['false_positive_rate']:.4f}")
        print(f"  Stage 2 holdout | CSS={stage2['css']:.4f} | recall={stage2['recall']:.4f} | FPR={stage2['false_positive_rate']:.4f}")
        print(f"  Progress | Stage 2 CSS={score:.4f} | joint CSS={diagnostic:.4f} | train-holdout gap={gap:.4f} | best epoch={best_epoch} | best Stage 2 CSS={best:.4f} | time={elapsed:.1f}s")
        if stale >= c["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}."); break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    stage1, stage2 = evaluate(model, loaders["stage1_holdout"], criterion, device), evaluate(model, loaders["stage2_holdout"], criterion, device)
    future = evaluate(model, loaders["future_validation"], criterion, device); selected = checkpoint["selection_metrics"]
    final = {"selected_epoch": checkpoint["epoch"], "selected_stage2_css": selected["stage2_holdout_css"],
        "selected_stage2_recall": selected["stage2_holdout_recall"], "selected_stage2_train_css": selected["stage2_train_css"],
        "selected_stage2_train_holdout_css_gap": selected["stage2_train_holdout_css_gap"], "selected_joint_css": selected["joint_css"],
        "stage2_css_delta_from_epoch0": stage2["css"] - base2["css"], "stage1_css_retention_ratio": stage1["css"] / base1["css"] if base1["css"] > 0 else 0.,
        "replay/enabled": c["replay_enabled"], "replay/batch_size": c["replay_batch_size"], "replay/loss_weight": c["replay_loss_weight"],
        **prefix("stage1_retention", stage1), **prefix("stage2_holdout", stage2), **prefix("future_validation", future)}
    run.log(final)
    for k, v in final.items(): run.summary[k] = v
    for metric in CURVES:
        run.log({f"learning_curves/{metric}": wandb.plot.line_series(xs=history["epoch"],
            ys=[history[f"{g}/{metric}"] for g in GROUPS], keys=["Stage 2 training", "Stage 1 retention", "Stage 2 holdout"],
            title=f"Stage 2 EWC + FL replay: {metric.upper()}", xname="Epoch")})
    final_path = out / "final_metrics.csv"; write_csv(final_path, final)
    artifact = wandb.Artifact(f"noaa-stage2-ewc-fl-replay-{run.id}", type="model", metadata={
        "ewc_lambda": c["ewc_lambda"], "replay_enabled": c["replay_enabled"], "replay_batch_size": c["replay_batch_size"],
        "selected_epoch": checkpoint["epoch"], "selection": c["checkpoint_selection"]})
    for path in (checkpoint_path, initial_path, out / "epoch_metrics.csv", final_path): artifact.add_file(str(path))
    if c["replay_enabled"]: artifact.add_file(str(args.replay_csv))
    run.log_artifact(artifact)
    print(f"Best epoch: {checkpoint['epoch']}\nSelection: maximum Stage 2 holdout CSS among adapted epochs.")
    print(f"Selected Stage 1 retention | CSS={stage1['css']:.4f} | recall={stage1['recall']:.4f} | FPR={stage1['false_positive_rate']:.4f}")
    print(f"Selected Stage 2 holdout | CSS={stage2['css']:.4f} | recall={stage2['recall']:.4f} | FPR={stage2['false_positive_rate']:.4f}")
    print(f"Future validation (not used for selection) | CSS={future['css']:.4f} | recall={future['recall']:.4f} | FPR={future['false_positive_rate']:.4f}\nCheckpoint: {checkpoint_path}")
    run.finish()


if __name__ == "__main__": main()
