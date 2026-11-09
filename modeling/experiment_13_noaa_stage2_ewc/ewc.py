"""Correct, diagonal empirical-Fisher EWC utilities.

The empirical Fisher used here is

    F_j = (1 / N) * sum_i [d log p(y_i | x_i, theta) / d theta_j]^2.

The square is taken *before* averaging across examples. This differs from
squaring a gradient of a mean-reduced batch loss, where gradients from images
can cancel and make the estimate depend on batch size.
"""

import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


def save_parameter_snapshot(model):
    """Return a detached CPU copy of every trainable model parameter."""
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _single_example_negative_log_likelihood(model, parameters, buffers, image, target):
    """Return -log p(target | image, parameters) for exactly one image."""
    logits = functional_call(
        model,
        (parameters, buffers),
        (image.unsqueeze(0),),
    )
    return F.cross_entropy(logits, target.reshape(1), reduction="sum")


def calculate_per_example_empirical_fisher(model, fisher_loader, device):
    """Estimate diagonal Fisher by averaging squared per-example gradients.

    The model must be evaluated after Stage 1 training. ``model.eval()`` keeps
    BatchNorm fixed during this importance measurement. ``fisher_loader`` must
    contain original Stage 1 images, without random augmentation or sampling.
    """
    if len(fisher_loader.dataset) == 0:
        raise ValueError("The Fisher dataset is empty.")

    was_training = model.training
    model.eval()

    # All Stage 1 parameters are trainable in the selected checkpoint. Keeping
    # the mapping explicit makes name/shape mismatches fail immediately later.
    parameters = {
        name: parameter.detach().requires_grad_(True)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    buffers = {
        name: buffer.detach()
        for name, buffer in model.named_buffers()
    }
    if not parameters:
        raise ValueError("No trainable parameters were available for Fisher.")

    fisher_sums = {
        name: torch.zeros_like(parameter, dtype=torch.float64, device=device)
        for name, parameter in parameters.items()
    }

    # ``grad`` obtains one image's gradient. ``vmap`` batches that operation
    # without replacing it with the gradient of an averaged batch loss.
    single_gradient = grad(_single_example_negative_log_likelihood, argnums=1)
    batch_gradients = vmap(single_gradient, in_dims=(None, None, None, 0, 0))

    sample_count = 0
    try:
        for images, targets in fisher_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            gradients = batch_gradients(model, parameters, buffers, images, targets)
            batch_count = images.shape[0]
            for name, per_example_gradient in gradients.items():
                # Sum over this batch only after each individual gradient was
                # squared. Accumulation uses float64 for numerical safety.
                fisher_sums[name].add_(
                    per_example_gradient.detach().to(torch.float64).square().sum(dim=0)
                )
            sample_count += batch_count
    finally:
        model.train(was_training)

    if sample_count != len(fisher_loader.dataset):
        raise RuntimeError(
            "Fisher sample count does not match its dataset: "
            f"{sample_count} != {len(fisher_loader.dataset)}"
        )

    fisher = {
        name: (value / sample_count).to(dtype=torch.float32).cpu()
        for name, value in fisher_sums.items()
    }
    validate_fisher(fisher, expected_names=set(parameters))
    return fisher


def validate_fisher(fisher, expected_names=None):
    """Reject invalid Fisher estimates before they can destabilize EWC."""
    if not fisher:
        raise ValueError("Fisher information is empty.")
    if expected_names is not None and set(fisher) != set(expected_names):
        raise ValueError("Fisher parameter names do not match model parameters.")

    for name, value in fisher.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Fisher contains non-finite values for {name}.")
        if (value < 0).any():
            raise ValueError(f"Fisher contains negative values for {name}.")


def fisher_statistics(fisher):
    """Return auditable global diagnostics for a diagonal Fisher estimate."""
    validate_fisher(fisher)
    flattened = torch.cat([value.reshape(-1).float() for value in fisher.values()])
    return {
        "parameter_tensors": len(fisher),
        "parameter_elements": int(flattened.numel()),
        "sum": float(flattened.sum().item()),
        "mean": float(flattened.mean().item()),
        "max": float(flattened.max().item()),
        "p50": float(torch.quantile(flattened, 0.50).item()),
        "p95": float(torch.quantile(flattened, 0.95).item()),
        "p99": float(torch.quantile(flattened, 0.99).item()),
        "zero_fraction": float((flattened == 0).float().mean().item()),
    }


def move_ewc_anchor_to_device(anchor, device):
    """Move one validated Fisher/parameter anchor from checkpoint storage."""
    required = {"stage", "fisher", "parameters"}
    missing = required.difference(anchor)
    if missing:
        raise ValueError(f"EWC anchor is missing keys: {sorted(missing)}")

    fisher = anchor["fisher"]
    parameters = anchor["parameters"]
    validate_fisher(fisher, expected_names=set(parameters))
    return {
        "stage": int(anchor["stage"]),
        "fisher": {name: value.to(device) for name, value in fisher.items()},
        "parameters": {name: value.to(device) for name, value in parameters.items()},
    }


def ewc_penalty(model, anchors):
    """Return sum_s sum_j F_sj * (theta_j - theta*_sj)^2.

    ``anchors`` are intentionally passed explicitly. At Stage 2 there is one
    Stage 1 anchor only; Stage 3 can choose an accumulated/online strategy
    deliberately rather than silently duplicating penalties.
    """
    parameter_map = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    penalty = next(iter(parameter_map.values())).new_zeros(())

    for anchor in anchors:
        if set(anchor["fisher"]) != set(parameter_map):
            raise ValueError("EWC anchor names do not match current model parameters.")
        for name, parameter in parameter_map.items():
            difference = parameter - anchor["parameters"][name]
            penalty = penalty + (anchor["fisher"][name] * difference.square()).sum()

    return penalty
