"""Correct diagonal empirical-Fisher EWC utilities.

The empirical Fisher is defined as

    F_j = (1 / N) * sum_i [d log p(y_i | x_i, theta) / d theta_j]^2.

Each example's gradient is squared before averaging. This prevents gradients
from different images from cancelling before the Fisher importance is formed.
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


def _single_example_negative_log_likelihood(
    model,
    parameters,
    buffers,
    image,
    target,
):
    """Return -log p(target | image, parameters) for one image."""
    logits = functional_call(
        model,
        (parameters, buffers),
        (image.unsqueeze(0),),
    )
    return F.cross_entropy(logits, target.reshape(1), reduction="sum")


def calculate_per_example_empirical_fisher(model, fisher_loader, device):
    """Estimate diagonal Fisher from squared per-example gradients."""
    if len(fisher_loader.dataset) == 0:
        raise ValueError("The Fisher dataset is empty.")

    was_training = model.training
    model.eval()

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

    single_gradient = grad(_single_example_negative_log_likelihood, argnums=1)
    batch_gradients = vmap(single_gradient, in_dims=(None, None, None, 0, 0))

    sample_count = 0
    try:
        for images, targets in fisher_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            gradients = batch_gradients(model, parameters, buffers, images, targets)
            for name, per_example_gradient in gradients.items():
                fisher_sums[name].add_(
                    per_example_gradient.detach()
                    .to(torch.float64)
                    .square()
                    .sum(dim=0)
                )
            sample_count += images.shape[0]
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
    """Reject invalid Fisher estimates before EWC training begins."""
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
    """Return global diagnostics for an auditable diagonal Fisher."""
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
    """Move one validated Fisher/parameter anchor to the training device."""
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
        "parameters": {
            name: value.to(device) for name, value in parameters.items()
        },
    }


def ewc_penalty(model, anchors):
    """Return sum_s sum_j F_sj * (theta_j - theta*_sj)^2."""
    parameter_map = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not parameter_map:
        raise ValueError("The model contains no trainable parameters for EWC.")

    penalty = next(iter(parameter_map.values())).new_zeros(())
    for anchor in anchors:
        if set(anchor["fisher"]) != set(parameter_map):
            raise ValueError("EWC anchor names do not match current model parameters.")
        if set(anchor["parameters"]) != set(parameter_map):
            raise ValueError(
                "EWC parameter snapshot names do not match current model parameters."
            )

        for name, parameter in parameter_map.items():
            fisher_value = anchor["fisher"][name]
            reference_parameter = anchor["parameters"][name]
            if fisher_value.shape != parameter.shape:
                raise ValueError(f"Fisher shape mismatch for {name}.")
            if reference_parameter.shape != parameter.shape:
                raise ValueError(f"Parameter snapshot shape mismatch for {name}.")

            difference = parameter - reference_parameter
            penalty = penalty + (fisher_value * difference.square()).sum()

    return penalty
