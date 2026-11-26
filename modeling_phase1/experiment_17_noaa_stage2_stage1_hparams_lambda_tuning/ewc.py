"""Self-contained validated EWC penalty utilities for Experiment 17."""
import torch


def validate_fisher(fisher, expected_names=None):
    if not fisher: raise ValueError("Fisher information is empty")
    if expected_names is not None and set(fisher) != set(expected_names):
        raise ValueError("Fisher parameter names do not match model parameters")
    for name, value in fisher.items():
        if not torch.isfinite(value).all(): raise ValueError(f"Non-finite Fisher values for {name}")
        if (value < 0).any(): raise ValueError(f"Negative Fisher values for {name}")


def move_ewc_anchor_to_device(anchor, device):
    missing = {"stage", "fisher", "parameters"}.difference(anchor)
    if missing: raise ValueError(f"EWC anchor is missing keys: {sorted(missing)}")
    validate_fisher(anchor["fisher"], expected_names=set(anchor["parameters"]))
    for name, fisher_value in anchor["fisher"].items():
        parameter_value = anchor["parameters"][name]
        if fisher_value.shape != parameter_value.shape:
            raise ValueError(f"Fisher/snapshot shape mismatch for {name}")
        if not torch.isfinite(parameter_value).all():
            raise ValueError(f"Non-finite Stage 1 parameter snapshot for {name}")
    return {"stage": int(anchor["stage"]),
        "fisher": {name: value.to(device) for name, value in anchor["fisher"].items()},
        "parameters": {name: value.to(device) for name, value in anchor["parameters"].items()}}


def ewc_penalty(model, anchors):
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if not parameters: raise ValueError("Model has no trainable parameters")
    penalty = next(iter(parameters.values())).new_zeros(())
    for anchor in anchors:
        if set(anchor["fisher"]) != set(parameters):
            raise ValueError("EWC anchor names do not match current model parameters")
        for name, parameter in parameters.items():
            if anchor["fisher"][name].shape != parameter.shape or anchor["parameters"][name].shape != parameter.shape:
                raise ValueError(f"EWC tensor shape does not match current parameter {name}")
            penalty = penalty + (anchor["fisher"][name] * (parameter - anchor["parameters"][name]).square()).sum()
    return penalty
