"""Validated diagonal-EWC operations used by Experiment 15."""
import torch


def load_ewc_anchor(saved_anchor, device):
    required = {"stage", "fisher", "parameters"}
    if required - set(saved_anchor): raise ValueError(f"Incomplete EWC anchor: {required - set(saved_anchor)}")
    fisher, parameters = saved_anchor["fisher"], saved_anchor["parameters"]
    if not fisher or set(fisher) != set(parameters): raise ValueError("Fisher and parameter names do not match")
    for name in fisher:
        if fisher[name].shape != parameters[name].shape: raise ValueError(f"EWC shape mismatch: {name}")
        if not torch.isfinite(fisher[name]).all() or (fisher[name] < 0).any():
            raise ValueError(f"Invalid Fisher values: {name}")
        if not torch.isfinite(parameters[name]).all(): raise ValueError(f"Invalid parameter snapshot: {name}")
    return {"stage": int(saved_anchor["stage"]),
        "fisher": {name: value.to(device) for name, value in fisher.items()},
        "parameters": {name: value.to(device) for name, value in parameters.items()}}


def ewc_penalty(model, anchors):
    current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not current: raise ValueError("The model has no trainable parameters")
    penalty = next(iter(current.values())).new_zeros(())
    for anchor in anchors:
        if set(anchor["fisher"]) != set(current): raise ValueError("EWC/model parameter names do not match")
        for name, parameter in current.items():
            fisher, old_parameter = anchor["fisher"][name], anchor["parameters"][name]
            if fisher.shape != parameter.shape or old_parameter.shape != parameter.shape:
                raise ValueError(f"EWC/model shape mismatch: {name}")
            penalty += (fisher * (parameter - old_parameter).square()).sum()
    return penalty
