"""Classic multi-anchor Elastic Weight Consolidation (EWC).

For every completed stage we retain:

* an anchor: the selected value of each trainable parameter; and
* a diagonal empirical Fisher: an importance value for each scalar parameter.

The helper returns the raw quadratic sum. The training loop applies
``0.5 * ewc_lambda`` exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


TensorDictionary = dict[str, torch.Tensor]


def unwrap_model(model: nn.Module) -> nn.Module:
    """Remove DataParallel-like wrappers before matching parameter names."""

    return model.module if hasattr(model, "module") else model


def _trainable_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {
        name: parameter
        for name, parameter in unwrap_model(model).named_parameters()
        if parameter.requires_grad
    }


def clone_trainable_parameters(model: nn.Module) -> TensorDictionary:
    """Make a detached CPU anchor that cannot receive gradients."""

    return {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in _trainable_parameters(model).items()
    }


def _model_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError("Model output must be logits or a sequence beginning with logits")


def estimate_diagonal_fisher(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 0,
) -> tuple[TensorDictionary, int]:
    """Estimate the exact per-example empirical diagonal Fisher.

    Each sample is differentiated separately. Squaring one batch-summed gradient
    would introduce cross-sample terms and is therefore deliberately avoided.
    """

    base_model = unwrap_model(model)
    parameters = _trainable_parameters(base_model)
    fisher = {name: torch.zeros_like(value, device=device) for name, value in parameters.items()}
    sample_count = 0
    previous_training_state = base_model.training
    base_model.eval()

    try:
        for images, targets, _paths in loader:
            images, targets = images.to(device), targets.to(device)
            for sample_index in range(images.shape[0]):
                if max_samples and sample_count >= max_samples:
                    break
                base_model.zero_grad(set_to_none=True)
                logits = _model_logits(base_model(images[sample_index : sample_index + 1]))
                log_probability = F.log_softmax(logits, dim=1)[0, targets[sample_index]]
                log_probability.backward()

                for name, parameter in parameters.items():
                    if parameter.grad is not None:
                        fisher[name].add_(parameter.grad.detach().square())
                sample_count += 1
            if max_samples and sample_count >= max_samples:
                break
    finally:
        base_model.zero_grad(set_to_none=True)
        base_model.train(previous_training_state)

    if sample_count == 0:
        raise ValueError("Cannot estimate Fisher from an empty loader")
    return {
        name: (value / sample_count).detach().cpu().float()
        for name, value in fisher.items()
    }, sample_count


@dataclass
class EWCState:
    """The anchors and Fishers for all completed chronological stages."""

    anchors: list[TensorDictionary] = field(default_factory=list)
    fishers: list[TensorDictionary] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.anchors)

    def add_stage(self, model: nn.Module, fisher: TensorDictionary) -> None:
        anchor = clone_trainable_parameters(model)
        if set(anchor) != set(fisher):
            raise ValueError("Fisher keys do not match the model's trainable parameters")
        self.anchors.append(anchor)
        self.fishers.append(
            {name: value.detach().cpu().float().clone() for name, value in fisher.items()}
        )

    def to(self, device: torch.device) -> "EWCState":
        """Cache the state on the training device once, not once per batch."""

        self.anchors = [
            {name: value.to(device) for name, value in anchor.items()}
            for anchor in self.anchors
        ]
        self.fishers = [
            {name: value.to(device) for name, value in fisher.items()}
            for fisher in self.fishers
        ]
        return self

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Return sum(F * (parameter - anchor)^2) across completed stages."""

        parameters = _trainable_parameters(model)
        first_parameter = next(iter(parameters.values()))
        penalty = first_parameter.new_zeros(())
        expected = set(parameters)

        if len(self.anchors) != len(self.fishers):
            raise ValueError("EWC state has a different number of anchors and Fishers")
        for anchor, fisher in zip(self.anchors, self.fishers):
            if set(anchor) != expected or set(fisher) != expected:
                raise ValueError("Saved EWC parameter keys do not match the current model")
            for name, parameter in parameters.items():
                old_value = anchor[name]
                importance = fisher[name]
                if old_value.device != parameter.device or importance.device != parameter.device:
                    raise ValueError(
                        "EWC state is on a different device. Call ewc_state.to(device) "
                        "once before training."
                    )
                penalty = penalty + (importance * (parameter - old_value).square()).sum()
        return penalty

    def state_dict(self) -> dict[str, list[TensorDictionary]]:
        """Return a portable CPU copy suitable for a checkpoint."""

        return {
            "anchors": [
                {name: value.detach().cpu() for name, value in anchor.items()}
                for anchor in self.anchors
            ],
            "fishers": [
                {name: value.detach().cpu() for name, value in fisher.items()}
                for fisher in self.fishers
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "EWCState":
        anchors = state.get("anchors", [])
        fishers = state.get("fishers", [])
        if len(anchors) != len(fishers):
            raise ValueError("Invalid EWC state: anchor/Fisher counts differ")
        return cls(anchors=anchors, fishers=fishers)
