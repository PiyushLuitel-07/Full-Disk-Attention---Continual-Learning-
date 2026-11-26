import torch
import torch.nn.functional as F
from torch import nn


def unwrap_model(model):
    """Return the original model when DataParallel is being used."""
    if isinstance(model, nn.DataParallel):
        return model.module

    return model


def calculate_fisher(model, fisher_loader, device):
    """
    Estimate parameter importance using every batch from the
    original, unaugmented training data.
    """
    base_model = unwrap_model(model)

    fisher = {
        name: torch.zeros_like(parameter)
        for name, parameter in base_model.named_parameters()
        if parameter.requires_grad
    }

    was_training = model.training
    model.eval()

    number_of_batches = 0

    for images, targets in fisher_loader:
        images = images.to(device)
        targets = targets.to(device)

        model.zero_grad(set_to_none=True)

        # attention_model.py returns:
        # [class scores, attention1, attention2, attention3]
        scores = model(images)[0]

        loss = F.cross_entropy(scores, targets)
        loss.backward()

        for name, parameter in base_model.named_parameters():
            if parameter.grad is not None:
                fisher[name] += parameter.grad.detach().pow(2)

        number_of_batches += 1

    if number_of_batches == 0:
        raise ValueError("The Fisher DataLoader is empty.")

    for name in fisher:
        fisher[name] /= number_of_batches

    model.zero_grad(set_to_none=True)
    model.train(was_training)

    return fisher


def save_parameters(model):
    """Save model parameters after completing one stage."""
    base_model = unwrap_model(model)

    return {
        name: parameter.detach().clone()
        for name, parameter in base_model.named_parameters()
        if parameter.requires_grad
    }


def ewc_penalty(model, ewc_history):
    """
    Calculate protection against forgetting previous stages.

    Each ewc_history entry must contain:
        {
            "fisher": fisher_dictionary,
            "parameters": saved_parameter_dictionary
        }
    """
    base_model = unwrap_model(model)
    penalty = next(base_model.parameters()).new_zeros(())

    for old_stage in ewc_history:
        old_fisher = old_stage["fisher"]
        old_parameters = old_stage["parameters"]

        for name, current_parameter in base_model.named_parameters():
            if name in old_fisher:
                change = current_parameter - old_parameters[name]

                penalty += (
                    old_fisher[name] * change.pow(2)
                ).sum()

    return penalty