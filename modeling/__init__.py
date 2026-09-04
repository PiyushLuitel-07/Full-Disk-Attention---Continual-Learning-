"""Small, explainable continual-learning package for full-disk magnetograms."""

from .attention_model import AttnNet
from .ewc import EWCState, estimate_diagonal_fisher

__all__ = ["AttnNet", "EWCState", "estimate_diagonal_fisher"]

