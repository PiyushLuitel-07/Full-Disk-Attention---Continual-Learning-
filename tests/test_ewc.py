import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from modeling.ewc import EWCState, estimate_diagonal_fisher


class TinyDataset(Dataset):
    def __init__(self) -> None:
        self.inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.targets = torch.tensor([0, 1])

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return self.inputs[index], self.targets[index], str(index)


class EwcTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)
        self.model = nn.Linear(2, 2)
        self.loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False)

    def test_fisher_is_nonnegative_and_detached(self) -> None:
        fisher, count = estimate_diagonal_fisher(
            self.model, self.loader, torch.device("cpu")
        )
        self.assertEqual(count, 2)
        self.assertTrue(any(torch.count_nonzero(value) for value in fisher.values()))
        for value in fisher.values():
            self.assertTrue(torch.all(value >= 0))
            self.assertFalse(value.requires_grad)

    def test_penalty_is_zero_at_anchor_then_positive(self) -> None:
        fisher, _ = estimate_diagonal_fisher(
            self.model, self.loader, torch.device("cpu")
        )
        state = EWCState()
        state.add_stage(self.model, fisher)
        self.assertAlmostEqual(state.penalty(self.model).item(), 0.0)
        with torch.no_grad():
            self.model.weight[0, 0].add_(1.0)
        self.assertGreater(state.penalty(self.model).item(), 0.0)

    def test_missing_key_is_rejected(self) -> None:
        fisher, _ = estimate_diagonal_fisher(
            self.model, self.loader, torch.device("cpu")
        )
        fisher.pop(next(iter(fisher)))
        with self.assertRaises(ValueError):
            EWCState().add_stage(self.model, fisher)

    def test_multiple_stage_anchors_survive_serialization(self) -> None:
        fisher, _ = estimate_diagonal_fisher(
            self.model, self.loader, torch.device("cpu")
        )
        state = EWCState()
        state.add_stage(self.model, fisher)
        with torch.no_grad():
            self.model.weight.add_(0.25)
        state.add_stage(self.model, fisher)

        restored = EWCState.from_state_dict(state.state_dict()).to(
            torch.device("cpu")
        )
        self.assertEqual(len(restored), 2)
        with torch.no_grad():
            self.model.weight.add_(0.25)
        self.assertGreater(restored.penalty(self.model).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
