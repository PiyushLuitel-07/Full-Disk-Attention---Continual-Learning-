import unittest

import torch

from modeling.attention_model import AttnNet


class ModelTests(unittest.TestCase):
    def test_output_contract(self) -> None:
        model = AttnNet(attention=True).eval()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 7_458_178)
        with torch.inference_mode():
            logits, attention1, attention2, attention3 = model(
                torch.zeros(1, 1, 256, 256)
            )
        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertEqual(attention1.shape[0:2], (1, 1))
        self.assertEqual(attention2.shape[0:2], (1, 1))
        self.assertEqual(attention3.shape[0:2], (1, 1))


if __name__ == "__main__":
    unittest.main()
