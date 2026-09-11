import unittest

import pandas as pd

from Baselines.benchmark import _attach_realism_metrics


class AttachRealismMetricsTest(unittest.TestCase):
    def test_preserves_one_row_per_training_seed_rollout(self) -> None:
        benchmark = pd.DataFrame(
            {
                "model": ["residual_marl"] * 4,
                "seed": [0, 1, 0, 1],
                "train_seed": [0, 0, 1, 1],
                "arrival_rate": [0.6, 0.7, 0.8, 0.9],
            }
        )
        realism = pd.DataFrame(
            {
                "model": ["residual_marl"] * 4,
                "seed": [0, 1, 0, 1],
                "realism_score": [0.4, 0.5, 0.6, 0.7],
            }
        )

        joined = _attach_realism_metrics(benchmark, realism)

        self.assertEqual(len(joined), len(benchmark))
        self.assertEqual(joined["train_seed"].tolist(), [0, 0, 1, 1])
        self.assertEqual(joined["realism_score"].tolist(), [0.4, 0.5, 0.6, 0.7])

    def test_rejects_misaligned_rollout_order(self) -> None:
        benchmark = pd.DataFrame(
            {"model": ["utility_pt", "residual_marl"], "seed": [0, 0]}
        )
        realism = pd.DataFrame(
            {
                "model": ["residual_marl", "utility_pt"],
                "seed": [0, 0],
                "realism_score": [0.4, 0.5],
            }
        )

        with self.assertRaisesRegex(ValueError, "not aligned"):
            _attach_realism_metrics(benchmark, realism)


if __name__ == "__main__":
    unittest.main()
