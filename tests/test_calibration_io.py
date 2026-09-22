"""The deployed utility prior must match the calibration's validation choice."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from RL.calibration_io import clip_params_rl, load_base_params


class CalibrationIOTest(unittest.TestCase):
    def test_default_uses_validation_selected_working_params(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps({
                "working_params": {"S_v": 2.5, "xi_i": 4.5},
                "robust_params": {"S_v": 1.5, "xi_i": 3.5},
                "best_params": {"S_v": 3.5, "xi_i": 2.5},
            }))
            self.assertEqual(load_base_params(path)["S_v"], 2.5)
            self.assertEqual(load_base_params(path, prefer="robust")["S_v"], 1.5)
            self.assertEqual(load_base_params(path, prefer="best")["S_v"], 3.5)

    def test_residual_shape_stays_in_admissible_range(self) -> None:
        self.assertEqual(clip_params_rl({"xi_i": 12.0})["xi_i"], 5.0)
        self.assertEqual(clip_params_rl({"xi_i": 0.0})["xi_i"], 1.1)


if __name__ == "__main__":
    unittest.main()
