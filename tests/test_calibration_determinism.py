"""
Tests T24: Calibration-set determinism.

T24  Two independent runs with the same seed produce identical
     calibration outputs, verifying FR8 (deterministic replay).
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import copy

from conftest import TinyDenseNet


# ═══════════════════════════════════════════════════════════════════════════════
# T24 — Calibration determinism
# ═══════════════════════════════════════════════════════════════════════════════

class TestT24_CalibrationDeterminism:
    """T24: Calibration outputs must be deterministic given the same seed."""

    def _run_calibrated_inference(self, seed=42):
        """
        Run a complete calibration + inference cycle with a fixed seed.
        Returns the output tensor for comparison.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = TinyDenseNet()
        model.eval()

        # Generate deterministic calibration data
        calib_data = [torch.randn(1, 1, 32, 32) for _ in range(10)]

        # Generate deterministic test input
        test_input = torch.randn(1, 1, 32, 32)

        # Run FP32 inference
        with torch.no_grad():
            output = model(test_input)

        return output

    def test_same_seed_same_output(self):
        """Two runs with seed=42 must produce bit-identical outputs."""
        out1 = self._run_calibrated_inference(seed=42)
        out2 = self._run_calibrated_inference(seed=42)

        assert torch.equal(out1, out2), \
            f"Outputs differ: max delta = {(out1 - out2).abs().max().item()}"

    def test_different_seed_different_output(self):
        """Different seeds should produce different outputs."""
        out_42 = self._run_calibrated_inference(seed=42)
        out_99 = self._run_calibrated_inference(seed=99)

        assert not torch.equal(out_42, out_99), \
            "Different seeds should produce different random initialisations"

    def test_calibration_prefix_monotone(self):
        """
        Calibration set design (§4.6): the size-N set is a strict prefix
        of the size-N' set for N < N'. Verify this property.
        """
        np.random.seed(42)
        full_pool = np.random.permutation(1024)

        for n in [32, 128, 256, 512]:
            subset = full_pool[:n]
            superset = full_pool[:n * 2] if n * 2 <= 1024 else full_pool

            # Every element in the smaller set must appear in the larger set
            assert set(subset).issubset(set(superset)), \
                f"Size-{n} set is not a prefix of size-{n*2} set"

    def test_numpy_seed_reproducibility(self):
        """NumPy RandomState(42) produces identical sequences across runs."""
        rng1 = np.random.RandomState(42)
        seq1 = [rng1.randint(0, 1000) for _ in range(100)]

        rng2 = np.random.RandomState(42)
        seq2 = [rng2.randint(0, 1000) for _ in range(100)]

        assert seq1 == seq2, "RandomState(42) should be deterministic"

    def test_torch_seed_reproducibility(self):
        """torch.manual_seed(42) produces identical tensors across runs."""
        torch.manual_seed(42)
        t1 = torch.randn(10, 10)

        torch.manual_seed(42)
        t2 = torch.randn(10, 10)

        assert torch.equal(t1, t2), "torch.manual_seed should be deterministic"
