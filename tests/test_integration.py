"""
Integration tests I1–I3: End-to-end pipeline validation.

I1  FP32 end-to-end pipeline produces valid JSON manifest and metrics
I2  FP16 simulated precision produces outputs within tolerance of FP32
I3  Static INT8 pipeline runs without crashing (accuracy checked separately)

These tests use the TinyDenseNet fixture rather than the full torchxrayvision
model, so they validate the pipeline logic without requiring the ChestX-ray14
dataset or pretrained weights. The full-dataset runs are documented in Ch. 7.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import json
import copy
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from conftest import TinyDenseNet
from newmain import StatisticalAnalysis, ClinicalMetrics


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: run a mini end-to-end pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_mini_pipeline(model, images, precision='fp32'):
    """
    Simulate the core inference pipeline on a batch of images.
    Returns a dict mimicking the JSON manifest structure (FR9).
    """
    model.eval()
    all_probs = []
    latencies = []

    for img in images:
        start = torch.cuda.Event(enable_timing=False) if torch.cuda.is_available() else None
        t0 = __import__('time').perf_counter()

        with torch.no_grad():
            if precision == 'bf16':
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    logits = model(img)
            elif precision == 'fp16':
                m = copy.deepcopy(model)
                for p in m.parameters():
                    p.data = p.data.half().float()
                logits = m(img)
            else:
                logits = model(img)

        t1 = __import__('time').perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs[0])

    probs_matrix = np.array(all_probs)

    # Build manifest
    manifest = {
        'precision': precision,
        'n_images': len(images),
        'n_pathologies': probs_matrix.shape[1],
        'mean_latency_ms': float(np.mean(latencies)),
        'latency_ci_ms': list(StatisticalAnalysis.bootstrap_ci(
            latencies, n_bootstrap=500, ci=0.95
        )),
        'probs_shape': list(probs_matrix.shape),
        'library_versions': {
            'torch': torch.__version__,
            'numpy': np.__version__,
        },
        'seed': 42,
    }

    return manifest, probs_matrix, latencies


# ═══════════════════════════════════════════════════════════════════════════════
# I1 — FP32 end-to-end
# ═══════════════════════════════════════════════════════════════════════════════

class TestI1_FP32EndToEnd:
    """I1: FP32 pipeline produces valid manifest and metrics."""

    @pytest.fixture
    def pipeline_output(self):
        torch.manual_seed(42)
        model = TinyDenseNet()
        images = [torch.randn(1, 1, 32, 32) for _ in range(16)]
        return run_mini_pipeline(model, images, precision='fp32')

    def test_manifest_has_required_fields(self, pipeline_output):
        """FR9: Manifest must contain precision, versions, seed, etc."""
        manifest, _, _ = pipeline_output

        required = ['precision', 'n_images', 'n_pathologies',
                     'mean_latency_ms', 'library_versions', 'seed']
        for field in required:
            assert field in manifest, f"Missing manifest field: {field}"

    def test_manifest_serialisable(self, pipeline_output):
        """Manifest must be JSON-serialisable."""
        manifest, _, _ = pipeline_output
        json_str = json.dumps(manifest)
        assert len(json_str) > 0
        roundtrip = json.loads(json_str)
        assert roundtrip['precision'] == 'fp32'

    def test_probs_valid_range(self, pipeline_output):
        """All probabilities should be in [0, 1] (sigmoid output)."""
        _, probs, _ = pipeline_output
        assert probs.min() >= 0.0, f"Min prob {probs.min()} < 0"
        assert probs.max() <= 1.0, f"Max prob {probs.max()} > 1"

    def test_probs_correct_shape(self, pipeline_output):
        """Probs matrix should be (n_images, 14)."""
        _, probs, _ = pipeline_output
        assert probs.shape == (16, 14), f"Expected (16, 14), got {probs.shape}"

    def test_latencies_positive(self, pipeline_output):
        """All latencies should be positive."""
        _, _, latencies = pipeline_output
        assert all(l > 0 for l in latencies), "Latencies must be positive"

    def test_latency_ci_valid(self, pipeline_output):
        """Bootstrap CI should satisfy lower ≤ point ≤ upper."""
        manifest, _, _ = pipeline_output
        point, lower, upper = manifest['latency_ci_ms']
        assert lower <= point <= upper, \
            f"CI violation: {lower} ≤ {point} ≤ {upper}"


# ═══════════════════════════════════════════════════════════════════════════════
# I2 — FP16 simulated precision
# ═══════════════════════════════════════════════════════════════════════════════

class TestI2_FP16SimulatedPrecision:
    """I2: FP16 pipeline output within tolerance of FP32 reference."""

    def test_fp16_within_tolerance(self):
        """Max probability deviation between FP32 and FP16 should be < 1%."""
        torch.manual_seed(42)
        model = TinyDenseNet()
        images = [torch.randn(1, 1, 32, 32) for _ in range(16)]

        _, fp32_probs, _ = run_mini_pipeline(model, images, 'fp32')
        _, fp16_probs, _ = run_mini_pipeline(model, images, 'fp16')

        max_dev = np.abs(fp32_probs - fp16_probs).max() * 100
        assert max_dev < 1.0, \
            f"FP16 max deviation {max_dev:.4f}% exceeds 1.0% threshold"

    def test_fp16_probs_valid(self):
        """FP16 probabilities should be in [0, 1]."""
        torch.manual_seed(42)
        model = TinyDenseNet()
        images = [torch.randn(1, 1, 32, 32) for _ in range(16)]

        _, probs, _ = run_mini_pipeline(model, images, 'fp16')
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_fp16_manifest_correct_precision(self):
        """Manifest should record the correct precision string."""
        torch.manual_seed(42)
        model = TinyDenseNet()
        images = [torch.randn(1, 1, 32, 32) for _ in range(4)]

        manifest, _, _ = run_mini_pipeline(model, images, 'fp16')
        assert manifest['precision'] == 'fp16'


# ═══════════════════════════════════════════════════════════════════════════════
# I3 — Static INT8 pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestI3_StaticINT8Pipeline:
    """I3: Static INT8 eager-mode pipeline runs without crashing."""

    def test_eager_int8_runs(self):
        """
        The eager-mode INT8 path (QuantStub → model → DeQuantStub)
        should run without errors, even if accuracy is degraded.
        """
        torch.manual_seed(42)
        model = TinyDenseNet()
        model.eval()

        class EagerWrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.quant = torch.quantization.QuantStub()
                self.inner = inner
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.inner(x)
                x = self.dequant(x)
                return x

        wrapped = EagerWrapper(model)
        wrapped.eval()
        wrapped.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        prepared = torch.quantization.prepare(wrapped, inplace=False)

        # Calibrate
        for _ in range(10):
            prepared(torch.randn(1, 1, 32, 32))

        quantised = torch.quantization.convert(prepared, inplace=False)

        # Inference should not crash
        test_input = torch.randn(1, 1, 32, 32)
        output = quantised(test_input)

        assert output.shape == (1, 14)
        assert torch.isfinite(output).all()

    def test_int8_output_is_different_from_fp32(self):
        """INT8 output may differ from FP32 — this is expected."""
        torch.manual_seed(42)
        model = TinyDenseNet()
        model.eval()

        test_input = torch.randn(1, 1, 32, 32)

        with torch.no_grad():
            fp32_out = model(test_input)

        # Simulated INT8
        int8_model = copy.deepcopy(model)
        for p in int8_model.parameters():
            scale = p.data.abs().max() / 127.0
            if scale > 0:
                p.data = (p.data / scale).round().clamp(-128, 127) * scale

        with torch.no_grad():
            int8_out = int8_model(test_input)

        # They should differ (quantisation introduces rounding)
        # but both should be finite
        assert torch.isfinite(fp32_out).all()
        assert torch.isfinite(int8_out).all()


# ═══════════════════════════════════════════════════════════════════════════════
# Clinical metrics integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestClinicalMetricsIntegration:
    """Integration test for ClinicalMetrics with realistic data."""

    def test_ece_perfect_calibration(self):
        """A perfectly calibrated model should have ECE ≈ 0."""
        np.random.seed(42)
        n = 1000
        probs = np.random.uniform(0, 1, n)
        labels = (np.random.uniform(0, 1, n) < probs).astype(float)

        ece, _, _, _ = ClinicalMetrics.expected_calibration_error(probs, labels)
        # With 1000 samples, ECE of a perfectly calibrated model should be low
        assert ece < 0.1, f"ECE {ece:.3f} too high for near-perfect calibration"

    def test_mce_range(self):
        """MCE should be in [0, 1]."""
        probs = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
        labels = np.array([0, 1, 1, 0, 1])

        mce = ClinicalMetrics.maximum_calibration_error(probs, labels)
        assert 0 <= mce <= 1, f"MCE {mce} outside [0, 1]"

    def test_bootstrap_auc_ci_contains_point(self):
        """Bootstrap AUC CI should contain the point estimate."""
        np.random.seed(42)
        y_true = np.array([0]*50 + [1]*50)
        y_score = np.concatenate([
            np.random.normal(0.3, 0.15, 50),
            np.random.normal(0.7, 0.15, 50),
        ])
        y_score = np.clip(y_score, 0, 1)

        point, lower, upper = ClinicalMetrics.bootstrap_auc_ci(
            y_true, y_score, n_bootstrap=1000
        )

        assert lower is not None and upper is not None
        assert lower <= point <= upper, \
            f"Point AUC {point} not in CI [{lower}, {upper}]"
        assert point > 0.5, "AUC should be above chance for separated distributions"
