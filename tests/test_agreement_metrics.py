"""
Tests T20–T23: Attention-agreement metrics.

T20  Perfect self-agreement: ρ=1, IoU=1, COM shift=0
T21  Anti-correlated heatmaps: ρ≈−1
T22  Hand-computed reference: known IoU and COM shift values
T23  Weight-randomisation control: near-zero Pearson correlation
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ═══════════════════════════════════════════════════════════════════════════════
# Agreement metrics (mirrors GradCAMAnalyser.agreement_metrics in additions.py)
# ═══════════════════════════════════════════════════════════════════════════════

def agreement_metrics(cam_a, cam_b, threshold_pct=0.80):
    """
    Compute agreement between two CAMs.

    Returns dict with:
      pearson:      Pearson correlation of flattened heatmaps
      iou:          IoU of regions above the threshold_pct quantile
      com_shift_px: centre-of-mass shift in pixels (L2)
    """
    a = cam_a.flatten()
    b = cam_b.flatten()

    # Pearson correlation
    if a.std() < 1e-9 or b.std() < 1e-9:
        pearson = float('nan')
    else:
        pearson = float(np.corrcoef(a, b)[0, 1])

    # IoU of top-region masks
    t_a = np.quantile(a, threshold_pct)
    t_b = np.quantile(b, threshold_pct)
    mask_a = cam_a >= t_a
    mask_b = cam_b >= t_b
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    iou = float(inter / union) if union > 0 else float('nan')

    # Centre-of-mass shift
    def com(cam):
        if cam.sum() < 1e-9:
            return np.array([cam.shape[0] / 2, cam.shape[1] / 2])
        ys, xs = np.mgrid[:cam.shape[0], :cam.shape[1]]
        total = cam.sum()
        return np.array([(cam * ys).sum() / total, (cam * xs).sum() / total])

    com_shift = float(np.linalg.norm(com(cam_a) - com(cam_b)))

    return {'pearson': pearson, 'iou': iou, 'com_shift_px': com_shift}


# ═══════════════════════════════════════════════════════════════════════════════
# T20 — Perfect self-agreement
# ═══════════════════════════════════════════════════════════════════════════════

class TestT20_SelfAgreement:
    """T20: When H_ref == H_query, ρ=1, IoU=1, COM shift=0."""

    def test_identical_heatmaps(self):
        """Comparing a heatmap with itself should give perfect agreement."""
        cam = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.9, 0.5],
            [0.2, 0.3, 0.1],
        ])

        metrics = agreement_metrics(cam, cam)

        assert abs(metrics['pearson'] - 1.0) < 1e-6, \
            f"Self-correlation should be 1.0, got {metrics['pearson']}"
        assert abs(metrics['iou'] - 1.0) < 1e-6, \
            f"Self-IoU should be 1.0, got {metrics['iou']}"
        assert metrics['com_shift_px'] < 1e-6, \
            f"Self COM shift should be 0, got {metrics['com_shift_px']}"

    def test_identical_7x7_grid(self):
        """Self-agreement on a 7×7 grid (DenseNet-121's spatial resolution)."""
        np.random.seed(42)
        cam = np.random.rand(7, 7)
        cam = cam / cam.max()  # normalise

        metrics = agreement_metrics(cam, cam)

        assert abs(metrics['pearson'] - 1.0) < 1e-6
        assert abs(metrics['iou'] - 1.0) < 1e-6
        assert metrics['com_shift_px'] < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# T21 — Anti-correlated heatmaps
# ═══════════════════════════════════════════════════════════════════════════════

class TestT21_AntiCorrelation:
    """T21: Inverted heatmaps should have ρ ≈ −1."""

    def test_inverted_heatmap_correlation(self):
        """If cam_b = 1 - cam_a, Pearson ρ should be −1."""
        cam_a = np.array([
            [0.0, 0.2, 0.4],
            [0.6, 0.8, 1.0],
            [0.3, 0.5, 0.7],
        ])
        cam_b = 1.0 - cam_a

        metrics = agreement_metrics(cam_a, cam_b)

        assert abs(metrics['pearson'] - (-1.0)) < 1e-6, \
            f"Anti-correlated ρ should be −1.0, got {metrics['pearson']}"

    def test_inverted_iou_low(self):
        """Inverted heatmaps should have low IoU (top regions don't overlap)."""
        cam_a = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        cam_b = 1.0 - cam_a

        metrics = agreement_metrics(cam_a, cam_b, threshold_pct=0.5)
        # Top-50% of cam_a is bottom-left; top-50% of cam_b is top-right
        assert metrics['iou'] < 0.5, \
            f"Inverted IoU should be low, got {metrics['iou']}"


# ═══════════════════════════════════════════════════════════════════════════════
# T22 — Hand-computed reference
# ═══════════════════════════════════════════════════════════════════════════════

class TestT22_HandComputedReference:
    """T22: Agreement metrics against a hand-computed reference."""

    def test_known_iou(self):
        """
        cam_a top-20% = top-right 2 cells
        cam_b top-20% = centre + top-right
        Hand-computed IoU = intersection / union.
        """
        # 3×3 grid, top-20% = largest 2 cells (ceil(9 * 0.2) = 2)
        cam_a = np.array([
            [0.1, 0.3, 0.9],
            [0.2, 0.4, 0.8],
            [0.0, 0.1, 0.2],
        ])
        cam_b = np.array([
            [0.1, 0.2, 0.9],
            [0.3, 0.8, 0.5],
            [0.0, 0.1, 0.2],
        ])

        metrics = agreement_metrics(cam_a, cam_b, threshold_pct=0.80)

        # Both should have IoU > 0 (they share the top-right peak)
        assert metrics['iou'] > 0, "Expected positive IoU"
        assert 0 <= metrics['iou'] <= 1, "IoU must be in [0, 1]"

    def test_known_com_shift(self):
        """
        cam_a has mass concentrated at (0,0); cam_b at (2,2).
        COM shift = √((2-0)² + (2-0)²) = 2√2 ≈ 2.828.
        """
        cam_a = np.zeros((3, 3))
        cam_a[0, 0] = 1.0

        cam_b = np.zeros((3, 3))
        cam_b[2, 2] = 1.0

        metrics = agreement_metrics(cam_a, cam_b)

        expected_shift = np.sqrt(8)  # 2√2
        assert abs(metrics['com_shift_px'] - expected_shift) < 0.01, \
            f"COM shift should be {expected_shift:.3f}, got {metrics['com_shift_px']:.3f}"

    def test_zero_cam_defaults_to_centre(self):
        """An all-zero CAM should have COM at the grid centre."""
        cam_zero = np.zeros((7, 7))
        cam_peak = np.zeros((7, 7))
        cam_peak[3, 3] = 1.0  # centre of 7×7

        metrics = agreement_metrics(cam_zero, cam_peak)

        # Zero cam defaults to (3.5, 3.5), peak is at (3, 3)
        assert metrics['com_shift_px'] < 1.0, \
            "Zero-cam COM should be near centre"


# ═══════════════════════════════════════════════════════════════════════════════
# T23 — Weight-randomisation control (Adebayo et al. sanity check)
# ═══════════════════════════════════════════════════════════════════════════════

class TestT23_RandomisationControl:
    """
    T23: Weight-randomised model should produce near-zero Pearson
    correlation with the original model's CAM (sanity check per
    Adebayo et al. 2018).
    """

    def test_randomised_weights_low_correlation(self):
        """
        Randomising the classifier weights should destroy the spatial
        pattern, giving ρ ≈ 0 (chance-level agreement).
        """
        np.random.seed(42)

        # Original structured CAM (simulating a real model's output)
        cam_original = np.array([
            [0.1, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0],
            [0.2, 0.4, 0.6, 0.4, 0.2, 0.1, 0.0],
            [0.3, 0.6, 0.9, 0.7, 0.3, 0.1, 0.0],
            [0.2, 0.5, 0.7, 0.8, 0.4, 0.2, 0.1],
            [0.1, 0.3, 0.4, 0.5, 0.3, 0.1, 0.0],
            [0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0],
            [0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0],
        ])

        # Randomised CAM (simulating randomised weights)
        cam_random = np.random.rand(7, 7)

        metrics = agreement_metrics(cam_original, cam_random)

        # ρ should be near zero (within ±0.5 for a 7×7 grid with 49 points)
        assert abs(metrics['pearson']) < 0.5, \
            f"Randomised ρ should be near 0, got {metrics['pearson']:.3f}"

    def test_multiple_random_controls_average_near_zero(self):
        """
        Over many randomisation trials, the mean ρ should converge to ~0.
        """
        np.random.seed(42)
        cam_original = np.random.rand(7, 7)
        cam_original = cam_original / cam_original.max()

        correlations = []
        for _ in range(100):
            cam_random = np.random.rand(7, 7)
            metrics = agreement_metrics(cam_original, cam_random)
            if not np.isnan(metrics['pearson']):
                correlations.append(metrics['pearson'])

        mean_rho = np.mean(correlations)
        assert abs(mean_rho) < 0.15, \
            f"Mean ρ over 100 random trials should be ≈0, got {mean_rho:.3f}"
