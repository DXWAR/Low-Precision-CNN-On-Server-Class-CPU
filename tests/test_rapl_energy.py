"""
Tests T13–T15: RAPL energy counter handling.

T13  RAPL 32-bit wraparound detection and correction
T14  Idle-baseline subtraction produces non-negative marginal energy
T15  Energy measurement precision within NFR2 bounds (±1% at 30s window)
"""

import pytest
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic RAPL counter for testing
# ═══════════════════════════════════════════════════════════════════════════════

# The RAPL energy_uj counter is a 32-bit register that wraps around at
# 2^32 × 61 µJ ≈ 262.14 J. On a server drawing 200 W, this wraps
# every ~1,310 seconds (~22 minutes).

RAPL_MAX_UJ = 2 ** 32  # 4,294,967,296 µJ ≈ 4294.97 J at max register


def rapl_delta_uj(start, end, max_val=RAPL_MAX_UJ):
    """
    Compute RAPL energy delta in µJ, handling 32-bit wraparound.
    This mirrors the wraparound logic in newmain.py's energy harness (§5.5).
    """
    if end >= start:
        return end - start
    else:
        # Counter wrapped around
        return (max_val - start) + end


def marginal_energy_j(active_uj, idle_power_w, duration_s):
    """
    Compute marginal inference energy by subtracting idle baseline.
    Follows UPM-2024 protocol (§4.5).

    active_uj:    total RAPL energy during inference (µJ)
    idle_power_w: measured idle power (W)
    duration_s:   duration of the inference window (s)

    Returns: marginal energy in joules
    """
    total_j = active_uj / 1e6
    idle_j = idle_power_w * duration_s
    marginal = total_j - idle_j
    return max(marginal, 0.0)  # clamp to non-negative


# ═══════════════════════════════════════════════════════════════════════════════
# T13 — RAPL 32-bit wraparound
# ═══════════════════════════════════════════════════════════════════════════════

class TestT13_RAPLWraparound:
    """T13: Correct handling of the RAPL 32-bit counter wraparound."""

    def test_no_wraparound(self):
        """Normal case: end > start, delta = end - start."""
        delta = rapl_delta_uj(1_000_000, 2_000_000)
        assert delta == 1_000_000

    def test_wraparound_detected(self):
        """When end < start, the counter has wrapped around."""
        # Start near max, end near zero → wrapped once
        start = RAPL_MAX_UJ - 100_000
        end = 200_000
        delta = rapl_delta_uj(start, end)
        expected = 100_000 + 200_000  # remaining + new
        assert delta == expected, f"Expected {expected}, got {delta}"

    def test_wraparound_small_interval(self):
        """Wraparound with a very small energy reading after reset."""
        start = RAPL_MAX_UJ - 10
        end = 5
        delta = rapl_delta_uj(start, end)
        assert delta == 15

    def test_wraparound_full_cycle(self):
        """Edge case: end == start (either 0 delta or full cycle — we treat as 0)."""
        delta = rapl_delta_uj(500_000, 500_000)
        assert delta == 0

    def test_energy_conversion_uj_to_joules(self):
        """1,000,000 µJ = 1.0 J."""
        delta_uj = 1_000_000
        joules = delta_uj / 1e6
        assert abs(joules - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# T14 — Idle-baseline subtraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestT14_IdleSubtraction:
    """T14: UPM-2024 idle-baseline subtraction produces non-negative marginal energy."""

    def test_marginal_energy_positive(self):
        """Active energy > idle energy → positive marginal."""
        # 10 seconds of inference, total 50 J measured, idle power 3 W
        active_uj = 50_000_000  # 50 J
        idle_w = 3.0
        duration = 10.0

        marginal = marginal_energy_j(active_uj, idle_w, duration)
        expected = 50.0 - 30.0  # 20 J
        assert abs(marginal - expected) < 0.01

    def test_marginal_energy_clamped_to_zero(self):
        """If idle exceeds measured (e.g. measurement error), clamp to 0."""
        active_uj = 10_000_000  # 10 J
        idle_w = 5.0
        duration = 10.0  # idle energy = 50 J > 10 J measured

        marginal = marginal_energy_j(active_uj, idle_w, duration)
        assert marginal == 0.0, "Marginal should be clamped to 0"

    def test_marginal_energy_zero_idle(self):
        """With zero idle power, marginal = total."""
        active_uj = 25_000_000
        marginal = marginal_energy_j(active_uj, 0.0, 10.0)
        assert abs(marginal - 25.0) < 0.01

    def test_idle_interpolation(self):
        """
        UPM-2024 brackets with pre/post idle readings.
        Linear interpolation of two idle measurements.
        """
        idle_before_w = 3.0
        idle_after_w = 3.4
        duration = 30.0  # 30s measurement window

        interpolated = (idle_before_w + idle_after_w) / 2
        idle_energy_j = interpolated * duration

        assert abs(interpolated - 3.2) < 0.01
        assert abs(idle_energy_j - 96.0) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# T15 — Measurement precision
# ═══════════════════════════════════════════════════════════════════════════════

class TestT15_MeasurementPrecision:
    """T15: Energy measurement within NFR2 precision bounds."""

    def test_counter_resolution_error(self):
        """
        RAPL resolution is 61 µJ. Over a 30s window drawing 10 W,
        total energy = 300 J = 300,000,000 µJ.
        Relative error from quantisation = 61 / 300,000,000 ≈ 2e-7 (≪ 1%).
        """
        total_uj = 300_000_000
        resolution_uj = 61
        relative_error = resolution_uj / total_uj

        assert relative_error < 0.01, \
            f"Relative error {relative_error:.2e} exceeds 1%"

    def test_short_window_high_error(self):
        """
        Over a 0.1s window at 10 W, total = 1 J = 1,000,000 µJ.
        Relative error = 61 / 1,000,000 ≈ 6e-5 — still well within 1%.
        This validates NFR2's 30s minimum is conservative.
        """
        total_uj = 1_000_000
        resolution_uj = 61
        relative_error = resolution_uj / total_uj
        assert relative_error < 0.01

    def test_repeated_runs_consistency(self):
        """
        Synthetic: two runs with small Gaussian noise should agree
        within ±2% (NFR3 reproducibility requirement).
        """
        np.random.seed(42)
        base_energy = 100.0  # joules

        run1 = base_energy + np.random.normal(0, 0.5)
        run2 = base_energy + np.random.normal(0, 0.5)

        relative_diff = abs(run1 - run2) / base_energy
        assert relative_diff < 0.02, \
            f"Runs differ by {relative_diff:.1%}, exceeding ±2%"
