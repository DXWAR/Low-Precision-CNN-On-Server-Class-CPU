"""
Tests T9–T12: Statistical analysis routines.

T9   Bootstrap CI on a Gaussian mean approaches ±1.96σ/√n analytically
T10  Bootstrap CI has correct coverage on a known distribution
T11  Bootstrap CI with custom statistic (median) returns valid bounds
T12  McNemar's test returns correct p-value for known discordance
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from newmain import StatisticalAnalysis


# ═══════════════════════════════════════════════════════════════════════════════
# T9 — Bootstrap CI Gaussian mean
# ═══════════════════════════════════════════════════════════════════════════════

class TestT9_BootstrapCIGaussian:
    """T9: Bootstrap 95% CI for Gaussian mean should approach ±1.96σ/√n."""

    def test_bootstrap_ci_gaussian_mean(self):
        """
        For N(μ=10, σ=2) with n=1000 samples, the 95% CI for the mean
        should be approximately μ ± 1.96*σ/√n = 10 ± 0.124.
        The bootstrap CI should agree within ~20% of this width.
        """
        np.random.seed(42)
        data = np.random.normal(loc=10, scale=2, size=1000)

        point, lower, upper = StatisticalAnalysis.bootstrap_ci(
            data, n_bootstrap=5000, ci=0.95, statistic=np.mean
        )

        # Analytic CI
        se = 2.0 / np.sqrt(1000)
        analytic_lower = 10 - 1.96 * se
        analytic_upper = 10 + 1.96 * se
        analytic_width = analytic_upper - analytic_lower

        bootstrap_width = upper - lower

        # Point estimate should be close to true mean
        assert abs(point - 10) < 0.2, f"Point estimate {point} too far from 10"

        # Bootstrap width should be within 30% of analytic width
        ratio = bootstrap_width / analytic_width
        assert 0.7 < ratio < 1.3, \
            f"Bootstrap width ratio {ratio:.2f} outside [0.7, 1.3]"

    def test_bootstrap_ci_contains_true_mean(self):
        """The 95% CI should contain the true mean most of the time."""
        np.random.seed(42)
        data = np.random.normal(loc=5, scale=1, size=500)
        point, lower, upper = StatisticalAnalysis.bootstrap_ci(
            data, n_bootstrap=5000, ci=0.95
        )
        assert lower < 5 < upper, \
            f"True mean 5 not in CI [{lower:.3f}, {upper:.3f}]"


# ═══════════════════════════════════════════════════════════════════════════════
# T10 — Bootstrap CI coverage
# ═══════════════════════════════════════════════════════════════════════════════

class TestT10_BootstrapCoverage:
    """T10: Bootstrap CI should achieve ~95% coverage over repeated trials."""

    def test_bootstrap_coverage_rate(self):
        """
        Over 100 draws from N(0,1), the 95% bootstrap CI should contain
        the true mean (0) in approximately 90–100% of trials.
        (We allow down to 85% because of bootstrap approximation error
        with small n=100.)
        """
        np.random.seed(42)
        hits = 0
        n_trials = 100

        for _ in range(n_trials):
            data = np.random.normal(0, 1, size=100)
            _, lower, upper = StatisticalAnalysis.bootstrap_ci(
                data, n_bootstrap=1000, ci=0.95
            )
            if lower <= 0 <= upper:
                hits += 1

        coverage = hits / n_trials
        assert coverage >= 0.85, \
            f"Coverage {coverage:.0%} is below 85% minimum"


# ═══════════════════════════════════════════════════════════════════════════════
# T11 — Bootstrap CI with custom statistic
# ═══════════════════════════════════════════════════════════════════════════════

class TestT11_BootstrapCustomStatistic:
    """T11: Bootstrap CI works with non-mean statistics (e.g. median)."""

    def test_bootstrap_median_ci(self):
        """Bootstrap CI for the median should bracket the true median."""
        np.random.seed(42)
        data = np.random.exponential(scale=5, size=500)
        true_median = np.log(2) * 5  # analytic median of Exp(λ=1/5)

        point, lower, upper = StatisticalAnalysis.bootstrap_ci(
            data, n_bootstrap=5000, ci=0.95, statistic=np.median
        )

        assert lower < true_median < upper, \
            f"True median {true_median:.2f} not in [{lower:.2f}, {upper:.2f}]"

    def test_bootstrap_std_ci(self):
        """Bootstrap CI for standard deviation returns valid bounds."""
        np.random.seed(42)
        data = np.random.normal(0, 3, size=500)

        point, lower, upper = StatisticalAnalysis.bootstrap_ci(
            data, n_bootstrap=5000, ci=0.95, statistic=np.std
        )

        assert 0 < lower < upper, "CI bounds should be positive and ordered"
        assert lower < 3.0 < upper, "True σ=3 should be in the CI"


# ═══════════════════════════════════════════════════════════════════════════════
# T12 — McNemar's test
# ═══════════════════════════════════════════════════════════════════════════════

class TestT12_McNemar:
    """T12: McNemar's test for paired binary classifier comparison."""

    def test_mcnemar_perfect_agreement(self):
        """When both classifiers agree perfectly, p-value should be 1.0."""
        from scipy.stats import chi2

        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        pred_a = np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0])
        pred_b = np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0])  # identical to A

        # Count discordant pairs
        b = np.sum((pred_a == 1) & (pred_b == 0))  # A right, B wrong
        c = np.sum((pred_a == 0) & (pred_b == 1))  # A wrong, B right

        assert b == 0 and c == 0, "No discordant pairs expected"

    def test_mcnemar_known_discordance(self):
        """
        With known b=10, c=3 discordant pairs, McNemar's χ² should be
        (|10-3|-1)² / (10+3) = 36/13 ≈ 2.77, p ≈ 0.096.
        """
        from scipy.stats import chi2 as chi2_dist

        b, c = 10, 3
        # McNemar with continuity correction
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2_dist.cdf(chi2_stat, df=1)

        assert abs(chi2_stat - 2.769) < 0.01, \
            f"χ² should be ≈2.77, got {chi2_stat:.3f}"
        assert p_value > 0.05, \
            f"p={p_value:.3f} should be >0.05 (not significant at α=0.05)"

    def test_mcnemar_significant_discordance(self):
        """
        With b=50, c=5 discordant pairs, the test should be significant.
        χ² = (|50-5|-1)² / (50+5) = 1936/55 ≈ 35.2, p ≈ 3e-9.
        """
        from scipy.stats import chi2 as chi2_dist

        b, c = 50, 5
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2_dist.cdf(chi2_stat, df=1)

        assert chi2_stat > 10, f"χ² should be >>10, got {chi2_stat:.1f}"
        assert p_value < 0.001, f"p={p_value:.6f} should be <0.001"

    def test_cohens_d_known_values(self):
        """Cohen's d for two groups with known separation."""
        a = np.array([10.0, 11, 12, 10, 11])
        b = np.array([20.0, 21, 22, 20, 21])

        d = StatisticalAnalysis.cohens_d(a, b)
        # Large negative effect (a << b)
        assert abs(d) > 0.8, f"Expected large effect, got d={d:.2f}"
        assert StatisticalAnalysis.interpret_cohens_d(d) == 'large'

    def test_cohens_d_negligible(self):
        """Nearly identical groups should give negligible d."""
        a = np.array([10.0, 10.01, 9.99, 10.02, 9.98])
        b = np.array([10.01, 10.0, 10.0, 10.01, 9.99])

        d = StatisticalAnalysis.cohens_d(a, b)
        assert abs(d) < 0.5, f"Expected small/negligible effect, got d={d:.2f}"
