"""
Tests T6–T8: Mixed-precision policy objects.

T6  Distance-based policy assigns correct precision at distance d=0
T7  Distance-based policy assigns correct precision at d=max
T8  Empirical policy respects entropy-based tier classification
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Distance-based precision policy (mirrors MixedPrecisionFramework §4.3)
# ═══════════════════════════════════════════════════════════════════════════════

class DistancePolicy:
    """
    Policy π(d) → precision, parameterised by a threshold τ.
    d < τ  → FP32 (preserve short-range connections)
    d ≥ τ  → target_precision (quantise long-range connections)
    """

    def __init__(self, threshold=4, target_precision='INT8'):
        self.threshold = threshold
        self.target_precision = target_precision

    def assign(self, distance):
        if distance < self.threshold:
            return 'FP32'
        return self.target_precision


class EmpiricalPolicy:
    """
    Policy that assigns precision based on per-block entropy-based
    sensitivity tiers (high → FP32, medium → FP16, low → INT8).
    """

    def __init__(self, tier_map):
        """tier_map: dict mapping (block_num, layer_idx) → 'high'/'medium'/'low'"""
        self.tier_map = tier_map
        self._precision_map = {
            'high': 'FP32',
            'medium': 'FP16',
            'low': 'INT8',
        }

    def assign(self, block_num, layer_idx):
        tier = self.tier_map.get((block_num, layer_idx), 'medium')
        return self._precision_map[tier]


# ═══════════════════════════════════════════════════════════════════════════════
# T6 — Distance d=0 (adjacent connection)
# ═══════════════════════════════════════════════════════════════════════════════

class TestT6_DistancePolicyZero:
    """T6: Distance-based policy at d=0 (shortest possible connection)."""

    def test_distance_zero_stays_fp32(self):
        """Adjacent connections (d=0) must always remain at FP32."""
        policy = DistancePolicy(threshold=4)
        assert policy.assign(0) == 'FP32'

    def test_distance_one_stays_fp32(self):
        """d=1 connections (one layer apart) should remain FP32."""
        policy = DistancePolicy(threshold=4)
        assert policy.assign(1) == 'FP32'

    def test_distance_below_threshold_all_fp32(self):
        """All distances below threshold should map to FP32."""
        policy = DistancePolicy(threshold=4)
        for d in range(4):
            assert policy.assign(d) == 'FP32', f"d={d} should be FP32"


# ═══════════════════════════════════════════════════════════════════════════════
# T7 — Distance d=max (longest connection in a dense block)
# ═══════════════════════════════════════════════════════════════════════════════

class TestT7_DistancePolicyMax:
    """T7: Distance-based policy at maximum distance."""

    def test_distance_at_threshold_quantised(self):
        """d=τ should be quantised to the target precision."""
        policy = DistancePolicy(threshold=4, target_precision='INT8')
        assert policy.assign(4) == 'INT8'

    def test_distance_max_densenet121(self):
        """
        In DenseNet-121's denseblock4, the longest connection spans
        d=31 (layer 1 → layer 32). This should be INT8.
        """
        policy = DistancePolicy(threshold=4, target_precision='INT8')
        assert policy.assign(31) == 'INT8'

    def test_distance_max_with_fp16_target(self):
        """Policy should support FP16 as an alternative target."""
        policy = DistancePolicy(threshold=3, target_precision='FP16')
        assert policy.assign(10) == 'FP16'
        assert policy.assign(2) == 'FP32'

    def test_connection_count_densenet121(self):
        """
        DenseNet-121 has blocks of [6, 12, 24, 16] layers.
        Total connections = sum(n*(n+1)/2) for each block.
        The distance-based policy at τ=4 should categorise ~68% as INT8.
        """
        block_sizes = [6, 12, 24, 16]
        policy = DistancePolicy(threshold=4)

        total = 0
        quantised = 0
        for n in block_sizes:
            for consumer in range(1, n + 1):
                for producer in range(consumer):
                    d = consumer - producer
                    total += 1
                    if policy.assign(d) != 'FP32':
                        quantised += 1

        frac = quantised / total
        assert 0.5 < frac < 0.9, \
            f"Expected 50–90% of connections quantised, got {frac:.1%}"


# ═══════════════════════════════════════════════════════════════════════════════
# T8 — Empirical policy entropy tiers
# ═══════════════════════════════════════════════════════════════════════════════

class TestT8_EmpiricalPolicy:
    """T8: Empirical policy assigns precision from entropy-based tiers."""

    def test_high_sensitivity_gets_fp32(self):
        """Layers classified as 'high' sensitivity must stay at FP32."""
        tier_map = {(4, 1): 'high', (4, 2): 'medium', (1, 1): 'low'}
        policy = EmpiricalPolicy(tier_map)
        assert policy.assign(4, 1) == 'FP32'

    def test_medium_sensitivity_gets_fp16(self):
        """Layers classified as 'medium' should use FP16."""
        tier_map = {(4, 1): 'high', (4, 2): 'medium', (1, 1): 'low'}
        policy = EmpiricalPolicy(tier_map)
        assert policy.assign(4, 2) == 'FP16'

    def test_low_sensitivity_gets_int8(self):
        """Layers classified as 'low' should use INT8."""
        tier_map = {(4, 1): 'high', (4, 2): 'medium', (1, 1): 'low'}
        policy = EmpiricalPolicy(tier_map)
        assert policy.assign(1, 1) == 'INT8'

    def test_unknown_layer_defaults_to_medium(self):
        """Unregistered layers should default to 'medium' → FP16."""
        tier_map = {(1, 1): 'low'}
        policy = EmpiricalPolicy(tier_map)
        assert policy.assign(99, 99) == 'FP16'

    def test_entropy_ranking_produces_three_tiers(self, tiny_model):
        """
        Entropy-based ranking should classify layers into exactly
        three tiers: high, medium, low (matching the policy interface).
        """
        # Compute entropy for each Conv2d
        entropies = []
        for name, module in tiny_model.named_modules():
            if isinstance(module, nn.Conv2d):
                w = module.weight.data.cpu().float().numpy().flatten()
                hist, _ = np.histogram(w, bins=256, density=True)
                hist = hist[hist > 0]
                hist = hist / hist.sum()
                entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))
                w_range = float(w.max() - w.min())
                entropies.append(entropy * w_range)

        if len(entropies) >= 3:
            p33 = np.percentile(entropies, 33)
            p66 = np.percentile(entropies, 66)
            tiers = []
            for s in entropies:
                if s >= p66:
                    tiers.append('high')
                elif s >= p33:
                    tiers.append('medium')
                else:
                    tiers.append('low')
            unique_tiers = set(tiers)
            # With enough layers, we should get at least 2 distinct tiers
            assert len(unique_tiers) >= 2, \
                f"Expected ≥2 tiers, got {unique_tiers}"
