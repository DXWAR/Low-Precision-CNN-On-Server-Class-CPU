"""
================================================================================
Dissertation : Low-Precision CNNs on Server-Class CPUs
DenseNet-121 Chest X-ray Diagnosis — Complete Experiment Suite
================================================================================

Hardware target: AMD EPYC 9005 (Zen 5) on Azure Standard_F4s_v2 (4 vCPUs)
  - AVX-512 VNNI:  INT8 multiply-accumulate via VPDPBUSD instruction
  - AVX-512 BF16:  native BF16 dot-product via VDPBF16PS instruction
  - No hardware FP16: x86 lacks FP16 compute; autocast(float16) falls back
                       to software emulation (~40x slower)
  - Backend: PyTorch fbgemm (works on both Intel and AMD AVX-512 CPUs;
             fbgemm dispatches to the same VNNI/BF16 ISA on both vendors)

Note on Intel vs AMD: Much of the quantisation literature and PyTorch
documentation references Intel-specific tools (Intel Extension for PyTorch,
oneDNN, Intel VNNI). However, AVX-512 VNNI and AVX-512 BF16 are x86 ISA
extensions implemented by BOTH Intel (Cascade Lake+, Cooper Lake+) and
AMD (Zen 4+). The fbgemm backend used here dispatches to the same AVX-512
instructions regardless of vendor — the key requirement is ISA support,
not vendor-specific runtime libraries.

This script runs the full experimental pipeline for the dissertation:

  1. Precision spectrum comparison   (FP64 → FP32 → BF16 → FP16 → INT8)
  2. True dynamic quantisation       (PyTorch quantisation APIs)
  3. Per-channel vs per-tensor INT8   (quantisation granularity comparison)
  4. Block sensitivity analysis       (which dense blocks tolerate low precision)
  5. Layer sensitivity analysis       (per-layer sensitivity + type breakdown)
  6. Clinical accuracy evaluation     (per-pathology probability differences)
  7. McNemar's test                   (binary classification equivalence)
  8. Mixed-precision forward pass     (distance-based + entropy strategies)
  9. Statistical analysis             (CIs, p-values, effect sizes)
 10. Native BF16 / FP16 inference     (torch.cpu.amp.autocast — real compute)
 11. Static INT8 quantisation         (torch.quantization prepare/convert — VNNI)
 12. Energy & carbon footprint        (TDP-based or RAPL measurement)
 13. Publication-quality figures      (error bars, heatmaps, energy plot)

Run:
    python newmain.py                   # Full pipeline (default 1000 images)
    python newmain.py --quick           # Quick run (100 images, fewer layers)
    python newmain.py --images 500      # Custom image count

Requirements:
    pip install torch torchvision torchxrayvision numpy matplotlib Pillow scipy scikit-learn tqdm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv
import numpy as np
import time
import json
import copy
import os
import glob
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import platform
import re
import tempfile
import warnings
warnings.filterwarnings('ignore')

try:
    import csv as csv_module
except ImportError:
    pass

try:
    from PIL import Image
    import torchvision.transforms as transforms
except ImportError:
    print("Need Pillow + torchvision: pip install Pillow torchvision")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found — statistical tests will be skipped")

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not found — AUC metrics will be skipped")

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# FX TRACING SETUP — must be at module scope
# ═══════════════════════════════════════════════════════════════════════════════
# torch.fx.wrap() registers Python builtins so FX symbolic tracing can handle
# them. torchxrayvision's DenseNet forward() uses len() etc., and FX chokes
# unless they're wrapped. This MUST be at module top-level (not inside a
# function) — PyTorch enforces this to ensure the wrapper is visible during
# tracing.
for _builtin in ['len', 'range', 'int', 'float', 'list', 'tuple']:
    try:
        torch.fx.wrap(_builtin)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Search common locations for ChestX-ray14 images
_CANDIDATE_PATHS = [
    os.path.expanduser("~/Documents/dissation Project/images"),
    os.path.expanduser("~/images"),
    os.path.expanduser("~/data/images"),
    os.path.expanduser("~/chest-xray/images"),
    os.path.expanduser("~/ChestXray-NIHCC/images"),
    './images',
]
DATA_PATH = next((p for p in _CANDIDATE_PATHS if os.path.exists(p)), _CANDIDATE_PATHS[0])

# ChestX-ray14 pathology labels
PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
    'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
    'Pneumonia', 'Pneumothorax'
]

# Safety-critical conditions where a missed diagnosis is dangerous
CRITICAL_PATHOLOGIES = ['Pneumothorax', 'Cardiomegaly', 'Pneumonia']

# Clinical safety threshold: max acceptable probability difference
SAFETY_THRESHOLD_PERCENT = 1.0

# Colour palette for figures
COLOURS = {
    'FP64': '#1a5276', 'FP32': '#2E86AB', 'BF16': '#8E44AD',
    'FP16': '#A23B72', 'INT8': '#F18F01',
    'INT8_PerChannel': '#E74C3C', 'Dynamic_INT8': '#27AE60',
    'Dynamic INT8': '#27AE60', 'Per-Channel INT8': '#E74C3C',
    'Mixed (Distance)': '#3498DB', 'Mixed (Empirical)': '#2ECC71',
    'Native_BF16': '#9B59B6', 'Native_FP16': '#E91E63',
    'Static_INT8': '#FF5722',
}


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class ImageLoader:
    """
    Loads chest X-ray images from the dataset folders.

    Rationale: We need real medical images to properly test quantisation.
    Random tensors don't have the same statistical properties as actual
    X-rays, so results from synthetic data aren't clinically meaningful.

    The ChestX-ray14 images are 1024x1024 PNG files. We resize to 224x224
    because that's what DenseNet-121 expects (it was trained on this size).

    Patient deduplication: ChestX-ray14 filenames follow the pattern
    NNNNN_NNN.png where the first 5 digits are the patient ID. We keep
    only one image per patient to prevent patient-level data leakage.
    """

    def __init__(self, base_path, max_images=1000, one_per_patient=True):
        self.base_path = base_path
        self.max_images = max_images
        self.one_per_patient = one_per_patient
        self.image_paths = []
        self.patient_ids = []  # track which patient each image belongs to

        # Standard preprocessing for chest X-rays
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self._find_images()

    @staticmethod
    def _extract_patient_id(filepath):
        """
        Extract patient ID from ChestX-ray14 filename.
        Format: NNNNN_NNN.png → patient ID is NNNNN (first 5 digits).
        Returns None if filename doesn't match expected pattern.
        """
        basename = os.path.basename(filepath)
        match = re.match(r'^(\d{5})_\d{3}\.png$', basename, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _find_images(self):
        """
        Searches for PNG/JPG images in the data folder and subfolders.
        ChestX-ray14 is split across multiple folders (images_001 to images_012).

        When one_per_patient=True, keeps only one image per patient ID to
        avoid patient-level leakage in evaluation.
        """
        if not os.path.exists(self.base_path):
            if os.path.exists('./images'):
                self.base_path = './images'
            else:
                print(f"  Warning: {self.base_path} not found")
                return

        for ext in ['*.png', '*.jpg', '*.PNG', '*.JPG']:
            pattern = os.path.join(self.base_path, '**', ext)
            self.image_paths.extend(glob.glob(pattern, recursive=True))

        self.image_paths = sorted(set(self.image_paths))
        total_before = len(self.image_paths)

        # Patient deduplication
        if self.one_per_patient:
            seen_patients = set()
            deduplicated = []
            for path in self.image_paths:
                pid = self._extract_patient_id(path)
                if pid is not None:
                    if pid not in seen_patients:
                        seen_patients.add(pid)
                        deduplicated.append(path)
                else:
                    # Non-standard filename: keep it
                    deduplicated.append(path)

            removed = total_before - len(deduplicated)
            if removed > 0:
                print(f"  Patient deduplication: {total_before} → {len(deduplicated)} "
                      f"(removed {removed} duplicate-patient images)")
            self.image_paths = deduplicated

        if len(self.image_paths) > self.max_images:
            np.random.seed(42)
            indices = np.random.choice(len(self.image_paths), self.max_images, replace=False)
            self.image_paths = [self.image_paths[i] for i in sorted(indices)]

        # Record patient IDs for the final selection
        self.patient_ids = [self._extract_patient_id(p) or 'unknown' for p in self.image_paths]
        n_unique = len(set(pid for pid in self.patient_ids if pid != 'unknown'))

        print(f"  Found {len(self.image_paths)} images ({n_unique} unique patients)")

    def load_image(self, path):
        """Load a single image and convert to tensor."""
        try:
            img = Image.open(path).convert('L')
            tensor = self.transform(img)
            return tensor.unsqueeze(0)  # [1, 1, 224, 224]
        except Exception:
            return None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.load_image(self.image_paths[idx])


# ═══════════════════════════════════════════════════════════════════════════════
# GROUND-TRUTH LABEL LOADER (ChestX-ray14 CSV)
# ═══════════════════════════════════════════════════════════════════════════════

class GroundTruthLoader:
    """
    Load ground-truth labels from the ChestX-ray14 Data_Entry CSV.

    The CSV has columns: Image Index, Finding Labels, Follow-up #,
    Patient ID, Patient Age, Patient Gender, View Position, etc.

    Finding Labels is a pipe-delimited string, e.g. "Atelectasis|Effusion"
    or "No Finding" for healthy cases.

    Returns a binary labels matrix (N, C) aligned with the image filenames
    used in the experiment, where C = len(PATHOLOGIES).
    """

    # Common filenames for the ChestX-ray14 ground-truth CSV
    _CSV_NAMES = [
        'Data_Entry_2017_v2020.csv',
        'Data_Entry_2017.csv',
        'Data_Entry.csv',
    ]

    @staticmethod
    def find_csv(data_path):
        """
        Search for the ground-truth CSV in common locations:
        - Same directory as images
        - Parent directory of images
        - Sibling directories
        """
        search_dirs = [
            data_path,
            os.path.dirname(data_path),
            os.path.join(os.path.dirname(data_path), '..'),
        ]
        # Also search common dataset root patterns
        for parent in [os.path.expanduser('~'), os.path.expanduser('~/data'),
                       os.path.expanduser('~/Documents')]:
            search_dirs.append(parent)
            for d in ['ChestXray-NIHCC', 'chest-xray', 'chestxray14',
                      'dissation Project']:
                search_dirs.append(os.path.join(parent, d))

        for search_dir in search_dirs:
            for csv_name in GroundTruthLoader._CSV_NAMES:
                csv_path = os.path.join(search_dir, csv_name)
                if os.path.isfile(csv_path):
                    return csv_path
        return None

    @staticmethod
    def load_labels(csv_path, image_filenames, pathology_names=None):
        """
        Parse the CSV and build a binary labels matrix.

        Args:
            csv_path: path to Data_Entry CSV
            image_filenames: list of image filenames (basename only,
                             e.g. ['00000001_000.png', ...]) in the same
                             order as the experiment's test_images
            pathology_names: list of pathology names to use as columns
                             (default: PATHOLOGIES global)

        Returns:
            labels_matrix: np.ndarray of shape (N, C) with 0/1 values,
                           or None if loading fails
        """
        if pathology_names is None:
            pathology_names = PATHOLOGIES

        # Build filename → row index lookup from the CSV
        csv_labels = {}  # filename -> set of findings
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    fname = row.get('Image Index', '').strip()
                    findings = row.get('Finding Labels', '').strip()
                    if fname:
                        csv_labels[fname] = set(findings.split('|'))
        except Exception as e:
            print(f"  ERROR reading CSV: {e}")
            return None

        if not csv_labels:
            print(f"  ERROR: CSV is empty or unreadable")
            return None

        print(f"  CSV entries: {len(csv_labels)}")

        # Build the labels matrix aligned with the experiment images
        n_images = len(image_filenames)
        n_pathologies = len(pathology_names)
        labels = np.zeros((n_images, n_pathologies), dtype=np.float32)
        matched = 0
        unmatched = []

        for i, fname in enumerate(image_filenames):
            basename = os.path.basename(fname)
            if basename in csv_labels:
                findings = csv_labels[basename]
                for j, pathology in enumerate(pathology_names):
                    if pathology in findings:
                        labels[i, j] = 1.0
                matched += 1
            else:
                unmatched.append(basename)

        print(f"  Matched: {matched}/{n_images} images to ground-truth labels")
        if unmatched and len(unmatched) <= 5:
            print(f"  Unmatched: {unmatched}")
        elif unmatched:
            print(f"  Unmatched: {len(unmatched)} images (first 3: {unmatched[:3]})")

        if matched == 0:
            print("  WARNING: No images matched CSV — falling back to pseudo-labels")
            return None

        # Report label prevalence (important for understanding AUC results)
        print(f"\n  Label prevalence in {matched} matched images:")
        for j, pathology in enumerate(pathology_names):
            pos = int(labels[:, j].sum())
            prev = pos / n_images * 100
            print(f"    {pathology:<22} {pos:>5} ({prev:>5.1f}%)")

        return labels


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class StatisticalAnalysis:
    """
    Statistical tests and confidence intervals for comparing precision levels.

    Methods:
      - Bootstrap 95% confidence intervals
      - Paired t-test (or Wilcoxon if non-normal)
      - Cohen's d effect size
      - McNemar's test for binary classification agreement
    """

    @staticmethod
    def bootstrap_ci(data, n_bootstrap=10000, ci=0.95, statistic=np.mean):
        """
        Compute bootstrap confidence interval for a statistic.

        Args:
            data: 1D array of observations
            n_bootstrap: number of bootstrap resamples
            ci: confidence level (default 0.95)
            statistic: function to compute (default np.mean)

        Returns:
            (point_estimate, lower_bound, upper_bound)
        """
        data = np.array(data)
        point = float(statistic(data))
        boot_stats = []
        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            sample = rng.choice(data, size=len(data), replace=True)
            boot_stats.append(statistic(sample))
        boot_stats = np.array(boot_stats)
        alpha = 1 - ci
        lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
        return point, lower, upper

    @staticmethod
    def paired_ttest(a, b):
        """
        Paired t-test comparing two matched samples.
        Falls back to Wilcoxon signed-rank if scipy unavailable.

        Returns: (t_statistic, p_value)
        """
        if not HAS_SCIPY:
            return None, None
        a, b = np.array(a), np.array(b)
        t_stat, p_val = scipy_stats.ttest_rel(a, b)
        return float(t_stat), float(p_val)

    @staticmethod
    def wilcoxon_test(a, b):
        """Non-parametric alternative to paired t-test."""
        if not HAS_SCIPY:
            return None, None
        a, b = np.array(a), np.array(b)
        try:
            stat, p_val = scipy_stats.wilcoxon(a - b)
            return float(stat), float(p_val)
        except ValueError:
            return None, None

    @staticmethod
    def cohens_d(a, b):
        """
        Cohen's d effect size for paired samples.
        Small = 0.2, Medium = 0.5, Large = 0.8
        """
        a, b = np.array(a), np.array(b)
        diff = a - b
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        return float(d)

    @staticmethod
    def bootstrap_speedup_ci(fp32_times, other_times, n_bootstrap=10000, ci=0.95):
        """
        Bootstrap confidence interval for the speedup ratio (FP32 / other).

        Instead of just reporting a point estimate like "1.20x", this gives
        a defensible range like "1.20x [1.18–1.22]" which is important for
        peer review.

        Args:
            fp32_times: list of per-image FP32 latencies (ms)
            other_times: list of per-image latencies for comparison format (ms)
            n_bootstrap: number of resamples
            ci: confidence level

        Returns:
            (point_speedup, lower_bound, upper_bound)
        """
        fp32_times = np.array(fp32_times)
        other_times = np.array(other_times)
        point_speedup = float(np.mean(fp32_times) / np.mean(other_times))

        rng = np.random.RandomState(42)
        boot_speedups = []
        n = len(fp32_times)
        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            fp32_mean = np.mean(fp32_times[idx])
            other_mean = np.mean(other_times[idx])
            if other_mean > 0:
                boot_speedups.append(fp32_mean / other_mean)

        boot_speedups = np.array(boot_speedups)
        alpha = 1 - ci
        lower = float(np.percentile(boot_speedups, 100 * alpha / 2))
        upper = float(np.percentile(boot_speedups, 100 * (1 - alpha / 2)))
        return point_speedup, lower, upper

    @staticmethod
    def interpret_cohens_d(d):
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"


# ═══════════════════════════════════════════════════════════════════════════════
# CLINICAL EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class ClinicalMetrics:
    """
    Calibration and discrimination metrics for clinical evaluation.

    Expected Calibration Error (ECE) measures how well predicted probabilities
    match observed frequencies. For medical AI, this matters because a model
    predicting 0.8 probability of Pneumothorax should be correct ~80% of the
    time — miscalibrated models can't be trusted for clinical decisions.

    AUC-ROC measures discrimination: can the model distinguish positive from
    negative cases? We compute per-pathology AUC for each precision level.
    """

    @staticmethod
    def expected_calibration_error(probs, labels, n_bins=15):
        """
        Compute Expected Calibration Error (ECE).

        ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

        where B_b is the set of predictions in bin b,
        acc(B_b) is the accuracy of predictions in that bin,
        conf(B_b) is the mean confidence in that bin.

        Args:
            probs: predicted probabilities (N,)
            labels: binary ground-truth labels (N,)
            n_bins: number of calibration bins

        Returns:
            (ece, bin_accs, bin_confs, bin_counts) for plotting reliability diagrams
        """
        probs = np.array(probs)
        labels = np.array(labels)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_accs = []
        bin_confs = []
        bin_counts = []
        ece = 0.0

        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
            count = mask.sum()
            bin_counts.append(int(count))

            if count > 0:
                acc = labels[mask].mean()
                conf = probs[mask].mean()
                bin_accs.append(float(acc))
                bin_confs.append(float(conf))
                ece += (count / len(probs)) * abs(acc - conf)
            else:
                bin_accs.append(0.0)
                bin_confs.append((lo + hi) / 2)

        return float(ece), bin_accs, bin_confs, bin_counts

    @staticmethod
    def maximum_calibration_error(probs, labels, n_bins=15):
        """
        Compute Maximum Calibration Error (MCE).

        MCE = max_b |acc(B_b) - conf(B_b)|

        Worst-case calibration gap across all bins. Important for safety:
        even if ECE is low, a high MCE means some probability range is
        badly miscalibrated.
        """
        _, bin_accs, bin_confs, bin_counts = ClinicalMetrics.expected_calibration_error(
            probs, labels, n_bins
        )
        mce = 0.0
        for acc, conf, count in zip(bin_accs, bin_confs, bin_counts):
            if count > 0:
                mce = max(mce, abs(acc - conf))
        return float(mce)

    @staticmethod
    def compute_auc_per_pathology(probs_matrix, labels_matrix, pathology_names):
        """
        Compute AUC-ROC per pathology.

        Args:
            probs_matrix: (N, C) predicted probabilities
            labels_matrix: (N, C) binary ground-truth labels
            pathology_names: list of C pathology names

        Returns:
            dict mapping pathology name to AUC (or None if not computable)
        """
        if not HAS_SKLEARN:
            return {}

        aucs = {}
        for i, name in enumerate(pathology_names):
            if i >= probs_matrix.shape[1] or i >= labels_matrix.shape[1]:
                continue
            y_true = labels_matrix[:, i]
            y_score = probs_matrix[:, i]

            # AUC needs both classes present
            unique = np.unique(y_true)
            if len(unique) < 2:
                aucs[name] = None
                continue

            try:
                auc = roc_auc_score(y_true, y_score)
                aucs[name] = float(auc)
            except Exception:
                aucs[name] = None

        return aucs

    @staticmethod
    def bootstrap_auc_ci(y_true, y_score, n_bootstrap=5000, ci=0.95):
        """
        Bootstrap confidence interval for AUC-ROC.

        Returns:
            (point_auc, lower, upper) or (None, None, None) if not computable
        """
        if not HAS_SKLEARN:
            return None, None, None

        y_true = np.array(y_true)
        y_score = np.array(y_score)

        if len(np.unique(y_true)) < 2:
            return None, None, None

        try:
            point_auc = roc_auc_score(y_true, y_score)
        except Exception:
            return None, None, None

        rng = np.random.RandomState(42)
        boot_aucs = []
        for _ in range(n_bootstrap):
            idx = rng.choice(len(y_true), size=len(y_true), replace=True)
            if len(np.unique(y_true[idx])) < 2:
                continue
            try:
                boot_aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
            except Exception:
                continue

        if len(boot_aucs) < 100:
            return float(point_auc), None, None

        alpha = 1 - ci
        lower = float(np.percentile(boot_aucs, 100 * alpha / 2))
        upper = float(np.percentile(boot_aucs, 100 * (1 - alpha / 2)))
        return float(point_auc), lower, upper

    @staticmethod
    def bootstrap_prob_diff_ci(fp32_probs, other_probs, n_bootstrap=5000, ci=0.95):
        """
        Bootstrap CI for mean absolute probability difference between
        FP32 and another precision.

        Returns:
            (mean_diff, lower, upper)
        """
        diffs = np.abs(np.array(fp32_probs) - np.array(other_probs))
        mean_diff = float(np.mean(diffs))

        rng = np.random.RandomState(42)
        boot_means = []
        for _ in range(n_bootstrap):
            idx = rng.choice(len(diffs), size=len(diffs), replace=True)
            boot_means.append(np.mean(diffs[idx]))

        alpha = 1 - ci
        lower = float(np.percentile(boot_means, 100 * alpha / 2))
        upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        return mean_diff, lower, upper


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTISATION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class QuantWrapper(nn.Module):
    """
    Wraps a model with QuantStub/DeQuantStub so that static quantisation
    works correctly.  Without these stubs the converted model's quantized
    Conv2d layers receive plain float tensors and raise
    NotImplementedError: 'quantized::conv2d.new' … 'CPU' backend.

    QuantStub  converts float  → quantized tensor  at model entry.
    DeQuantStub converts quantized → float tensor   at model exit.

    After prepare() + convert(), QuantStub becomes a real quantize op
    that maps float32 → quint8, and DeQuantStub maps quint8 → float32.
    """

    def __init__(self, model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


class DenseNetQuantWrapper(nn.Module):
    """
    Quantisation-aware wrapper for torchxrayvision's DenseNet.

    Unlike the generic QuantWrapper, this splits the model so that:
      - QuantStub → features_body (conv0…denseblock4) run in INT8 (QUInt8)
      - DeQuantStub converts back to float before norm5
      - norm5, F.relu, adaptive_avg_pool2d, classifier run in float

    This avoids the type-promotion crash that occurs when standalone
    BatchNorm2d (norm5) receives a QUInt8 tensor, and also bypasses
    torchxrayvision's utility calls (fix_resolution, warn_normalization)
    which don't handle quantised tensors.
    """

    def __init__(self, model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()

        # Everything before norm5 — these layers will be quantised to INT8
        self.features_body = nn.Sequential()
        for name, module in model.features.named_children():
            if name == 'norm5':
                break
            self.features_body.add_module(name, module)

        self.dequant = torch.quantization.DeQuantStub()

        # Tail runs in float — negligible compute here
        self.norm5 = model.features.norm5
        self.classifier = model.classifier

    def forward(self, x):
        x = self.quant(x)                                     # float → QUInt8
        x = self.features_body(x)                              # QUInt8 throughout
        x = self.dequant(x)                                    # QUInt8 → float
        x = self.norm5(x)                                      # float
        x = F.relu(x)                                          # float
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)  # float
        x = self.classifier(x)                                 # float
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# CORE PRECISION EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

class PrecisionExperiment:
    """
    Main experiment class for testing different numerical precisions.

    Tests include:
      - Simulated precision (convert weights, run in FP32)
      - Dynamic quantisation (PyTorch quantisation APIs)
      - Per-channel vs per-tensor INT8
      - Native BF16/FP16 via torch.cpu.amp.autocast
      - Static INT8 via torch.quantization prepare/convert pipeline

    All accuracy comparisons use FP32 as the baseline since that is
    the precision the model was trained in.
    """

    def __init__(self, data_path=DATA_PATH, max_images=1000):
        self.data_path = data_path
        self.max_images = max_images
        self.results = {}

        # Force fbgemm for x86/AMD EPYC — uses AVX-512 VNNI for INT8
        try:
            torch.backends.quantized.engine = 'fbgemm'
            self.backend = 'fbgemm'
        except Exception:
            torch.backends.quantized.engine = 'qnnpack'
            self.backend = 'qnnpack'

    def load_model_and_data(self):
        """Load the pre-trained DenseNet-121 and test images."""
        print("\nLoading model...")
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model.eval()

        # Build index mapping: model output columns → PATHOLOGIES order
        # torchxrayvision outputs 18 pathologies in its own order (not alphabetical).
        # We need to know which model output column corresponds to each PATHOLOGIES entry.
        model_pathology_list = list(self.model.pathologies)
        self.pathology_col_map = {}  # PATHOLOGIES index → model output column index
        for i, name in enumerate(PATHOLOGIES):
            if name in model_pathology_list:
                self.pathology_col_map[i] = model_pathology_list.index(name)
            else:
                print(f"  WARNING: {name} not in model output — will skip")
        print(f"  Model pathology order: {model_pathology_list[:len(PATHOLOGIES)]}")
        print(f"  Mapped {len(self.pathology_col_map)}/14 ChestX-ray14 labels to model columns")

        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        self.model_size_mb = param_bytes / (1024 * 1024)
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {num_params:,}")
        print(f"  Size: {self.model_size_mb:.1f} MB")

        print(f"\nLoading images from {self.data_path}...")
        loader = ImageLoader(self.data_path, self.max_images)

        self.test_images = []
        self.image_filenames = []  # track filenames for ground-truth matching
        for i in tqdm(range(len(loader)), desc="  Loading"):
            img = loader[i]
            if img is not None:
                self.test_images.append(img)
                self.image_filenames.append(os.path.basename(loader.image_paths[i]))

        print(f"  Loaded: {len(self.test_images)} images")
        self.using_real_images = len(self.test_images) > 0

        if not self.using_real_images:
            print("  No images found, using synthetic data")
            torch.manual_seed(42)
            self.test_images = [torch.randn(1, 1, 224, 224) for _ in range(50)]
            self.image_filenames = [f"synthetic_{i:04d}.png" for i in range(50)]

        # Load ground-truth labels from ChestX-ray14 CSV
        self.ground_truth_labels = None
        if self.using_real_images:
            csv_path = GroundTruthLoader.find_csv(self.data_path)
            if csv_path:
                print(f"\n  Found ground-truth CSV: {csv_path}")
                self.ground_truth_labels = GroundTruthLoader.load_labels(
                    csv_path, self.image_filenames, PATHOLOGIES
                )
                if self.ground_truth_labels is not None:
                    print(f"  Ground-truth labels loaded: {self.ground_truth_labels.shape}")
                else:
                    print("  WARNING: Ground-truth loading failed — AUC will use pseudo-labels")
            else:
                print("  No ground-truth CSV found — AUC will use pseudo-labels")
                print("  To enable ground-truth AUC, place Data_Entry_2017_v2020.csv")
                print("  in the same directory as the images or its parent directory.")

    def _reindex_probs(self, probs):
        """
        Reindex model output probabilities from model column order to
        PATHOLOGIES order using self.pathology_col_map.

        Args:
            probs: np.ndarray of shape (N, 18) — sigmoid probabilities in model order

        Returns:
            np.ndarray of shape (N, 14) — probabilities aligned with PATHOLOGIES
        """
        n = probs.shape[0]
        reindexed = np.zeros((n, len(PATHOLOGIES)), dtype=probs.dtype)
        for pathology_idx, model_col in self.pathology_col_map.items():
            reindexed[:, pathology_idx] = probs[:, model_col]
        return reindexed

    def report_model_sizes(self):
        """
        Report effective model sizes per precision (theoretical) and
        measure actual on-disk sizes for saved state_dicts.

        Theoretical sizes are computed from parameter count × bytes per element.
        On-disk sizes reflect what you'd actually store for deployment.
        """
        print("\n" + "=" * 60)
        print("MODEL SIZE ANALYSIS")
        print("=" * 60)

        num_params = sum(p.numel() for p in self.model.parameters())

        # Theoretical sizes
        bytes_per_param = {
            'FP64': 8, 'FP32': 4, 'BF16': 2, 'FP16': 2, 'INT8': 1
        }
        print(f"\n  Parameters: {num_params:,}")
        print(f"\n  Theoretical model sizes:")
        theoretical = {}
        for prec, bpp in bytes_per_param.items():
            size_mb = (num_params * bpp) / (1024 * 1024)
            theoretical[prec] = size_mb
            print(f"    {prec:<6}  {size_mb:>8.2f} MB  ({bpp} bytes/param)")

        # On-disk sizes for actual saved models
        print(f"\n  On-disk sizes (saved state_dicts):")
        disk_sizes = {}

        # FP32 baseline
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=True) as f:
            torch.save(self.model.state_dict(), f.name)
            size_mb = os.path.getsize(f.name) / (1024 * 1024)
            disk_sizes['FP32'] = size_mb
            print(f"    FP32 (native):     {size_mb:.2f} MB")

        # Dynamic INT8
        try:
            dyn_model = torch.quantization.quantize_dynamic(
                copy.deepcopy(self.model).float().eval(),
                {nn.Linear}, dtype=torch.qint8
            )
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=True) as f:
                torch.save(dyn_model.state_dict(), f.name)
                size_mb = os.path.getsize(f.name) / (1024 * 1024)
                disk_sizes['Dynamic_INT8'] = size_mb
                print(f"    Dynamic INT8:      {size_mb:.2f} MB")
        except Exception as e:
            print(f"    Dynamic INT8:      failed ({e})")

        # BF16 weights
        bf16_model = copy.deepcopy(self.model).float().eval()
        for p in bf16_model.parameters():
            p.data = p.data.to(torch.bfloat16)
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=True) as f:
            torch.save(bf16_model.state_dict(), f.name)
            size_mb = os.path.getsize(f.name) / (1024 * 1024)
            disk_sizes['BF16'] = size_mb
            print(f"    BF16 weights:      {size_mb:.2f} MB")

        # FP16 weights
        fp16_model = copy.deepcopy(self.model).float().eval()
        for p in fp16_model.parameters():
            p.data = p.data.half()
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=True) as f:
            torch.save(fp16_model.state_dict(), f.name)
            size_mb = os.path.getsize(f.name) / (1024 * 1024)
            disk_sizes['FP16'] = size_mb
            print(f"    FP16 weights:      {size_mb:.2f} MB")

        self.results['model_sizes'] = {
            'num_parameters': num_params,
            'theoretical_mb': theoretical,
            'on_disk_mb': disk_sizes
        }
        return self.results['model_sizes']

    def benchmark_model(self, model, name, num_warmup=10, autocast_dtype=None):
        """
        Measure inference speed and collect outputs.

        Warmup runs are important because the first few inferences are
        slower (CPU caches aren't populated, JIT compilation, etc).

        Args:
            model: the model to benchmark
            name: label for printing
            num_warmup: number of warmup inferences
            autocast_dtype: if set, wrap inference in torch.cpu.amp.autocast
                            with this dtype (e.g. torch.bfloat16, torch.float16)
        """
        with torch.no_grad():
            for img in self.test_images[:num_warmup]:
                if autocast_dtype is not None:
                    with torch.amp.autocast('cpu', dtype=autocast_dtype):
                        _ = model(img.float())
                else:
                    _ = model(img.float())

        times = []
        outputs = []

        with torch.no_grad():
            for img in self.test_images:
                if autocast_dtype is not None:
                    start = time.perf_counter()
                    with torch.amp.autocast('cpu', dtype=autocast_dtype):
                        out = model(img.float())
                    elapsed = (time.perf_counter() - start) * 1000
                else:
                    start = time.perf_counter()
                    out = model(img.float())
                    elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                outputs.append(out.float().cpu())

        all_outputs = torch.cat(outputs, dim=0)
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = 1000.0 / avg_time if avg_time > 0 else 0  # images/second

        # Bootstrap CI for timing
        mean_est, ci_lo, ci_hi = StatisticalAnalysis.bootstrap_ci(times)

        print(f"  {name}: {avg_time:.1f} +/- {std_time:.1f} ms  "
              f"[95% CI: {ci_lo:.1f} - {ci_hi:.1f}]  "
              f"throughput: {throughput:.1f} img/s")

        return {
            'avg_ms': avg_time,
            'std_ms': std_time,
            'ci_lower': ci_lo,
            'ci_upper': ci_hi,
            'throughput_ips': throughput,
            'times': times,
            'outputs': all_outputs
        }

    # ── Simulated precision tests ────────────────────────────────────────

    def test_fp64(self):
        """
        Test FP64 (64-bit / double precision) — theoretical comparison.

        The model was trained in FP32, so converting to FP64 adds no real
        precision. We run this in FP32 and report the theoretical FP64 memory
        cost to demonstrate that double precision is wasteful.
        """
        model = copy.deepcopy(self.model).float()
        model.eval()
        result = self.benchmark_model(model, "FP64 (theoretical)", num_warmup=30)
        result['note'] = ("Model trained in FP32 so FP64 adds no accuracy. "
                          "Theoretical size would be 2x FP32.")
        return result

    def test_fp32(self):
        """FP32 baseline — the model's native training precision."""
        model = copy.deepcopy(self.model).float()
        model.eval()
        return self.benchmark_model(model, "FP32 (baseline)", num_warmup=30)

    def test_bf16_simulated(self):
        """
        Test BF16 (Brain Float 16) by converting weights.

        BF16 keeps FP32's dynamic range (8 exponent bits) but reduces
        mantissa to 7 bits. Simulated: convert weights to BF16 then
        back to FP32 for computation.

        Uses 30 warmup iterations (matching native BF16) to ensure
        CPU caches are fully populated and JIT compilation is complete
        before timing begins.
        """
        model = copy.deepcopy(self.model).float()
        model.eval()
        for param in model.parameters():
            param.data = param.data.to(torch.bfloat16).float()
        return self.benchmark_model(model, "BF16 (simulated)", num_warmup=30)

    def test_fp16_simulated(self):
        """
        Test FP16 (half precision) by converting weights.

        FP16 has 5 exponent + 10 mantissa bits. Higher precision than
        BF16 but limited dynamic range (max ~65504).

        Uses 30 warmup iterations (matching native benchmarks) for
        consistent comparison across all precision levels.
        """
        model = copy.deepcopy(self.model).float()
        model.eval()
        for param in model.parameters():
            param.data = param.data.half().float()
        return self.benchmark_model(model, "FP16 (simulated)", num_warmup=30)

    def test_int8_simulated(self, per_channel=False):
        """
        Test INT8 (8-bit integer) by quantising weights.

        per_channel=False: one scale per tensor  (default, simpler)
        per_channel=True:  one scale per output channel (preserves more info)

        Per-channel quantisation (Jacob et al. 2018) uses separate scale
        factors for each output channel, preserving more information than
        a single per-tensor scale.
        """
        model = copy.deepcopy(self.model).float()
        model.eval()

        label = "INT8 Per-Channel (sim)" if per_channel else "INT8 Per-Tensor (sim)"

        for param in model.parameters():
            if param.dim() >= 2:
                if per_channel and param.dim() == 4:
                    # Per-channel: quantise along output channel dimension
                    for c in range(param.shape[0]):
                        channel = param.data[c]
                        scale = channel.abs().max() / 127.0
                        if scale > 0:
                            param.data[c] = (channel / scale).round().clamp(-128, 127) * scale
                else:
                    # Per-tensor: single scale for entire weight matrix
                    scale = param.abs().max() / 127.0
                    if scale > 0:
                        param.data = (param.data / scale).round().clamp(-128, 127) * scale

        return self.benchmark_model(model, label)

    # ── True dynamic quantisation ────────────────────────────────────────

    def test_dynamic_int8(self):
        """
        Test INT8 using PyTorch's dynamic quantisation API.

        Dynamic quantisation quantises weights ahead of time and activations
        on-the-fly during inference. This uses real INT8 compute kernels
        (via fbgemm or qnnpack) rather than simulating in FP32.

        Note: torch.quantization.quantize_dynamic works reliably for
        nn.Linear layers. Conv2d support depends on the backend and
        may fall back to FP32 silently.
        """
        model = copy.deepcopy(self.model).float()
        model.eval()

        try:
            quantised = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            return self.benchmark_model(quantised, "Dynamic INT8 (PyTorch)")
        except Exception as e:
            print(f"  Dynamic INT8 failed: {e}")
            return None

    # ── Native BF16 / FP16 via autocast ──────────────────────────────────

    def test_native_bf16(self):
        """
        Native BF16 inference using torch.cpu.amp.autocast.

        On AMD EPYC 9005 (Zen 5) with AVX-512 BF16 instructions, this
        runs actual BF16 matrix multiplications — not simulated. The CPU
        performs conv/matmul in BF16 natively, giving real speedup.

        We use 30 warmup iterations (instead of the default 10) because
        autocast triggers lazy compilation of BF16 kernels on first use.
        Without enough warmup, the first few BF16 inferences are slower
        and drag down the average, underreporting the true speedup.
        """
        model = copy.deepcopy(self.model).float()
        model.eval()
        try:
            result = self.benchmark_model(
                model, "Native BF16 (autocast)",
                num_warmup=30, autocast_dtype=torch.bfloat16
            )
            return result
        except Exception as e:
            print(f"  Native BF16 failed: {e}")
            return None

    def test_native_fp16(self):
        """
        Native FP16 inference using torch.cpu.amp.autocast.

        IMPORTANT: x86 CPUs (Intel Xeon, AMD EPYC) do NOT have hardware
        FP16 compute instructions. AVX-512 supports BF16 (VNNI-BF16) but
        not FP16 arithmetic. When autocast(dtype=float16) is used on x86,
        PyTorch falls back to software emulation which is ~40x SLOWER
        than FP32 — this is not a useful benchmark on server CPUs.

        We detect x86 architecture and skip this test, logging why.
        FP16 native compute is only useful on ARM (e.g., Apple M-series,
        Graviton) or GPU architectures.
        """
        # Detect x86 architecture
        cpu_arch = platform.machine().lower()
        if cpu_arch in ('x86_64', 'amd64', 'x86'):
            print("  Native FP16: SKIPPED — x86 CPUs lack hardware FP16 compute")
            print("    AVX-512 supports BF16 but not FP16 arithmetic.")
            print("    autocast(float16) on x86 falls back to software emulation (~40x slower)")
            print("    FP16 native compute requires ARM or GPU architectures.")
            return None

        model = copy.deepcopy(self.model).float()
        model.eval()
        try:
            result = self.benchmark_model(
                model, "Native FP16 (autocast)",
                num_warmup=30, autocast_dtype=torch.float16
            )
            return result
        except Exception as e:
            print(f"  Native FP16 failed: {e}")
            return None

    # ── Static INT8 quantisation (the real deal) ─────────────────────────

    def _try_fx_standard(self, model_fp32):
        """
        Strategy 1: Standard FX graph-mode static quantisation.

        This is the ideal path — FX traces the computation graph and
        correctly handles DenseNet's dense connections by automatically
        inserting quant/dequant ops at every boundary.

        May fail if torchxrayvision uses Python builtins (len, range)
        that FX can't trace symbolically.
        """
        try:
            from torch.ao.quantization import quantize_fx, QConfigMapping
        except ImportError:
            return None

        qconfig = torch.quantization.get_default_qconfig('fbgemm')
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        example_input = (self.test_images[0].float(),)

        print("    Strategy 1: Standard FX graph-mode tracing...")
        try:
            model_prepared = quantize_fx.prepare_fx(
                model_fp32, qconfig_mapping, example_input
            )
            print("    SUCCESS — full graph traced")
            return model_prepared, 'fx_standard'
        except Exception as e:
            print(f"    Failed: {e}")
            return None

    def _try_fx_safe_tracer(self, model_fp32):
        """
        Strategy 2: FX with custom SafeTracer that treats DenseLayer as leaf.

        When standard FX tracing fails because DenseNet's _DenseLayer uses
        len() internally, we can stop the tracer from descending into those
        layers. The tracer treats each _DenseLayer as an opaque leaf module,
        which avoids the len() problem while still quantising the conv/bn
        layers inside each DenseLayer via eager-mode qconfig propagation.

        This gives us the best of both worlds: FX handles the top-level
        graph (transitions, classifier) while DenseLayer internals get
        quantised via their qconfig.
        """
        try:
            from torch.ao.quantization import quantize_fx, QConfigMapping
            from torch.ao.quantization.fx.tracer import QuantizationTracer
        except ImportError:
            return None

        print("    Strategy 2: FX with SafeTracer (DenseLayer as leaf)...")

        class SafeTracer(QuantizationTracer):
            """Custom tracer that treats DenseLayer as an opaque leaf module."""
            def is_leaf_module(self, m, module_qualified_name):
                # Treat DenseLayer (and any subclass) as a leaf to avoid
                # tracing into it — this is where len() is called
                class_name = type(m).__name__
                if 'DenseLayer' in class_name or 'DenseBlock' in class_name:
                    return True
                return super().is_leaf_module(m, module_qualified_name)

        qconfig = torch.quantization.get_default_qconfig('fbgemm')
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        example_input = (self.test_images[0].float(),)

        try:
            # Trace with SafeTracer — DenseLayer/DenseBlock are leaves,
            # so len() inside them is never reached by the tracer
            tracer = SafeTracer(
                skipped_module_names=[],
                skipped_module_classes=[]
            )
            graph = tracer.trace(model_fp32)
            traced_model = torch.fx.GraphModule(model_fp32, graph)
            traced_model.eval()

            # Apply qconfig and insert observers via eager-style prepare
            # on the FX-traced GraphModule. This gives us the structural
            # benefits of FX tracing (correct graph) with reliable prepare.
            traced_model.qconfig = qconfig
            torch.quantization.prepare(traced_model, inplace=True)
            print("    SUCCESS — SafeTracer + prepare on GraphModule")
            return traced_model, 'fx_safe_tracer'
        except Exception as e:
            print(f"    Failed: {e}")
            return None

    def _try_eager_quantisation(self, model_fp32):
        """
        Strategy 3: Eager-mode static quantisation with DenseNet-aware wiring.

        Key insight: on this PyTorch build, quantised BatchNorm (including
        fused BNReLU2d) calls aten::aminmax which has NO QuantizedCPU kernel.
        Therefore ALL BatchNorm and ReLU modules must stay in float.

        The approach:
          - Only Conv2d layers run in INT8 (via VNNI VPDPBUSD).
          - BN + ReLU stay in float; DeQuantStub/QuantStub boundaries are
            inserted around each BN-ReLU pair so that convs get QUInt8 input.
          - FloatFunctional.cat() handles the dense-connection concatenation.
          - Stem [conv0, norm0, relu0] is fused — safe because fuse_modules
            folds BN into conv weights at conversion time (no runtime BN).
          - All extra modules (stubs, FloatFunctional) are stored on the
            wrapper via ModuleDicts (NOT on _DenseLayer which is nn.Sequential).
        """
        import types
        from torch.ao.nn.quantized import FloatFunctional

        print("    Strategy 3: Eager-mode with conv-only INT8 + float BN...")

        model = copy.deepcopy(model_fp32)
        model.eval()

        # ── Step 1: Wrap with DenseNetQuantWrapper ──
        wrapped = DenseNetQuantWrapper(model)
        wrapped.eval()

        # ── Step 2: Patch DenseLayer forwards ──
        # Each DenseLayer: norm1 → relu1 → conv1 → norm2 → relu2 → conv2
        # After patching:
        #   x (QUInt8) → dequant → norm1 (float) → relu1 (float) →
        #   quant → conv1 (INT8) → dequant → norm2 (float) → relu2 (float) →
        #   quant → conv2 (INT8) → cat_fn.cat([x, new]) (QUInt8)
        #
        # IMPORTANT: Stubs are stored as sub-modules directly on each
        # _DenseLayer (via add_module), NOT in wrapper-level ModuleDicts.
        # The patched forward accesses them via `self._dq_pre` etc.
        # This ensures that torch.quantization.convert() — which replaces
        # DeQuantStub → DeQuantize and QuantStub → Quantize by walking
        # the module tree — updates the exact objects the forward uses.
        # The previous approach stored stubs in ModuleDicts on the wrapper
        # and captured them via closures; convert() replaced the ModuleDict
        # entries but the closures still referenced the old (unconverted)
        # stubs, so DeQuantStub remained identity after conversion and BN
        # received quantised tensors → native_batch_norm crash.
        patched_layers = 0

        for block_name in ['denseblock1', 'denseblock2', 'denseblock3', 'denseblock4']:
            block = getattr(wrapped.features_body, block_name, None)
            if block is None:
                continue
            for layer_name, layer_module in block.named_children():
                if 'DenseLayer' not in type(layer_module).__name__:
                    continue

                # Register stubs directly on the _DenseLayer module.
                # _DenseLayer extends nn.Sequential, but add_module works
                # fine — the stubs become named children and are found by
                # prepare() and convert() during the module-tree walk.
                layer_module.add_module('_dq_pre', torch.quantization.DeQuantStub())
                layer_module.add_module('_q_mid1', torch.quantization.QuantStub())
                layer_module.add_module('_dq_mid', torch.quantization.DeQuantStub())
                layer_module.add_module('_q_mid2', torch.quantization.QuantStub())
                layer_module.add_module('_cat_fn', FloatFunctional())

                def _make_patched_forward(drop_rate):
                    def _patched_forward(self, x):
                        # x is QUInt8 from previous cat / model entry
                        # ── BN1 + ReLU1 in float ──
                        h = self._dq_pre(x)        # QUInt8 → float
                        h = self.norm1(h)           # float BatchNorm2d
                        h = self.relu1(h)           # float ReLU
                        # ── Conv1 in INT8 ──
                        h = self._q_mid1(h)         # float → QUInt8
                        h = self.conv1(h)           # quantised Conv2d
                        # ── BN2 + ReLU2 in float ──
                        h = self._dq_mid(h)         # QUInt8 → float
                        h = self.norm2(h)           # float BatchNorm2d
                        h = self.relu2(h)           # float ReLU
                        # ── Conv2 in INT8 ──
                        h = self._q_mid2(h)         # float → QUInt8
                        new_features = self.conv2(h)  # quantised Conv2d
                        if drop_rate > 0:
                            new_features = F.dropout(
                                new_features, p=drop_rate, training=self.training
                            )
                        # ── Cat in QUInt8 domain ──
                        return self._cat_fn.cat([x, new_features], dim=1)
                    return _patched_forward

                layer_module.forward = types.MethodType(
                    _make_patched_forward(layer_module.drop_rate),
                    layer_module
                )
                patched_layers += 1

        print(f"    Patched {patched_layers} DenseLayer forwards (conv-only INT8)")

        # ── Step 2b: Patch Transition forwards ──
        # Transition: norm → relu → conv → pool
        # After patching: dequant → norm (float) → relu (float) →
        #                 quant → conv (INT8) → pool (QUInt8)
        # Same fix as DenseLayers: stubs stored on the Transition module
        # so convert() updates the objects the forward actually uses.

        for i in [1, 2, 3]:
            tname = f'transition{i}'
            trans = getattr(wrapped.features_body, tname, None)
            if trans is None:
                continue

            trans.add_module('_dq_pre', torch.quantization.DeQuantStub())
            trans.add_module('_q_mid', torch.quantization.QuantStub())

            def _make_trans_forward():
                def _trans_forward(self, x):
                    h = self._dq_pre(x)   # QUInt8 → float
                    h = self.norm(h)       # float BatchNorm2d
                    h = self.relu(h)       # float ReLU
                    h = self._q_mid(h)     # float → QUInt8
                    h = self.conv(h)       # quantised Conv2d
                    h = self.pool(h)       # pool on QUInt8
                    return h
                return _trans_forward

            trans.forward = types.MethodType(
                _make_trans_forward(), trans
            )

        print(f"    Patched 3 Transition forwards (conv-only INT8)")

        # ── Step 3: Set qconfig and propagate ──
        wrapped.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.propagate_qconfig_(wrapped)

        # AFTER propagation: disable qconfig on all modules that must stay float.
        # propagate_qconfig_ sets qconfig on every submodule, so we override here.
        wrapped.norm5.qconfig = None
        wrapped.classifier.qconfig = None

        for block_name in ['denseblock1', 'denseblock2', 'denseblock3', 'denseblock4']:
            block = getattr(wrapped.features_body, block_name, None)
            if block is None:
                continue
            for layer_name, layer_module in block.named_children():
                if 'DenseLayer' not in type(layer_module).__name__:
                    continue
                # BN and ReLU stay float — no observers, no conversion
                for attr in ['norm1', 'relu1', 'norm2', 'relu2']:
                    sub = getattr(layer_module, attr, None)
                    if sub is not None:
                        sub.qconfig = None

        for i in [1, 2, 3]:
            trans = getattr(wrapped.features_body, f'transition{i}', None)
            if trans is None:
                continue
            for attr in ['norm', 'relu']:
                sub = getattr(trans, attr, None)
                if sub is not None:
                    sub.qconfig = None

        # ── Step 4: Fuse stem only ──
        # [conv0, norm0, relu0] → ConvBnReLU2d. BN is folded into conv
        # weights at conversion time (no runtime BN → no aminmax issue).
        fused = 0
        try:
            torch.quantization.fuse_modules(
                wrapped.features_body,
                [['conv0', 'norm0', 'relu0']],
                inplace=True
            )
            fused += 1
        except Exception as e:
            print(f"    Stem fusion failed: {e}")

        # NO [norm, relu] fusion in DenseLayer or Transition — quantised
        # BNReLU2d calls aten::aminmax which has no QuantizedCPU kernel.
        print(f"    Fused {fused} module group (stem conv+bn+relu only)")

        # ── Step 5: Prepare ──
        torch.quantization.prepare(wrapped, inplace=True)
        print("    SUCCESS — eager-mode prepared (conv-only INT8 + float BN)")
        return wrapped, 'eager'

    def run_static_quantisation(self):
        """
        TRUE static INT8 quantisation — converts layers to actual INT8.

        Tries three strategies in order:
          1. Standard FX graph-mode (best optimisation, may fail on DenseNet)
          2. FX with SafeTracer (treats DenseLayer as leaf to avoid len())
          3. Eager-mode with QuantStub wrapper (always works, good VNNI usage)

        On AMD EPYC 9005 with AVX-512 VNNI, quantised Conv2d and Linear
        layers execute using hardware-accelerated INT8 multiply-accumulate
        instructions (VPDPBUSD), which should give 1.5–2.5x speedup.
        """
        print("\n" + "=" * 60)
        print("STATIC INT8 QUANTISATION (Hardware-Accelerated)")
        print("=" * 60)

        torch.backends.quantized.engine = 'fbgemm'
        print(f"  Backend: fbgemm (AVX-512 VNNI)")
        print(f"  Target: VPDPBUSD INT8 multiply-accumulate instructions")

        model_fp32 = copy.deepcopy(self.model).float().eval()

        # Try strategies in order of preference
        model_prepared = None
        mode = None

        for strategy_fn in [self._try_fx_standard,
                            self._try_fx_safe_tracer,
                            self._try_eager_quantisation]:
            result = strategy_fn(model_fp32)
            if result is not None:
                model_prepared, mode = result
                break

        if model_prepared is None:
            print("  ERROR: All quantisation strategies failed")
            return None

        print(f"  Selected method: {mode}")

        # Step 3: Calibrate — run images through to collect activation statistics
        # DenseNet's dense connectivity (concatenation-heavy forward pass) needs
        # more calibration images than typical architectures to capture the full
        # activation range. 500+ images recommended for stable observer statistics.
        n_calib = min(500, len(self.test_images))
        print(f"  Calibrating with {n_calib} images...")

        with torch.no_grad():
            for i, img in enumerate(tqdm(self.test_images[:n_calib], desc="  Calibrating")):
                model_prepared(img.float())

        # Step 4: Convert to true INT8
        print("  Converting to INT8...")
        try:
            if mode == 'fx_standard':
                # Only standard FX (prepared via prepare_fx) uses convert_fx
                from torch.ao.quantization import quantize_fx
                model_int8 = quantize_fx.convert_fx(model_prepared)
            else:
                # SafeTracer and eager both use torch.quantization.prepare,
                # so they need torch.quantization.convert
                model_int8 = torch.quantization.convert(model_prepared)
        except Exception as e:
            print(f"  Conversion failed ({mode} mode): {e}")
            return None

        # Count how many layers were quantised
        n_quantized = sum(
            1 for m in model_int8.modules()
            if type(m).__module__.startswith('torch.ao.nn.quantized')
        )
        fused_count = n_quantized  # alias used in result dict below
        print(f"  Quantised layers: {n_quantized}")

        # Sanity check: run one image through to verify it works
        try:
            with torch.no_grad():
                _test = model_int8(self.test_images[0].float())
            print(f"  Sanity check passed — output shape: {_test.shape}")
        except Exception as e:
            print(f"  ERROR: Sanity check failed: {e}")
            print("  FX quantisation produced an invalid model.")
            return None

        print("  Static INT8 model ready — benchmarking with hardware INT8 kernels...")

        # Step 7: Benchmark — this uses real AVX-512 VNNI INT8 compute
        result = self.benchmark_model(model_int8, "Static INT8 (fbgemm/VNNI)")

        # Compare accuracy vs FP32
        if hasattr(self, '_outputs') and 'FP32' in self._outputs:
            fp32_out = self._outputs['FP32']
            static_out = result['outputs']

            fp32_probs = torch.sigmoid(fp32_out)
            static_probs = torch.sigmoid(static_out)
            diff = torch.abs(static_probs - fp32_probs)

            mean_diff_pct = float(diff.mean()) * 100
            max_diff_pct = float(diff.max()) * 100
            fp32_time = self.results.get('precision', {}).get('FP32', {}).get('avg_ms', 1)
            speedup = fp32_time / result['avg_ms'] if result['avg_ms'] > 0 else 0

            print(f"\n  Static INT8 Results:")
            print(f"    Speedup vs FP32: {speedup:.2f}x")
            print(f"    Mean prob diff:  {mean_diff_pct:.4f}%")
            print(f"    Max prob diff:   {max_diff_pct:.4f}%")

            # Per-pathology accuracy (using model column mapping)
            per_path = {}
            for i, pathology in enumerate(PATHOLOGIES):
                model_col = self.pathology_col_map.get(i)
                if model_col is not None and model_col < diff.shape[1]:
                    per_path[pathology] = {
                        'mean_diff_pct': float(diff[:, model_col].mean()) * 100,
                        'max_diff_pct': float(diff[:, model_col].max()) * 100,
                    }

            # Clinical safety check
            safe = all(
                per_path.get(cp, {}).get('max_diff_pct', 999) < SAFETY_THRESHOLD_PERCENT
                for cp in CRITICAL_PATHOLOGIES
            )
            print(f"    Clinical safety: {'PASS' if safe else 'FAIL'}")

            for cp in CRITICAL_PATHOLOGIES:
                if cp in per_path:
                    p = per_path[cp]
                    status = "SAFE" if p['max_diff_pct'] < SAFETY_THRESHOLD_PERCENT else "REVIEW"
                    print(f"      {cp:<20} max {p['max_diff_pct']:.4f}%  [{status}]")

            static_result = {
                'avg_ms': result['avg_ms'],
                'std_ms': result['std_ms'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'speedup': speedup,
                'mean_diff_pct': mean_diff_pct,
                'max_diff_pct': max_diff_pct,
                'per_pathology': per_path,
                'clinically_safe': safe,
                'backend': 'fbgemm',
                'n_calibration_images': n_calib,
                'fused_layers': fused_count,
            }
        else:
            static_result = {
                'avg_ms': result['avg_ms'],
                'std_ms': result['std_ms'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'backend': 'fbgemm',
                'n_calibration_images': n_calib,
                'fused_layers': fused_count,
            }

        self.results['static_int8'] = static_result
        self._outputs['Static_INT8'] = result['outputs']
        self._times['Static_INT8'] = result['times']

        return static_result

    # ── Run native BF16/FP16 with accuracy comparison ────────────────────

    def run_native_precision(self):
        """
        Run native BF16 and FP16 inference using torch.cpu.amp.autocast.

        Unlike the simulated tests (which just cast weights and compute in FP32),
        autocast tells PyTorch to run the actual matmul/conv2d operations in the
        lower precision dtype. On EPYC 9005 with AVX-512 BF16/FP16 instructions,
        this uses the hardware's native low-precision compute units.
        """
        print("\n" + "=" * 60)
        print("NATIVE LOW-PRECISION INFERENCE (autocast)")
        print("=" * 60)

        fp32_out = self._outputs.get('FP32')
        fp32_time = self.results.get('precision', {}).get('FP32', {}).get('avg_ms', 1)
        native_results = {}

        for dtype_name, dtype_val in [('Native_BF16', torch.bfloat16),
                                       ('Native_FP16', torch.float16)]:
            print(f"\n  Testing {dtype_name}...")
            try:
                if dtype_name == 'Native_BF16':
                    res = self.test_native_bf16()
                else:
                    res = self.test_native_fp16()

                if res is None:
                    continue

                speedup = fp32_time / res['avg_ms'] if res['avg_ms'] > 0 else 0

                if fp32_out is not None:
                    fp32_probs = torch.sigmoid(fp32_out)
                    native_probs = torch.sigmoid(res['outputs'])
                    diff = torch.abs(native_probs - fp32_probs)

                    mean_diff_pct = float(diff.mean()) * 100
                    max_diff_pct = float(diff.max()) * 100

                    per_path = {}
                    for i, pathology in enumerate(PATHOLOGIES):
                        model_col = self.pathology_col_map.get(i)
                        if model_col is not None and model_col < diff.shape[1]:
                            per_path[pathology] = {
                                'mean_diff_pct': float(diff[:, model_col].mean()) * 100,
                                'max_diff_pct': float(diff[:, model_col].max()) * 100,
                            }

                    safe = all(
                        per_path.get(cp, {}).get('max_diff_pct', 999) < SAFETY_THRESHOLD_PERCENT
                        for cp in CRITICAL_PATHOLOGIES
                    )

                    print(f"    Speedup vs FP32: {speedup:.2f}x")
                    print(f"    Mean prob diff:  {mean_diff_pct:.4f}%")
                    print(f"    Max prob diff:   {max_diff_pct:.4f}%")
                    print(f"    Clinical safety: {'PASS' if safe else 'FAIL'}")

                    native_results[dtype_name] = {
                        'avg_ms': res['avg_ms'],
                        'std_ms': res['std_ms'],
                        'ci_lower': res['ci_lower'],
                        'ci_upper': res['ci_upper'],
                        'speedup': speedup,
                        'mean_diff_pct': mean_diff_pct,
                        'max_diff_pct': max_diff_pct,
                        'per_pathology': per_path,
                        'clinically_safe': safe,
                    }
                else:
                    native_results[dtype_name] = {
                        'avg_ms': res['avg_ms'],
                        'std_ms': res['std_ms'],
                        'ci_lower': res['ci_lower'],
                        'ci_upper': res['ci_upper'],
                        'speedup': speedup,
                    }

                self._outputs[dtype_name] = res['outputs']
                self._times[dtype_name] = res['times']

                # Statistical comparison vs FP32
                if hasattr(self, '_times') and 'FP32' in self._times:
                    t_stat, p_val = StatisticalAnalysis.paired_ttest(
                        self._times['FP32'], res['times']
                    )
                    d = StatisticalAnalysis.cohens_d(self._times['FP32'], res['times'])

                    # Bootstrap CI on speedup ratio
                    sp_point, sp_lo, sp_hi = StatisticalAnalysis.bootstrap_speedup_ci(
                        self._times['FP32'], res['times']
                    )

                    native_results[dtype_name]['t_statistic'] = t_stat
                    native_results[dtype_name]['p_value'] = p_val
                    native_results[dtype_name]['cohens_d'] = d
                    native_results[dtype_name]['effect'] = StatisticalAnalysis.interpret_cohens_d(d)
                    native_results[dtype_name]['speedup_ci'] = {
                        'point': sp_point, 'lower': sp_lo, 'upper': sp_hi
                    }
                    if p_val is not None:
                        print(f"    t-test vs FP32:  p={p_val:.6f}  d={d:.3f} ({StatisticalAnalysis.interpret_cohens_d(d)})")
                        print(f"    Speedup CI:      {sp_point:.2f}x [{sp_lo:.2f}-{sp_hi:.2f}]")

            except Exception as e:
                print(f"    {dtype_name} failed: {e}")

        self.results['native_precision'] = native_results
        return native_results

    # ── Run all precision comparisons ────────────────────────────────────

    def run_precision_comparison(self):
        """Compare all precision levels with statistical analysis."""
        print("\n" + "=" * 60)
        print("PRECISION COMPARISON")
        print("=" * 60)

        fp64 = self.test_fp64()
        fp32 = self.test_fp32()
        bf16 = self.test_bf16_simulated()
        fp16 = self.test_fp16_simulated()
        int8 = self.test_int8_simulated(per_channel=False)
        int8_pc = self.test_int8_simulated(per_channel=True)
        dyn_int8 = self.test_dynamic_int8()

        fp32_out = fp32['outputs']
        fp32_time = fp32['avg_ms']

        def compute_diff(other_out, label):
            mse = float(torch.mean((other_out - fp32_out) ** 2))
            fp32_probs = torch.sigmoid(fp32_out)
            other_probs = torch.sigmoid(other_out)
            abs_diff = torch.abs(other_probs - fp32_probs)
            mean_pct = float(abs_diff.mean()) * 100
            max_pct = float(abs_diff.max()) * 100

            # Per-pathology max difference (using model column mapping)
            per_path = {}
            for i, name in enumerate(PATHOLOGIES):
                model_col = self.pathology_col_map.get(i)
                if model_col is not None and model_col < abs_diff.shape[1]:
                    per_path[name] = {
                        'mean_diff_pct': float(abs_diff[:, model_col].mean()) * 100,
                        'max_diff_pct': float(abs_diff[:, model_col].max()) * 100,
                    }

            return {
                'mse': mse, 'mean_diff_pct': mean_pct, 'max_diff_pct': max_pct,
                'per_pathology': per_path
            }

        precisions = {
            'FP64': fp64, 'FP32': fp32, 'BF16': bf16, 'FP16': fp16,
            'INT8': int8, 'INT8_PerChannel': int8_pc,
        }
        if dyn_int8:
            precisions['Dynamic_INT8'] = dyn_int8

        accuracy = {}
        for label, res in precisions.items():
            if label == 'FP32':
                continue
            accuracy[label] = compute_diff(res['outputs'], label)

        # Statistical comparison of timing (paired t-tests vs FP32)
        stat_results = {}
        for label, res in precisions.items():
            if label == 'FP32':
                continue
            t_stat, p_val = StatisticalAnalysis.paired_ttest(fp32['times'], res['times'])
            d = StatisticalAnalysis.cohens_d(fp32['times'], res['times'])
            stat_results[label] = {
                't_statistic': t_stat, 'p_value': p_val,
                'cohens_d': d, 'effect': StatisticalAnalysis.interpret_cohens_d(d)
            }

        # Compute bootstrap CIs on speedup ratios
        speedup_cis = {}
        for label in ['FP64', 'BF16', 'FP16', 'INT8', 'INT8_PerChannel', 'Dynamic_INT8']:
            if label not in precisions or label == 'FP32':
                continue
            sp_point, sp_lo, sp_hi = StatisticalAnalysis.bootstrap_speedup_ci(
                fp32['times'], precisions[label]['times']
            )
            speedup_cis[label] = {'point': sp_point, 'lower': sp_lo, 'upper': sp_hi}

        # Print summary
        print("\n--- Summary (simulated — real hardware results below) ---")
        for label in ['FP64', 'BF16', 'FP16', 'INT8', 'INT8_PerChannel', 'Dynamic_INT8']:
            if label not in precisions:
                continue
            res = precisions[label]
            sp = fp32_time / res['avg_ms']
            acc = accuracy.get(label, {})
            md = acc.get('max_diff_pct', 0)
            stat = stat_results.get(label, {})
            p = stat.get('p_value')
            p_str = f"p={p:.4f}" if p is not None else "p=N/A"
            sci = speedup_cis.get(label, {})
            sp_ci_str = f"  [{sci.get('lower', 0):.2f}-{sci.get('upper', 0):.2f}]" if sci else ""
            print(f"  {label:<20} {res['avg_ms']:.1f} ms  speedup {sp:.2f}x{sp_ci_str}  "
                  f"max diff {md:.4f}%  {p_str}")

        print(f"\n  Key: FP64 is wasteful, FP16/BF16 are near-lossless, "
              f"INT8 per-channel is better than per-tensor")

        # Store everything
        self.results['precision'] = {
            label: {
                'avg_ms': r['avg_ms'], 'std_ms': r['std_ms'],
                'ci_lower': r['ci_lower'], 'ci_upper': r['ci_upper'],
                'speedup': fp32_time / r['avg_ms'],
            }
            for label, r in precisions.items()
        }
        self.results['accuracy'] = accuracy
        self.results['statistical_tests'] = stat_results
        self.results['speedup_cis'] = speedup_cis

        # Store outputs for later analysis
        self._outputs = {label: r['outputs'] for label, r in precisions.items()}
        self._times = {label: r['times'] for label, r in precisions.items()}

    # ── Block sensitivity ────────────────────────────────────────────────

    def analyse_block_sensitivity(self):
        """
        Test which DenseNet blocks are most sensitive to INT8 quantisation.

        Method: Quantise each block independently and measure output change.
        Higher MSE = more sensitive to quantisation = needs higher precision.
        """
        print("\n" + "=" * 60)
        print("BLOCK SENSITIVITY ANALYSIS")
        print("=" * 60)

        baseline = copy.deepcopy(self.model).float()
        baseline.eval()

        with torch.no_grad():
            baseline_out = torch.cat(
                [baseline(img.float()) for img in self.test_images], dim=0
            )

        sensitivities = {}

        for block_num in [1, 2, 3, 4]:
            block_name = f'denseblock{block_num}'
            test_model = copy.deepcopy(self.model).float()
            test_model.eval()

            for name, param in test_model.named_parameters():
                if block_name in name and param.dim() >= 2:
                    scale = param.abs().max() / 127.0
                    if scale > 0:
                        param.data = (param.data / scale).round().clamp(-128, 127) * scale

            with torch.no_grad():
                test_out = torch.cat(
                    [test_model(img.float()) for img in self.test_images], dim=0
                )

            mse = float(torch.mean((test_out - baseline_out) ** 2))
            max_diff = float(torch.abs(test_out - baseline_out).max())
            sensitivities[block_name] = {'mse': mse, 'max_diff': max_diff}

        # Also test transition layers
        for trans_num in [1, 2, 3]:
            trans_name = f'transition{trans_num}'
            test_model = copy.deepcopy(self.model).float()
            test_model.eval()

            for name, param in test_model.named_parameters():
                if trans_name in name and param.dim() >= 2:
                    scale = param.abs().max() / 127.0
                    if scale > 0:
                        param.data = (param.data / scale).round().clamp(-128, 127) * scale

            with torch.no_grad():
                test_out = torch.cat(
                    [test_model(img.float()) for img in self.test_images], dim=0
                )
            mse = float(torch.mean((test_out - baseline_out) ** 2))
            max_diff = float(torch.abs(test_out - baseline_out).max())
            sensitivities[trans_name] = {'mse': mse, 'max_diff': max_diff}

        print("\nComponent sensitivity (MSE when quantised to INT8):")
        for name, vals in sorted(sensitivities.items(),
                                 key=lambda x: x[1]['mse'], reverse=True):
            level = "HIGH" if vals['mse'] > 5e-7 else "MEDIUM" if vals['mse'] > 5e-8 else "LOW"
            print(f"  {name:<20} MSE: {vals['mse']:.2e}  max: {vals['max_diff']:.4f}  [{level}]")

        # Hypothesis check
        dense_mses = {k: v['mse'] for k, v in sensitivities.items()
                      if k.startswith('denseblock')}
        early = (dense_mses.get('denseblock1', 0) + dense_mses.get('denseblock2', 0)) / 2
        late = (dense_mses.get('denseblock3', 0) + dense_mses.get('denseblock4', 0)) / 2

        if early > late:
            print("\n  Hypothesis SUPPORTED: early blocks more sensitive")
        else:
            print("\n  Hypothesis NOT SUPPORTED: sensitivity is non-monotonic")
            print("  Revised approach: assign precision empirically by measured sensitivity")

        self.results['block_sensitivity'] = sensitivities
        self.results['hypothesis_supported'] = bool(early > late)

    # ── Layer-level sensitivity (integrated from Layersensitivity.py) ─────

    def analyse_layer_sensitivity(self, max_layers=120):
        """
        Per-layer sensitivity analysis — tests each conv layer individually.

        Integrated from work-in-progress/Layersensitivity.py.
        Enhanced to test both FP16 and INT8 sensitivity, and to use
        real images rather than synthetic data.
        """
        print("\n" + "=" * 60)
        print("LAYER SENSITIVITY ANALYSIS")
        print("=" * 60)

        # Use a subset of images for per-layer testing (speed)
        subset = self.test_images[:min(50, len(self.test_images))]

        baseline = copy.deepcopy(self.model).float()
        baseline.eval()
        with torch.no_grad():
            baseline_out = torch.cat([baseline(img.float()) for img in subset], dim=0)

        # Find all conv layers
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append(name)

        print(f"\n  Found {len(conv_layers)} Conv2d layers")
        print(f"  Testing up to {min(max_layers, len(conv_layers))} layers...\n")

        layer_results = []

        for layer_name in tqdm(conv_layers[:max_layers], desc="  Layers"):
            result = {'layer': layer_name}

            # Parse position
            position = {'block': None, 'layer_num': None, 'type': 'other'}
            for part in layer_name.split('.'):
                if 'denseblock' in part:
                    position['block'] = int(part.replace('denseblock', ''))
                    position['type'] = 'dense'
                elif 'denselayer' in part:
                    position['layer_num'] = int(part.replace('denselayer', ''))
                elif 'transition' in part:
                    position['type'] = 'transition'
            if 'conv0' in layer_name and position['block'] is None:
                position['type'] = 'stem'
            result['position'] = position

            # Test INT8 sensitivity for this layer
            try:
                model_int8 = copy.deepcopy(self.model).float()
                model_int8.eval()
                for name, module in model_int8.named_modules():
                    if name == layer_name and isinstance(module, nn.Conv2d):
                        scale = module.weight.abs().max() / 127.0
                        if scale > 0:
                            module.weight.data = (
                                (module.weight.data / scale).round().clamp(-128, 127) * scale
                            )
                        break

                with torch.no_grad():
                    out_int8 = torch.cat(
                        [model_int8(img.float()) for img in subset], dim=0
                    )
                diff = out_int8 - baseline_out
                result['int8_mse'] = float(torch.mean(diff ** 2))
                result['int8_max_diff'] = float(torch.max(torch.abs(diff)))

            except Exception as e:
                result['int8_mse'] = None
                result['error'] = str(e)

            # Test FP16 sensitivity for this layer
            try:
                model_fp16 = copy.deepcopy(self.model).float()
                model_fp16.eval()
                for name, module in model_fp16.named_modules():
                    if name == layer_name and isinstance(module, nn.Conv2d):
                        module.weight.data = module.weight.data.half().float()
                        if module.bias is not None:
                            module.bias.data = module.bias.data.half().float()
                        break

                with torch.no_grad():
                    out_fp16 = torch.cat(
                        [model_fp16(img.float()) for img in subset], dim=0
                    )
                diff = out_fp16 - baseline_out
                result['fp16_mse'] = float(torch.mean(diff ** 2))

            except Exception:
                result['fp16_mse'] = None

            layer_results.append(result)

        # Summarise by block
        block_agg = defaultdict(list)
        for r in layer_results:
            if r.get('int8_mse') is not None:
                pos = r['position']
                key = f"denseblock{pos['block']}" if pos['block'] else pos['type']
                block_agg[key].append(r['int8_mse'])

        print("\n  Per-block aggregation (INT8):")
        for block in sorted(block_agg.keys()):
            mses = block_agg[block]
            print(f"    {block:<20} avg MSE: {np.mean(mses):.2e}  "
                  f"max MSE: {np.max(mses):.2e}  ({len(mses)} layers)")

        # ── Layer-type sensitivity breakdown (RQ2) ──────────────────────
        # Group layers by architectural type to determine optimal precision
        # per layer type: conv1x1, conv3x3, transition, stem, classifier.
        type_agg = defaultdict(lambda: {'int8_mses': [], 'fp16_mses': []})

        for r in layer_results:
            layer_name = r['layer']
            pos = r['position']

            # Classify by architectural role
            if pos['type'] == 'stem':
                layer_type = 'stem_conv'
            elif pos['type'] == 'transition':
                layer_type = 'transition_conv'
            elif 'classifier' in layer_name or 'fc' in layer_name:
                layer_type = 'classifier_linear'
            elif pos['type'] == 'dense':
                # DenseNet dense layers have conv1 (1x1 bottleneck) and conv2 (3x3)
                if 'conv1' in layer_name:
                    layer_type = 'dense_conv1x1'
                elif 'conv2' in layer_name:
                    layer_type = 'dense_conv3x3'
                else:
                    layer_type = 'dense_other'
            else:
                layer_type = 'other'

            if r.get('int8_mse') is not None:
                type_agg[layer_type]['int8_mses'].append(r['int8_mse'])
            if r.get('fp16_mse') is not None:
                type_agg[layer_type]['fp16_mses'].append(r['fp16_mse'])

        print("\n  ── Layer-Type Sensitivity Breakdown (RQ2) ──")
        print(f"    {'Layer Type':<22} {'Count':>5} {'INT8 mean':>12} {'INT8 max':>12} {'FP16 mean':>12}")

        type_summary = {}
        for ltype in sorted(type_agg.keys()):
            data = type_agg[ltype]
            int8_mses = data['int8_mses']
            fp16_mses = data['fp16_mses']
            count = len(int8_mses)

            int8_mean = float(np.mean(int8_mses)) if int8_mses else None
            int8_max = float(np.max(int8_mses)) if int8_mses else None
            fp16_mean = float(np.mean(fp16_mses)) if fp16_mses else None

            int8_m_str = f"{int8_mean:.2e}" if int8_mean is not None else "N/A"
            int8_x_str = f"{int8_max:.2e}" if int8_max is not None else "N/A"
            fp16_m_str = f"{fp16_mean:.2e}" if fp16_mean is not None else "N/A"

            print(f"    {ltype:<22} {count:>5} {int8_m_str:>12} {int8_x_str:>12} {fp16_m_str:>12}")

            type_summary[ltype] = {
                'count': count,
                'int8_mean_mse': int8_mean,
                'int8_max_mse': int8_max,
                'fp16_mean_mse': fp16_mean,
            }

        # Recommend precision per type
        print("\n  ── Recommended Precision per Layer Type ──")
        for ltype, summary in sorted(type_summary.items(), key=lambda x: x[1].get('int8_mean_mse') or 0, reverse=True):
            int8_mse = summary.get('int8_mean_mse')
            fp16_mse = summary.get('fp16_mean_mse')

            if int8_mse is not None and int8_mse > 1e-3:
                recommendation = "FP32 or BF16 (high INT8 sensitivity)"
            elif int8_mse is not None and int8_mse > 1e-5:
                recommendation = "BF16 (moderate INT8 sensitivity)"
            else:
                recommendation = "INT8 safe"
            summary['recommendation'] = recommendation
            print(f"    {ltype:<22} → {recommendation}")

        # Analyse correlation between layer position and sensitivity
        positions = []
        mses = []
        for r in layer_results:
            if r.get('int8_mse') is not None and r['position']['layer_num'] is not None:
                positions.append(r['position']['layer_num'])
                mses.append(r['int8_mse'])

        if len(positions) > 5:
            correlation = float(np.corrcoef(positions, mses)[0, 1])
            print(f"\n  Correlation (layer position vs INT8 sensitivity): {correlation:.3f}")
            if correlation < -0.3:
                print("    Supports hypothesis: later layers less sensitive")
            elif correlation > 0.3:
                print("    Contradicts hypothesis: later layers MORE sensitive")
            else:
                print("    Inconclusive: no strong linear relationship")
        else:
            correlation = None

        self.results['layer_sensitivity'] = {
            'layers': layer_results,
            'block_aggregation': {k: {'mean': float(np.mean(v)), 'max': float(np.max(v)),
                                       'count': len(v)}
                                  for k, v in block_agg.items()},
            'type_breakdown': type_summary,
            'position_correlation': correlation
        }

    # ── Clinical accuracy analysis (integrated from real data evaluation.py) ─

    def analyse_clinical_accuracy(self):
        """
        Check if quantisation affects critical pathology detection.

        Integrated from work-in-progress/real data evaluation.py.
        Computes per-pathology probability differences and flags any
        that exceed the clinical safety threshold.
        """
        print("\n" + "=" * 60)
        print("CLINICAL ACCURACY ANALYSIS")
        print("=" * 60)

        if not hasattr(self, '_outputs'):
            print("  Run precision comparison first")
            return

        fp32_probs_raw = torch.sigmoid(self._outputs['FP32']).numpy()
        fp32_probs = self._reindex_probs(fp32_probs_raw)

        clinical_results = {}

        for label in ['BF16', 'FP16', 'INT8', 'INT8_PerChannel', 'Dynamic_INT8']:
            if label not in self._outputs:
                continue

            other_probs_raw = torch.sigmoid(self._outputs[label]).numpy()
            other_probs = self._reindex_probs(other_probs_raw)
            diff = np.abs(other_probs - fp32_probs)

            print(f"\n  {label} vs FP32:")
            print(f"    Overall:  mean {float(diff.mean()) * 100:.4f}%  "
                  f"max {float(diff.max()) * 100:.4f}%")

            per_path = {}
            for i, pathology in enumerate(PATHOLOGIES):
                if i >= diff.shape[1]:
                    continue
                path_diff = diff[:, i]
                mean_d = float(path_diff.mean()) * 100
                max_d = float(path_diff.max()) * 100
                status = "SAFE" if max_d < SAFETY_THRESHOLD_PERCENT else "REVIEW NEEDED"
                per_path[pathology] = {
                    'mean_diff_pct': mean_d, 'max_diff_pct': max_d,
                    'status': status
                }

            # Print critical pathologies
            print(f"    Critical pathologies:")
            for cp in CRITICAL_PATHOLOGIES:
                if cp in per_path:
                    p = per_path[cp]
                    print(f"      {cp:<20} mean {p['mean_diff_pct']:.4f}%  "
                          f"max {p['max_diff_pct']:.4f}%  [{p['status']}]")

            clinical_results[label] = {
                'overall_mean_pct': float(diff.mean()) * 100,
                'overall_max_pct': float(diff.max()) * 100,
                'per_pathology': per_path,
                'all_critical_safe': all(
                    per_path.get(cp, {}).get('max_diff_pct', 999) < SAFETY_THRESHOLD_PERCENT
                    for cp in CRITICAL_PATHOLOGIES
                )
            }

        self.results['clinical'] = clinical_results

    # ── Calibration and AUC analysis ────────────────────────────────────

    def analyse_calibration_and_auc(self, labels_matrix=None):
        """
        Compute ECE, MCE, AUC-ROC per pathology per precision, with bootstrap CIs.

        If labels_matrix is None, we use the FP32 model's predictions as
        pseudo-labels (binarised at 0.5 threshold). This measures whether
        quantised models agree with FP32's *decisions*, not ground truth.
        For ground-truth evaluation, pass actual ChestX-ray14 labels.

        Args:
            labels_matrix: (N, C) binary labels, or None for pseudo-label mode
        """
        print("\n" + "=" * 60)
        print("CALIBRATION & AUC-ROC ANALYSIS")
        print("=" * 60)

        if not hasattr(self, '_outputs'):
            print("  Run precision comparison first")
            return

        fp32_probs_raw = torch.sigmoid(self._outputs['FP32']).numpy()
        # Reindex from model column order to PATHOLOGIES order
        fp32_probs = self._reindex_probs(fp32_probs_raw)
        # Pseudo-labels: FP32 predictions binarised at 0.5
        if labels_matrix is None:
            labels_matrix = (fp32_probs > 0.5).astype(np.float32)
            print("  Mode: pseudo-labels from FP32 (measures decision agreement)")
        else:
            labels_matrix = np.array(labels_matrix)
            print("  Mode: ground-truth labels")

        calibration_results = {}

        # Compute for FP32 itself (baseline calibration)
        prec_labels = ['FP32', 'BF16', 'FP16', 'INT8', 'INT8_PerChannel', 'Dynamic_INT8']
        # Include native/static if available
        if hasattr(self, '_outputs'):
            for extra in ['Native_BF16', 'Native_FP16', 'Static_INT8']:
                if extra in self._outputs:
                    prec_labels.append(extra)

        for label in prec_labels:
            if label not in self._outputs:
                continue

            probs_raw = torch.sigmoid(self._outputs[label]).numpy()
            probs = self._reindex_probs(probs_raw)
            prec_result = {'ece': {}, 'mce': {}, 'auc': {}, 'auc_ci': {}}

            print(f"\n  {label}:")

            # Per-pathology ECE, MCE, AUC
            for i, pathology in enumerate(PATHOLOGIES):
                if i >= probs.shape[1] or i >= labels_matrix.shape[1]:
                    continue

                p = probs[:, i]
                y = labels_matrix[:, i]

                # ECE and MCE
                ece, bin_accs, bin_confs, bin_counts = ClinicalMetrics.expected_calibration_error(p, y)
                mce = ClinicalMetrics.maximum_calibration_error(p, y)
                prec_result['ece'][pathology] = ece
                prec_result['mce'][pathology] = mce

                # AUC with bootstrap CI
                auc, auc_lo, auc_hi = ClinicalMetrics.bootstrap_auc_ci(y, p, n_bootstrap=2000)
                prec_result['auc'][pathology] = auc
                prec_result['auc_ci'][pathology] = {'lower': auc_lo, 'upper': auc_hi}

            # Aggregate ECE and MCE
            ece_vals = [v for v in prec_result['ece'].values() if v is not None]
            mce_vals = [v for v in prec_result['mce'].values() if v is not None]
            mean_ece = np.mean(ece_vals) if ece_vals else None
            max_mce = max(mce_vals) if mce_vals else None

            prec_result['mean_ece'] = float(mean_ece) if mean_ece is not None else None
            prec_result['max_mce'] = float(max_mce) if max_mce is not None else None

            print(f"    Mean ECE: {mean_ece:.4f}" if mean_ece else "    Mean ECE: N/A")
            print(f"    Max MCE:  {max_mce:.4f}" if max_mce else "    Max MCE: N/A")

            # Print critical pathology AUCs
            for cp in CRITICAL_PATHOLOGIES:
                auc = prec_result['auc'].get(cp)
                ci = prec_result['auc_ci'].get(cp, {})
                if auc is not None:
                    ci_str = f" [{ci.get('lower', 0):.3f}-{ci.get('upper', 0):.3f}]" if ci.get('lower') else ""
                    print(f"    {cp:<20} AUC: {auc:.4f}{ci_str}  ECE: {prec_result['ece'].get(cp, 0):.4f}")

            # Bootstrap CI for probability differences vs FP32
            if label != 'FP32':
                prob_diff_cis = {}
                for i, pathology in enumerate(PATHOLOGIES):
                    if i >= probs.shape[1] or i >= fp32_probs.shape[1]:
                        continue
                    mean_d, lo, hi = ClinicalMetrics.bootstrap_prob_diff_ci(
                        fp32_probs[:, i], probs[:, i], n_bootstrap=2000
                    )
                    prob_diff_cis[pathology] = {
                        'mean_diff': float(mean_d) * 100,
                        'ci_lower': float(lo) * 100,
                        'ci_upper': float(hi) * 100
                    }
                prec_result['prob_diff_ci'] = prob_diff_cis

            calibration_results[label] = prec_result

        self.results['calibration'] = calibration_results
        return calibration_results

    # ── McNemar's test for clinical equivalence ─────────────────────────

    def mcnemar_test(self, labels_matrix=None, threshold=None):
        """
        McNemar's test comparing binary classification decisions between
        FP32 and each quantised format, per pathology.

        McNemar's test is appropriate here because we're comparing two
        classifiers (FP32 vs quantised) on the same samples. It tests
        whether the disagreements are symmetric — i.e. whether one
        classifier systematically flips decisions that the other gets right.

        The test statistic uses the discordant pairs:
          b = FP32 correct, quantised wrong
          c = FP32 wrong, quantised correct
          chi2 = (|b - c| - 1)^2 / (b + c)   [with continuity correction]

        A non-significant result (p > 0.05) means no evidence that the
        quantised format makes systematically different decisions — which
        is what we want for clinical equivalence.

        Args:
            labels_matrix: (N, C) binary ground-truth labels.
                           If None, uses FP32 pseudo-labels (binarised at threshold).
            threshold: probability threshold for binarising predictions
        """
        print("\n" + "=" * 60)
        print("McNEMAR'S TEST — Classification Decision Equivalence")
        print("=" * 60)

        if not HAS_SCIPY:
            print("  Skipped — scipy not available")
            return

        if not hasattr(self, '_outputs') or 'FP32' not in self._outputs:
            print("  Run precision comparison first")
            return

        fp32_probs_raw = torch.sigmoid(self._outputs['FP32']).numpy()
        fp32_probs = self._reindex_probs(fp32_probs_raw)

        # Adaptive threshold: use per-pathology prevalence-aware threshold.
        # torchxrayvision outputs tend to be well below 0.5 for most pathologies,
        # so a fixed 0.5 threshold makes everything predict negative → b=c=0.
        # Instead, use the median FP32 probability per pathology as a more
        # discriminative threshold, or fall back to a user-provided one.
        if threshold is None:
            # Use per-pathology Youden-optimal threshold from FP32 predictions
            # (maximises sensitivity + specificity) when we have ground truth
            per_pathology_thresholds = []
            if labels_matrix is not None:
                lm = np.array(labels_matrix)
                for i in range(len(PATHOLOGIES)):
                    if i >= fp32_probs.shape[1] or i >= lm.shape[1]:
                        per_pathology_thresholds.append(0.5)
                        continue
                    y = lm[:, i]
                    p = fp32_probs[:, i]
                    if len(np.unique(y)) < 2:
                        per_pathology_thresholds.append(float(np.median(p)))
                        continue
                    # Youden's J: find threshold that maximises (TPR - FPR)
                    sorted_p = np.sort(np.unique(p))
                    best_t, best_j = 0.5, -1
                    for t in sorted_p:
                        tp = np.sum((p >= t) & (y == 1))
                        fn = np.sum((p < t) & (y == 1))
                        fp = np.sum((p >= t) & (y == 0))
                        tn = np.sum((p < t) & (y == 0))
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                        j = tpr - fpr
                        if j > best_j:
                            best_j = j
                            best_t = float(t)
                    per_pathology_thresholds.append(best_t)
                use_adaptive = True
            else:
                per_pathology_thresholds = [0.5] * len(PATHOLOGIES)
                use_adaptive = False
        else:
            per_pathology_thresholds = [threshold] * len(PATHOLOGIES)
            use_adaptive = False

        if labels_matrix is None:
            # Pseudo-label mode: binarise FP32 predictions
            labels_matrix = (fp32_probs > 0.5).astype(np.float32)
            mode_str = "pseudo-labels (FP32 binarised)"
        else:
            labels_matrix = np.array(labels_matrix)
            mode_str = "ground-truth labels"

        print(f"  Mode: {mode_str}")
        if use_adaptive:
            mean_t = np.mean(per_pathology_thresholds)
            print(f"  Threshold: per-pathology Youden-optimal (mean={mean_t:.4f})")
        else:
            print(f"  Threshold: {per_pathology_thresholds[0]}")

        # FP32 binary predictions (per-pathology thresholds)
        fp32_binary = np.zeros_like(fp32_probs, dtype=int)
        for i in range(min(len(per_pathology_thresholds), fp32_probs.shape[1])):
            fp32_binary[:, i] = (fp32_probs[:, i] > per_pathology_thresholds[i]).astype(int)

        mcnemar_results = {}

        prec_labels = ['BF16', 'FP16', 'INT8', 'INT8_PerChannel', 'Dynamic_INT8',
                       'Native_BF16', 'Native_FP16', 'Static_INT8']

        for label in prec_labels:
            if label not in self._outputs:
                continue

            probs_raw = torch.sigmoid(self._outputs[label]).numpy()
            probs = self._reindex_probs(probs_raw)
            other_binary = np.zeros_like(probs, dtype=int)
            for i in range(min(len(per_pathology_thresholds), probs.shape[1])):
                other_binary[:, i] = (probs[:, i] > per_pathology_thresholds[i]).astype(int)

            per_pathology = {}
            print(f"\n  {label}:")
            print(f"    {'Pathology':<22} {'b':>4} {'c':>4} {'chi2':>8} {'p-value':>10} {'Result':<12}")

            for i, pathology in enumerate(PATHOLOGIES):
                if i >= fp32_binary.shape[1] or i >= other_binary.shape[1]:
                    continue
                if i >= labels_matrix.shape[1]:
                    continue

                y = labels_matrix[:, i].astype(int)
                fp32_correct = (fp32_binary[:, i] == y).astype(int)
                other_correct = (other_binary[:, i] == y).astype(int)

                # Discordant pairs
                b = int(np.sum((fp32_correct == 1) & (other_correct == 0)))  # FP32 right, other wrong
                c = int(np.sum((fp32_correct == 0) & (other_correct == 1)))  # FP32 wrong, other right

                if b + c == 0:
                    # Perfect agreement — no discordant pairs
                    per_pathology[pathology] = {
                        'b': b, 'c': c,
                        'chi2': 0.0, 'p_value': 1.0,
                        'significant': False,
                        'note': 'perfect agreement'
                    }
                    print(f"    {pathology:<22} {b:>4} {c:>4} {'--':>8} {'--':>10} {'AGREE':>12}")
                    continue

                # McNemar's test with continuity correction
                chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                p_value = 1.0 - scipy_stats.chi2.cdf(chi2, df=1)

                significant = p_value < 0.05
                result_str = "DIFFER *" if significant else "EQUIV"

                per_pathology[pathology] = {
                    'b': b, 'c': c,
                    'chi2': float(chi2),
                    'p_value': float(p_value),
                    'significant': significant,
                }

                marker = " *" if pathology in CRITICAL_PATHOLOGIES else ""
                print(f"    {pathology:<22} {b:>4} {c:>4} {chi2:>8.3f} {p_value:>10.6f} {result_str:<12}{marker}")

            # Summary for this precision
            n_sig = sum(1 for v in per_pathology.values() if v.get('significant', False))
            n_total = len(per_pathology)
            print(f"    Summary: {n_sig}/{n_total} pathologies show significant differences")

            # Check critical pathologies specifically
            critical_equiv = all(
                not per_pathology.get(cp, {}).get('significant', True)
                for cp in CRITICAL_PATHOLOGIES
            )
            print(f"    Critical pathologies equivalent: {'YES' if critical_equiv else 'NO'}")

            mcnemar_results[label] = {
                'per_pathology': per_pathology,
                'n_significant': n_sig,
                'n_total': n_total,
                'critical_equivalent': critical_equiv,
                'threshold': threshold,
                'mode': mode_str,
            }

        self.results['mcnemar'] = mcnemar_results
        return mcnemar_results

    # ── Weight distribution histograms ──────────────────────────────────

    def plot_weight_distributions(self, layer_sensitivity_results=None,
                                  filename='weight_distributions.png'):
        """
        Visualise weight distributions for the most and least sensitive layers
        before and after INT8 quantisation.

        Uses layer sensitivity results to identify which layers to plot.
        Shows how quantisation distorts the weight distribution, providing
        visual evidence for why some layers are more sensitive.
        """
        if not HAS_MPL:
            return

        print("\n" + "=" * 60)
        print("WEIGHT DISTRIBUTION ANALYSIS")
        print("=" * 60)

        # Get layer sensitivity data
        if layer_sensitivity_results is None:
            layer_sensitivity_results = self.results.get('layer_sensitivity', {}).get('layers', [])

        if not layer_sensitivity_results:
            print("  No layer sensitivity data — run layer sensitivity analysis first")
            return

        # Find most and least sensitive layers
        valid = [r for r in layer_sensitivity_results if r.get('int8_mse') is not None]
        if len(valid) < 2:
            print("  Not enough layers with sensitivity data")
            return

        sorted_by_sens = sorted(valid, key=lambda r: r['int8_mse'], reverse=True)
        most_sensitive = sorted_by_sens[:3]
        least_sensitive = sorted_by_sens[-3:]
        layers_to_plot = most_sensitive + least_sensitive

        fig, axes = plt.subplots(len(layers_to_plot), 2, figsize=(12, 3 * len(layers_to_plot)))
        if len(layers_to_plot) == 1:
            axes = axes.reshape(1, -1)

        for row, layer_info in enumerate(layers_to_plot):
            layer_name = layer_info['layer']
            mse = layer_info['int8_mse']
            is_sensitive = row < len(most_sensitive)

            # Get original weights
            original_weights = None
            for name, module in self.model.named_modules():
                if name == layer_name and isinstance(module, nn.Conv2d):
                    original_weights = module.weight.data.cpu().numpy().flatten()
                    break

            if original_weights is None:
                continue

            # Quantise weights
            w = torch.tensor(original_weights)
            scale = w.abs().max() / 127.0
            if scale > 0:
                quantised = ((w / scale).round().clamp(-128, 127) * scale).numpy()
            else:
                quantised = original_weights.copy()

            # Plot original
            ax = axes[row, 0]
            ax.hist(original_weights, bins=100, alpha=0.7, color='#2E86AB', density=True)
            ax.set_title(f"{'SENSITIVE' if is_sensitive else 'ROBUST'}: {layer_name}\n"
                         f"Original FP32 (MSE={mse:.2e})", fontsize=8)
            ax.set_xlabel('Weight value', fontsize=7)
            ax.tick_params(labelsize=7)

            # Plot quantised overlay
            ax = axes[row, 1]
            ax.hist(original_weights, bins=100, alpha=0.4, color='#2E86AB',
                    density=True, label='FP32')
            ax.hist(quantised, bins=100, alpha=0.4, color='#F18F01',
                    density=True, label='INT8')
            ax.set_title(f"FP32 vs INT8 overlay", fontsize=8)
            ax.set_xlabel('Weight value', fontsize=7)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

        plt.suptitle('Weight Distributions: Most Sensitive vs Most Robust Layers',
                      fontweight='bold', fontsize=12)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()

        self.results['weight_distributions'] = {
            'most_sensitive_layers': [r['layer'] for r in most_sensitive],
            'least_sensitive_layers': [r['layer'] for r in least_sensitive],
        }

    # ── Energy measurement via RAPL ─────────────────────────────────────

    def measure_energy(self):
        """
        Estimate energy consumption per inference and derive carbon footprint.

        Method 1 (preferred): Intel RAPL via /sys/class/powercap/ — gives
        real measured energy in microjoules. Works on bare metal and some
        HPC VMs but NOT on standard Azure/AWS/GCP VMs (MSR registers
        are virtualised).

        Method 2 (fallback): TDP-based estimation. Uses the CPU's published
        TDP (Thermal Design Power) as an upper bound for package power, then
        estimates energy as: E = TDP × utilisation_factor × inference_time.
        This is an ESTIMATE, not a measurement — the utilisation factor
        accounts for the fact that inference rarely saturates all cores.

        Carbon footprint derived using UK grid carbon intensity
        (~233 gCO2/kWh as of 2024, source: National Grid ESO).
        """
        print("\n" + "=" * 60)
        print("ENERGY & CARBON FOOTPRINT ESTIMATION")
        print("=" * 60)

        # UK grid carbon intensity (gCO2/kWh) — source: National Grid ESO 2024
        UK_CARBON_INTENSITY = 233.0

        # Check for RAPL availability
        rapl_base = '/sys/class/powercap/intel-rapl'
        rapl_available = os.path.exists(rapl_base)
        use_rapl = False

        if rapl_available:
            # Verify we can actually read the counters
            energy_files = glob.glob(os.path.join(rapl_base, '*/energy_uj'))
            for f in energy_files:
                try:
                    with open(f, 'r') as fh:
                        int(fh.read().strip())
                    use_rapl = True
                    break
                except (IOError, ValueError, PermissionError):
                    continue

        if use_rapl:
            print("  Method: RAPL (direct energy measurement)")
        else:
            print("  Method: TDP-based estimation (RAPL not available in VM)")
            print("    Note: This is an upper-bound estimate, not a direct measurement.")

        # Detect CPU TDP for estimation fallback
        # AMD EPYC 9005 series TDP values (per-socket):
        #   9754  = 360W, 9654  = 360W, 9554  = 360W
        #   9374F = 320W, 9274F = 320W
        #   9175F = 320W, 9124 = 200W
        # Default to 360W if we can't identify the exact model.
        # For a 4-vCPU VM, the share is roughly TDP * (vCPUs / total_cores).
        cpu_tdp_w = 360.0  # default: EPYC 9005 full socket
        total_cores = 128   # default: EPYC 9754 has 128 cores
        n_vcpus = os.cpu_count() or 4

        # Try to read actual CPU model
        cpu_model = 'unknown'
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        cpu_model = line.split(':')[1].strip()
                        break
        except IOError:
            pass

        # Estimate per-vCPU power share
        # In a VM, your 4 vCPUs share the socket's TDP proportionally.
        # Utilisation factor ~0.7 for sustained inference (not 100% ALU busy).
        vcpu_power_share_w = cpu_tdp_w * (n_vcpus / total_cores)
        utilisation_factor = 0.7
        effective_power_w = vcpu_power_share_w * utilisation_factor

        print(f"    CPU: {cpu_model}")
        print(f"    Socket TDP: {cpu_tdp_w:.0f}W, vCPUs: {n_vcpus}/{total_cores}")
        print(f"    Estimated power draw: {effective_power_w:.1f}W "
              f"(= {cpu_tdp_w:.0f} × {n_vcpus}/{total_cores} × {utilisation_factor})")

        def read_rapl_energy_uj():
            energy_files = glob.glob(os.path.join(rapl_base, '*/energy_uj'))
            total = 0
            for f in energy_files:
                try:
                    with open(f, 'r') as fh:
                        total += int(fh.read().strip())
                except (IOError, ValueError):
                    pass
            return total

        energy_results = {
            'method': 'rapl' if use_rapl else 'tdp_estimate',
            'cpu_model': cpu_model,
            'socket_tdp_w': cpu_tdp_w,
            'n_vcpus': n_vcpus,
            'total_cores': total_cores,
            'utilisation_factor': utilisation_factor,
            'effective_power_w': effective_power_w,
            'carbon_intensity_gco2_kwh': UK_CARBON_INTENSITY,
        }

        # Use timing data we already have from precision comparison
        timing_data = {}
        if hasattr(self, '_times'):
            for label in ['FP32', 'BF16', 'INT8', 'Native_BF16', 'Static_INT8']:
                if label in self._times:
                    avg_ms = np.mean(self._times[label])
                    timing_data[label] = avg_ms

        # Also get native precision times from results
        native = self.results.get('native_precision', {})
        for label in ['Native_BF16', 'Native_FP16']:
            if label in native and label not in timing_data:
                timing_data[label] = native[label].get('avg_ms', 0)

        static = self.results.get('static_int8', {})
        if static and 'Static_INT8' not in timing_data:
            timing_data['Static_INT8'] = static.get('avg_ms', 0)

        if not timing_data:
            print("  No timing data available — run precision comparison first")
            self.results['energy'] = energy_results
            return

        print(f"\n  {'Precision':<20} {'ms/img':>8} {'J/img':>10} {'gCO2/img':>12} {'gCO2/1000':>12}")
        print(f"  {'-'*62}")

        for label, avg_ms in sorted(timing_data.items(), key=lambda x: x[1]):
            if avg_ms <= 0:
                continue
            time_s = avg_ms / 1000.0

            if use_rapl:
                # RAPL: measure a short burst
                model = copy.deepcopy(self.model).float().eval()
                n_measure = min(20, len(self.test_images))

                with torch.no_grad():
                    for img in self.test_images[:3]:
                        _ = model(img.float())

                energy_before = read_rapl_energy_uj()
                with torch.no_grad():
                    for img in self.test_images[:n_measure]:
                        _ = model(img.float())
                energy_after = read_rapl_energy_uj()

                total_j = (energy_after - energy_before) / 1e6
                joules_per_image = total_j / n_measure
            else:
                # TDP estimation: E = P × t
                joules_per_image = effective_power_w * time_s

            kwh_per_image = joules_per_image / 3600.0
            gco2_per_image = kwh_per_image * UK_CARBON_INTENSITY
            gco2_per_1000 = gco2_per_image * 1000

            energy_results[label] = {
                'avg_ms': avg_ms,
                'joules_per_image': joules_per_image,
                'gco2_per_image': gco2_per_image,
                'gco2_per_1000_images': gco2_per_1000,
            }

            print(f"  {label:<20} {avg_ms:>7.1f}ms {joules_per_image:>9.4f}J "
                  f"{gco2_per_image:>11.6f}g {gco2_per_1000:>11.4f}g")

        # Carbon savings summary
        if 'FP32' in energy_results and isinstance(energy_results['FP32'], dict):
            fp32_co2 = energy_results['FP32']['gco2_per_1000_images']
            print(f"\n  Carbon savings vs FP32 (per 1000 images):")
            for label in ['Native_BF16', 'Static_INT8', 'INT8']:
                if label in energy_results and isinstance(energy_results[label], dict):
                    other_co2 = energy_results[label]['gco2_per_1000_images']
                    saving_pct = (1 - other_co2 / fp32_co2) * 100 if fp32_co2 > 0 else 0
                    print(f"    {label:<20} {saving_pct:>+.1f}% ({fp32_co2 - other_co2:.4f}g saved)")

        self.results['energy'] = energy_results

    # ── NUMA topology detection ──────────────────────────────────────────

    def detect_numa_topology(self):
        """
        Read NUMA topology from /sys/devices/system/node/ and report it.

        On AMD EPYC's chiplet architecture, vCPUs may span multiple NUMA
        nodes (CCDs / CCXs). Cross-node memory access adds ~50ns latency,
        which explains why 2 threads (likely on the same CCX) outperform
        4 or 8 threads (which may span CCXs/CCDs, incurring coherence overhead).

        This is Linux-specific and may not work in all VM environments.
        """
        print("\n" + "=" * 60)
        print("NUMA TOPOLOGY")
        print("=" * 60)

        numa_info = {'available': False}

        # Check /sys/devices/system/node/
        node_base = '/sys/devices/system/node'
        if not os.path.exists(node_base):
            print("  NUMA info not available (no /sys/devices/system/node/)")
            self.results['numa'] = numa_info
            return numa_info

        # Count NUMA nodes
        nodes = sorted([d for d in os.listdir(node_base) if d.startswith('node')])
        numa_info['n_nodes'] = len(nodes)
        numa_info['available'] = True
        print(f"  NUMA nodes: {len(nodes)}")

        # Read CPUs per node
        node_cpus = {}
        for node in nodes:
            cpulist_path = os.path.join(node_base, node, 'cpulist')
            try:
                with open(cpulist_path, 'r') as f:
                    cpulist = f.read().strip()
                node_cpus[node] = cpulist
                print(f"    {node}: CPUs {cpulist}")
            except (IOError, PermissionError):
                pass

        numa_info['node_cpus'] = node_cpus

        # Read inter-node distances
        distance_path = os.path.join(node_base, nodes[0], 'distance') if nodes else None
        if distance_path and os.path.exists(distance_path):
            try:
                with open(distance_path, 'r') as f:
                    distances = f.read().strip()
                print(f"    Inter-node distances: {distances}")
                numa_info['distances'] = distances
            except (IOError, PermissionError):
                pass

        # Read CPU cache topology
        try:
            cache_info = {}
            cpu0_cache = '/sys/devices/system/cpu/cpu0/cache'
            if os.path.exists(cpu0_cache):
                for idx in sorted(os.listdir(cpu0_cache)):
                    idx_path = os.path.join(cpu0_cache, idx)
                    if not os.path.isdir(idx_path):
                        continue
                    level_path = os.path.join(idx_path, 'level')
                    size_path = os.path.join(idx_path, 'size')
                    type_path = os.path.join(idx_path, 'type')
                    shared_path = os.path.join(idx_path, 'shared_cpu_list')
                    try:
                        level = open(level_path).read().strip()
                        size = open(size_path).read().strip()
                        ctype = open(type_path).read().strip()
                        shared = open(shared_path).read().strip() if os.path.exists(shared_path) else 'N/A'
                        cache_info[f'L{level}_{ctype}'] = {
                            'size': size, 'shared_cpus': shared
                        }
                        print(f"    L{level} {ctype}: {size} (shared: CPUs {shared})")
                    except (IOError, ValueError):
                        pass
                numa_info['cache'] = cache_info
        except Exception:
            pass

        # Interpret for the thread sweep
        n_vcpus = os.cpu_count() or 4
        if len(nodes) > 1:
            print(f"\n  Interpretation: {n_vcpus} vCPUs across {len(nodes)} NUMA nodes.")
            print(f"  Thread counts that keep work on a single NUMA node (and L3 domain)")
            print(f"  will avoid cross-node coherence traffic, explaining why 2 threads")
            print(f"  outperform 4+ on EPYC chiplet architecture.")
        elif len(nodes) == 1:
            print(f"\n  Single NUMA node detected — VM's {n_vcpus} vCPUs share one L3 domain.")
            print(f"  Thread scaling limited by L3 bandwidth and core contention.")

        self.results['numa'] = numa_info
        return numa_info

    # ── Thread count sweep ──────────────────────────────────────────────

    def thread_count_sweep(self, thread_counts=None):
        """
        Measure latency and throughput at different thread counts for
        FP32 and INT8 to find optimal parallelism on the server CPU.

        Args:
            thread_counts: list of thread counts to test
                           (default: [1, 2, 4, 8, 16, max_cores])
        """
        print("\n" + "=" * 60)
        print("THREAD COUNT SWEEP")
        print("=" * 60)

        max_cores = os.cpu_count() or 4
        if thread_counts is None:
            thread_counts = sorted(set([1, 2, 4, 8, min(16, max_cores), max_cores]))

        original_threads = torch.get_num_threads()
        subset = self.test_images[:min(100, len(self.test_images))]
        sweep_results = {}

        for n_threads in thread_counts:
            torch.set_num_threads(n_threads)
            print(f"\n  Threads: {n_threads}")
            sweep_results[n_threads] = {}

            for prec_name in ['FP32', 'INT8']:
                model = copy.deepcopy(self.model).float().eval()

                if prec_name == 'INT8':
                    # Simulate INT8 weights
                    for param in model.parameters():
                        if param.dim() >= 2:
                            scale = param.abs().max() / 127.0
                            if scale > 0:
                                param.data = (param.data / scale).round().clamp(-128, 127) * scale

                # Warmup
                with torch.no_grad():
                    for img in subset[:5]:
                        _ = model(img.float())

                # Benchmark
                times = []
                with torch.no_grad():
                    for img in subset:
                        start = time.perf_counter()
                        _ = model(img.float())
                        elapsed = (time.perf_counter() - start) * 1000
                        times.append(elapsed)

                avg_ms = np.mean(times)
                throughput = 1000.0 / avg_ms if avg_ms > 0 else 0

                sweep_results[n_threads][prec_name] = {
                    'avg_ms': avg_ms,
                    'std_ms': float(np.std(times)),
                    'throughput_ips': throughput,
                }
                print(f"    {prec_name}: {avg_ms:.1f} ms  ({throughput:.1f} img/s)")

        # Restore original thread count
        torch.set_num_threads(original_threads)

        self.results['thread_sweep'] = sweep_results

    # ── Batch size sweep ────────────────────────────────────────────────

    def batch_size_sweep(self, batch_sizes=None):
        """
        Measure latency and throughput at different batch sizes for
        FP32 vs INT8 to see how speedup changes with batching.

        Args:
            batch_sizes: list of batch sizes (default: [1, 4, 16, 32])
        """
        print("\n" + "=" * 60)
        print("BATCH SIZE SWEEP")
        print("=" * 60)

        if batch_sizes is None:
            batch_sizes = [1, 4, 16, 32]

        n_images = min(64, len(self.test_images))
        images = [img for img in self.test_images[:n_images]]
        sweep_results = {}

        for bs in batch_sizes:
            print(f"\n  Batch size: {bs}")
            sweep_results[bs] = {}

            # Create batches
            batches = []
            for i in range(0, n_images, bs):
                batch_imgs = images[i:i + bs]
                if batch_imgs:
                    batches.append(torch.cat(batch_imgs, dim=0))

            if not batches:
                continue

            for prec_name in ['FP32', 'INT8']:
                model = copy.deepcopy(self.model).float().eval()

                if prec_name == 'INT8':
                    for param in model.parameters():
                        if param.dim() >= 2:
                            scale = param.abs().max() / 127.0
                            if scale > 0:
                                param.data = (param.data / scale).round().clamp(-128, 127) * scale

                # Warmup
                with torch.no_grad():
                    _ = model(batches[0].float())

                # Benchmark
                times = []
                with torch.no_grad():
                    for batch in batches:
                        start = time.perf_counter()
                        _ = model(batch.float())
                        elapsed = (time.perf_counter() - start) * 1000
                        times.append(elapsed)

                avg_ms = np.mean(times)
                per_image_ms = avg_ms / bs
                throughput = 1000.0 / per_image_ms if per_image_ms > 0 else 0

                sweep_results[bs][prec_name] = {
                    'avg_batch_ms': avg_ms,
                    'per_image_ms': per_image_ms,
                    'throughput_ips': throughput,
                }
                print(f"    {prec_name}: {avg_ms:.1f} ms/batch  "
                      f"({per_image_ms:.1f} ms/img, {throughput:.1f} img/s)")

        # Compute speedup ratios
        for bs in batch_sizes:
            if bs in sweep_results:
                fp32 = sweep_results[bs].get('FP32', {})
                int8 = sweep_results[bs].get('INT8', {})
                if fp32 and int8 and int8.get('per_image_ms', 0) > 0:
                    sweep_results[bs]['speedup'] = fp32['per_image_ms'] / int8['per_image_ms']
                    print(f"  Batch {bs}: INT8 speedup = {sweep_results[bs]['speedup']:.2f}x")

        self.results['batch_sweep'] = sweep_results

    # ── Latency drift plot ──────────────────────────────────────────────

    def plot_latency_drift(self, filename='latency_drift.png'):
        """
        Plot per-image inference times across the full test set to reveal
        any vCPU scheduling effects, thermal throttling, or cache behaviour.

        Shows latency over time for FP32, INT8, and native BF16 if available.
        """
        if not HAS_MPL:
            return

        print("\n" + "=" * 60)
        print("LATENCY DRIFT ANALYSIS")
        print("=" * 60)

        if not hasattr(self, '_times'):
            print("  No timing data — run precision comparison first")
            return

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # Plot 1: Raw latency over time
        ax = axes[0]
        for label in ['FP32', 'INT8', 'Native_BF16', 'Static_INT8']:
            if label in self._times:
                times = self._times[label]
                ax.plot(range(len(times)), times, alpha=0.5, linewidth=0.5,
                        color=COLOURS.get(label, '#888888'), label=label)
                # Rolling average
                window = min(50, len(times) // 5)
                if window > 1:
                    rolling = np.convolve(times, np.ones(window) / window, mode='valid')
                    ax.plot(range(window - 1, len(times)), rolling, linewidth=2,
                            color=COLOURS.get(label, '#888888'))

        ax.set_xlabel('Image index')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('(a) Per-Image Inference Latency Over Time (thin=raw, thick=rolling avg)')
        ax.legend(fontsize=8)

        # Plot 2: Distribution comparison (violin or box)
        ax = axes[1]
        data_to_plot = []
        labels_to_plot = []
        colours_to_plot = []
        for label in ['FP32', 'BF16', 'FP16', 'INT8', 'Native_BF16', 'Static_INT8']:
            if label in self._times:
                data_to_plot.append(self._times[label])
                labels_to_plot.append(label.replace('_', '\n'))
                colours_to_plot.append(COLOURS.get(label, '#888888'))

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                            showfliers=False, whis=[5, 95])
            for patch, color in zip(bp['boxes'], colours_to_plot):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_ylabel('Latency (ms)')
            ax.set_title('(b) Latency Distribution (5th-95th percentile, no outliers)')

        plt.suptitle('Latency Drift and Distribution Analysis', fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()

    # ── ONNX Runtime comparison ──────────────────────────────────────────

    def run_onnx_runtime_comparison(self):
        """
        Export model to ONNX and benchmark via ONNX Runtime's INT8 path.

        Provides a second inference engine comparison alongside PyTorch's
        fbgemm, which is practically useful: hospital IT teams choosing
        a deployment stack need to know which runtime to pick.

        ONNX Runtime uses its own optimised INT8 kernels that may differ
        from fbgemm. On x86 with AVX-512, ORT uses MLAS (Microsoft Linear
        Algebra Subroutines) which has separate VNNI code paths.
        """
        print("\n" + "=" * 60)
        print("ONNX RUNTIME COMPARISON")
        print("=" * 60)

        if not HAS_ORT:
            print("  SKIPPED — onnxruntime not installed")
            print("  Install with: pip install onnxruntime")
            return None

        import tempfile

        model_fp32 = copy.deepcopy(self.model).float().eval()
        dummy_input = self.test_images[0].float()

        # Export to ONNX
        onnx_path = os.path.join(tempfile.gettempdir(), 'densenet121_cxr.onnx')
        print(f"  Exporting to ONNX...")
        try:
            torch.onnx.export(
                model_fp32, dummy_input, onnx_path,
                input_names=['input'], output_names=['output'],
                dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
                opset_version=17,
                do_constant_folding=True,
            )
            onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"  ONNX model: {onnx_size_mb:.1f} MB")
        except Exception as e:
            print(f"  ONNX export failed: {e}")
            return None

        ort_results = {}

        # FP32 ONNX Runtime baseline
        print(f"  Benchmarking ONNX Runtime FP32...")
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = torch.get_num_threads()
            sess_opts.inter_op_num_threads = 1
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            session = ort.InferenceSession(onnx_path, sess_opts,
                                            providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name

            # Warmup
            for img in self.test_images[:30]:
                session.run(None, {input_name: img.float().numpy()})

            # Benchmark
            times = []
            outputs = []
            for img in self.test_images:
                inp = img.float().numpy()
                start = time.perf_counter()
                out = session.run(None, {input_name: inp})
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                outputs.append(torch.tensor(out[0]))

            all_outputs = torch.cat(outputs, dim=0)
            avg_ms = np.mean(times)
            mean_est, ci_lo, ci_hi = StatisticalAnalysis.bootstrap_ci(times)
            throughput = 1000.0 / avg_ms if avg_ms > 0 else 0

            print(f"  ORT FP32: {avg_ms:.1f} +/- {np.std(times):.1f} ms  "
                  f"[95% CI: {ci_lo:.1f} - {ci_hi:.1f}]  throughput: {throughput:.1f} img/s")

            ort_results['ORT_FP32'] = {
                'avg_ms': avg_ms, 'std_ms': float(np.std(times)),
                'ci_lower': ci_lo, 'ci_upper': ci_hi,
                'throughput_ips': throughput,
            }
            self._outputs['ORT_FP32'] = all_outputs
            self._times['ORT_FP32'] = times

        except Exception as e:
            print(f"  ORT FP32 benchmark failed: {e}")

        # INT8 dynamic quantisation via ORT
        print(f"  Benchmarking ONNX Runtime Dynamic INT8...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            int8_onnx_path = onnx_path.replace('.onnx', '_int8.onnx')
            quantize_dynamic(
                onnx_path, int8_onnx_path,
                weight_type=QuantType.QInt8,
            )
            int8_size_mb = os.path.getsize(int8_onnx_path) / (1024 * 1024)
            print(f"  ORT INT8 model: {int8_size_mb:.1f} MB")

            session_int8 = ort.InferenceSession(int8_onnx_path, sess_opts,
                                                  providers=['CPUExecutionProvider'])
            input_name = session_int8.get_inputs()[0].name

            # Warmup
            for img in self.test_images[:30]:
                session_int8.run(None, {input_name: img.float().numpy()})

            # Benchmark
            times = []
            outputs = []
            for img in self.test_images:
                inp = img.float().numpy()
                start = time.perf_counter()
                out = session_int8.run(None, {input_name: inp})
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                outputs.append(torch.tensor(out[0]))

            all_outputs = torch.cat(outputs, dim=0)
            avg_ms = np.mean(times)
            mean_est, ci_lo, ci_hi = StatisticalAnalysis.bootstrap_ci(times)
            throughput = 1000.0 / avg_ms if avg_ms > 0 else 0
            fp32_time = self.results.get('precision', {}).get('FP32', {}).get('avg_ms', 1)
            speedup = fp32_time / avg_ms if avg_ms > 0 else 0

            # Accuracy vs PyTorch FP32
            if 'FP32' in self._outputs:
                fp32_probs = torch.sigmoid(self._outputs['FP32'])
                ort_probs = torch.sigmoid(all_outputs)
                diff = torch.abs(ort_probs - fp32_probs)
                mean_diff_pct = float(diff.mean()) * 100
                max_diff_pct = float(diff.max()) * 100
            else:
                mean_diff_pct = 0
                max_diff_pct = 0

            print(f"  ORT INT8: {avg_ms:.1f} +/- {np.std(times):.1f} ms  "
                  f"[95% CI: {ci_lo:.1f} - {ci_hi:.1f}]  throughput: {throughput:.1f} img/s")
            print(f"    Speedup vs PyTorch FP32: {speedup:.2f}x")
            print(f"    Mean prob diff vs PyTorch FP32: {mean_diff_pct:.4f}%")
            print(f"    Max prob diff vs PyTorch FP32:  {max_diff_pct:.4f}%")

            ort_results['ORT_INT8'] = {
                'avg_ms': avg_ms, 'std_ms': float(np.std(times)),
                'ci_lower': ci_lo, 'ci_upper': ci_hi,
                'throughput_ips': throughput,
                'speedup_vs_pytorch_fp32': speedup,
                'mean_diff_pct': mean_diff_pct,
                'max_diff_pct': max_diff_pct,
                'onnx_size_mb': int8_size_mb,
            }
            self._outputs['ORT_INT8'] = all_outputs
            self._times['ORT_INT8'] = times

            # Clean up temp files
            for p in [onnx_path, int8_onnx_path]:
                try:
                    os.remove(p)
                except OSError:
                    pass

        except ImportError:
            print("  ORT quantisation not available — install onnxruntime-extensions")
        except Exception as e:
            print(f"  ORT INT8 benchmark failed: {e}")

        self.results['onnx_runtime'] = ort_results
        return ort_results

    # ── Tail latency analysis ─────────────────────────────────────────────

    def analyse_tail_latencies(self):
        """
        Compute and report tail latencies (p50, p90, p95, p99) for each
        precision format.

        For clinical deployment, worst-case latency determines SLA compliance.
        A system with great mean latency but a fat p99 tail is unreliable
        in practice — a radiologist waiting 200ms 1% of the time is worse
        than consistent 30ms.
        """
        print("\n" + "=" * 60)
        print("TAIL LATENCY ANALYSIS")
        print("=" * 60)

        if not hasattr(self, '_times'):
            print("  No timing data — run precision comparison first")
            return

        percentiles = [50, 90, 95, 99]
        tail_results = {}

        print(f"\n  {'Precision':<20} {'p50':>8} {'p90':>8} {'p95':>8} {'p99':>8} {'max':>8} {'jitter':>8}")
        print(f"  {'-'*68}")

        for label in ['FP32', 'BF16', 'FP16', 'INT8', 'Dynamic_INT8',
                       'Native_BF16', 'Static_INT8']:
            if label not in self._times:
                continue
            times = np.array(self._times[label])
            pvals = {f'p{p}': float(np.percentile(times, p)) for p in percentiles}
            pvals['max'] = float(np.max(times))
            pvals['jitter'] = float(np.percentile(times, 99) - np.percentile(times, 1))
            tail_results[label] = pvals

            print(f"  {label:<20} {pvals['p50']:>7.1f}ms {pvals['p90']:>7.1f}ms "
                  f"{pvals['p95']:>7.1f}ms {pvals['p99']:>7.1f}ms "
                  f"{pvals['max']:>7.1f}ms {pvals['jitter']:>7.1f}ms")

        self.results['tail_latencies'] = tail_results
        return tail_results

    def plot_latency_histogram(self, filename='latency_histogram.png'):
        """
        Plot latency distribution histogram and CDF for each precision format.

        Shows the full distribution shape with vertical lines at p95/p99,
        which is more informative than just mean +/- std for understanding
        tail behaviour and SLA compliance.
        """
        if not HAS_MPL:
            return

        if not hasattr(self, '_times'):
            return

        precisions_to_plot = [l for l in ['FP32', 'INT8', 'Native_BF16', 'Static_INT8']
                              if l in self._times]
        if not precisions_to_plot:
            return

        n = len(precisions_to_plot)
        fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for row, label in enumerate(precisions_to_plot):
            times = np.array(self._times[label])
            colour = COLOURS.get(label, '#888888')

            # Histogram
            ax = axes[row, 0]
            ax.hist(times, bins=50, alpha=0.7, color=colour, density=True, edgecolor='white')
            p95 = np.percentile(times, 95)
            p99 = np.percentile(times, 99)
            ax.axvline(p95, color='orange', linestyle='--', linewidth=1.5, label=f'p95={p95:.1f}ms')
            ax.axvline(p99, color='red', linestyle='--', linewidth=1.5, label=f'p99={p99:.1f}ms')
            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel('Density')
            ax.set_title(f'{label} — Latency Distribution')
            ax.legend(fontsize=8)

            # CDF
            ax = axes[row, 1]
            sorted_times = np.sort(times)
            cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
            ax.plot(sorted_times, cdf * 100, color=colour, linewidth=2)
            ax.axhline(95, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='95th %ile')
            ax.axhline(99, color='red', linestyle='--', linewidth=1, alpha=0.7, label='99th %ile')
            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel('Cumulative %')
            ax.set_title(f'{label} — CDF')
            ax.legend(fontsize=8)
            ax.set_ylim(0, 101)

        plt.suptitle('Latency Distribution & Tail Analysis (SLA Compliance)',
                      fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()

    # ── Distance-based connection analysis ───────────────────────────────

    def analyse_distance_connections(self):
        """
        Count dense connections and assign precision by distance.

        DenseNet architecture: each layer receives input from ALL previous
        layers in the same block. This creates many redundant connections
        at large distances.
        """
        print("\n" + "=" * 60)
        print("DISTANCE-BASED PRECISION ASSIGNMENT")
        print("=" * 60)

        blocks = {}
        for name, _ in self.model.named_modules():
            if 'denseblock' in name and 'denselayer' in name:
                for part in name.split('.'):
                    if 'denseblock' in part:
                        block_num = int(part.replace('denseblock', ''))
                        if block_num not in blocks:
                            blocks[block_num] = set()
                    if 'denselayer' in part:
                        layer_num = int(part.replace('denselayer', ''))
                        blocks[block_num].add(layer_num)

        print("\n  DenseNet-121 structure:")
        for num in sorted(blocks.keys()):
            print(f"    Block {num}: {len(blocks[num])} layers")

        total = fp32_count = fp16_count = int8_count = 0
        distance_distribution = defaultdict(int)

        for block_num, layers in blocks.items():
            n = len(layers)
            for dest in range(2, n + 1):
                for src in range(1, dest):
                    distance = dest - src
                    total += 1
                    distance_distribution[distance] += 1
                    if distance == 1:
                        fp32_count += 1
                    elif distance <= 3:
                        fp16_count += 1
                    else:
                        int8_count += 1

        print(f"\n  Total dense connections: {total}")
        print(f"\n  Distance-based assignment:")
        print(f"    FP32 (distance 1):   {fp32_count} ({fp32_count / total * 100:.1f}%)")
        print(f"    FP16 (distance 2-3): {fp16_count} ({fp16_count / total * 100:.1f}%)")
        print(f"    INT8 (distance 4+):  {int8_count} ({int8_count / total * 100:.1f}%)")

        uniform_bytes = total * 4
        mixed_bytes = fp32_count * 4 + fp16_count * 2 + int8_count * 1
        savings = (1 - mixed_bytes / uniform_bytes) * 100
        print(f"\n  Theoretical memory savings: {savings:.1f}%")

        self.results['connections'] = {
            'total': total, 'fp32': fp32_count, 'fp16': fp16_count,
            'int8': int8_count, 'memory_savings_pct': savings,
            'distance_distribution': dict(distance_distribution),
            'blocks': {k: len(v) for k, v in blocks.items()}
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FINE-TUNING & QAT
# ═══════════════════════════════════════════════════════════════════════════════

class FineTuner:
    """
    Fine-tune DenseNet-121 on ChestX-ray14 for meaningful AUC-ROC values.

    Without fine-tuning, the pretrained ImageNet model produces near-chance
    AUCs (~0.5) on chest X-ray pathologies. Fine-tuning for even a few
    epochs on the real labels gives the model actual diagnostic ability,
    making the precision equivalence argument far more compelling.

    Uses a simple training loop with BCE loss (multi-label classification),
    Adam optimiser, and cosine annealing LR schedule.
    """

    def __init__(self, model, train_images, train_labels, pathology_names=None,
                 val_split=0.2, lr=1e-4, epochs=5, batch_size=16):
        self.model = copy.deepcopy(model).float()
        self.train_images = train_images
        self.train_labels = train_labels  # (N, C) numpy array
        self.pathology_names = pathology_names or PATHOLOGIES
        self.val_split = val_split
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def run(self):
        """
        Fine-tune the model and return the fine-tuned state_dict.

        Returns:
            dict with 'model' (state_dict), 'train_aucs', 'val_aucs',
            'train_losses', 'val_losses'
        """
        print("\n" + "=" * 60)
        print("FINE-TUNING DenseNet-121 on ChestX-ray14")
        print("=" * 60)

        n = len(self.train_images)
        n_val = int(n * self.val_split)
        n_train = n - n_val

        # Deterministic split
        rng = np.random.RandomState(42)
        indices = rng.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        print(f"  Train: {n_train} images, Val: {n_val} images")
        print(f"  Epochs: {self.epochs}, LR: {self.lr}, Batch: {self.batch_size}")
        print(f"  Pathologies: {len(self.pathology_names)}")

        self.model.train()

        # Only fine-tune the classifier and last dense block + norm5
        # Freeze earlier layers to avoid catastrophic forgetting
        for name, param in self.model.named_parameters():
            if any(k in name for k in ['classifier', 'denseblock4', 'norm5', 'transition3']):
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable parameters: {trainable:,} / {total:,} ({trainable / total * 100:.1f}%)")

        optimiser = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.epochs)
        criterion = nn.BCEWithLogitsLoss()

        history = {'train_loss': [], 'val_loss': [], 'val_aucs': []}

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            rng.shuffle(train_idx)
            train_loss = 0
            n_batches = 0

            for i in range(0, n_train, self.batch_size):
                batch_idx = train_idx[i:i + self.batch_size]
                batch_imgs = torch.cat([self.train_images[j] for j in batch_idx], dim=0)
                batch_labels = torch.tensor(self.train_labels[batch_idx], dtype=torch.float32)

                # torchxrayvision outputs 18 classes; we use first 14
                logits = self.model(batch_imgs.float())[:, :len(self.pathology_names)]

                loss = criterion(logits, batch_labels)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                train_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_train_loss = train_loss / max(n_batches, 1)
            history['train_loss'].append(avg_train_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_labels_list = []
            n_val_batches = 0

            with torch.no_grad():
                for i in range(0, n_val, self.batch_size):
                    batch_idx = val_idx[i:i + self.batch_size]
                    batch_imgs = torch.cat([self.train_images[j] for j in batch_idx], dim=0)
                    batch_labels = torch.tensor(self.train_labels[batch_idx], dtype=torch.float32)

                    logits = self.model(batch_imgs.float())[:, :len(self.pathology_names)]
                    loss = criterion(logits, batch_labels)
                    val_loss += loss.item()
                    n_val_batches += 1

                    val_preds.append(torch.sigmoid(logits).cpu().numpy())
                    val_labels_list.append(batch_labels.numpy())

            avg_val_loss = val_loss / max(n_val_batches, 1)
            history['val_loss'].append(avg_val_loss)

            # Compute per-pathology AUC on validation set
            val_preds_all = np.concatenate(val_preds, axis=0)
            val_labels_all = np.concatenate(val_labels_list, axis=0)

            epoch_aucs = {}
            if HAS_SKLEARN:
                for j, name in enumerate(self.pathology_names):
                    if j < val_preds_all.shape[1]:
                        y_true = val_labels_all[:, j]
                        if len(np.unique(y_true)) >= 2:
                            try:
                                epoch_aucs[name] = float(roc_auc_score(y_true, val_preds_all[:, j]))
                            except Exception:
                                epoch_aucs[name] = None
                        else:
                            epoch_aucs[name] = None

            history['val_aucs'].append(epoch_aucs)
            mean_auc = np.mean([v for v in epoch_aucs.values() if v is not None]) if epoch_aucs else 0

            # Report critical pathology AUCs
            crit_strs = []
            for cp in CRITICAL_PATHOLOGIES:
                auc_val = epoch_aucs.get(cp)
                if auc_val is not None:
                    crit_strs.append(f"{cp[:5]}={auc_val:.3f}")
            crit_report = "  ".join(crit_strs) if crit_strs else ""

            print(f"  Epoch {epoch + 1}/{self.epochs}  "
                  f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  "
                  f"mean_AUC={mean_auc:.3f}  {crit_report}")

        # Final model
        self.model.eval()
        print(f"\n  Fine-tuning complete. Final mean AUC: {mean_auc:.3f}")

        return {
            'model_state_dict': self.model.state_dict(),
            'model': self.model,
            'history': history,
        }


class QATTrainer:
    """
    Quantization-Aware Training (QAT) for static INT8.

    Post-training quantisation (PTQ) with only calibration produces
    ~70% max probability difference (clinically unusable). QAT inserts
    fake quantisation nodes during training so the model learns to be
    robust to INT8 rounding, closing the accuracy gap.

    Key design: uses the SAME DenseNetQuantWrapper + patched forwards
    as the static INT8 path (see _try_eager_quantisation). The generic
    QuantWrapper fails because torchxrayvision's DenseNet forward mixes
    quantised (QUInt8) and float tensors in ways that trigger PyTorch's
    "promoteTypes with quantized numbers is not handled yet" error.

    The DenseNet-aware wrapper solves this by:
      - Running Conv2d in INT8 (via QuantStub/DeQuantStub boundaries)
      - Keeping BatchNorm + ReLU in float (no qconfig)
      - Using FloatFunctional.cat() for dense connections
      - Keeping norm5 + classifier in float (after DeQuantStub)

    With QAT, the fake-quantize modules inserted by prepare_qat()
    simulate INT8 rounding during the forward pass, so backprop learns
    weight values that are robust to quantisation noise.
    """

    def __init__(self, model, train_images, train_labels, pathology_names=None,
                 lr=1e-5, epochs=3, batch_size=16, n_calib=500):
        self.model = copy.deepcopy(model).float()
        self.train_images = train_images
        self.train_labels = train_labels
        self.pathology_names = pathology_names or PATHOLOGIES
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_calib = n_calib

    def _build_qat_model(self):
        """
        Build DenseNet-aware QAT model using the same eager-mode patching
        as _try_eager_quantisation, but with prepare_qat instead of prepare.

        Returns the wrapped model ready for QAT training, or None on failure.
        """
        import types
        from torch.ao.nn.quantized import FloatFunctional

        model = copy.deepcopy(self.model).float()
        model.eval()  # eval for wrapping, will switch to train later

        # ── Step 1: Wrap with DenseNetQuantWrapper ──
        wrapped = DenseNetQuantWrapper(model)

        # ── Step 2: Patch DenseLayer forwards ──
        # Each DenseLayer: norm1 → relu1 → conv1 → norm2 → relu2 → conv2
        # Conv runs in INT8, BN + ReLU stay float, connected via stubs.
        patched_layers = 0

        for block_name in ['denseblock1', 'denseblock2', 'denseblock3', 'denseblock4']:
            block = getattr(wrapped.features_body, block_name, None)
            if block is None:
                continue
            for layer_name, layer_module in block.named_children():
                if 'DenseLayer' not in type(layer_module).__name__:
                    continue

                layer_module.add_module('_dq_pre', torch.quantization.DeQuantStub())
                layer_module.add_module('_q_mid1', torch.quantization.QuantStub())
                layer_module.add_module('_dq_mid', torch.quantization.DeQuantStub())
                layer_module.add_module('_q_mid2', torch.quantization.QuantStub())
                layer_module.add_module('_cat_fn', FloatFunctional())

                def _make_patched_forward(drop_rate):
                    def _patched_forward(self, x):
                        h = self._dq_pre(x)
                        h = self.norm1(h)
                        h = self.relu1(h)
                        h = self._q_mid1(h)
                        h = self.conv1(h)
                        h = self._dq_mid(h)
                        h = self.norm2(h)
                        h = self.relu2(h)
                        h = self._q_mid2(h)
                        new_features = self.conv2(h)
                        if drop_rate > 0:
                            new_features = F.dropout(
                                new_features, p=drop_rate, training=self.training
                            )
                        return self._cat_fn.cat([x, new_features], dim=1)
                    return _patched_forward

                layer_module.forward = types.MethodType(
                    _make_patched_forward(layer_module.drop_rate),
                    layer_module
                )
                patched_layers += 1

        print(f"    Patched {patched_layers} DenseLayer forwards for QAT")

        # ── Step 2b: Patch Transition forwards ──
        for i in [1, 2, 3]:
            tname = f'transition{i}'
            trans = getattr(wrapped.features_body, tname, None)
            if trans is None:
                continue

            trans.add_module('_dq_pre', torch.quantization.DeQuantStub())
            trans.add_module('_q_mid', torch.quantization.QuantStub())

            def _make_trans_forward():
                def _trans_forward(self, x):
                    h = self._dq_pre(x)
                    h = self.norm(h)
                    h = self.relu(h)
                    h = self._q_mid(h)
                    h = self.conv(h)
                    h = self.pool(h)
                    return h
                return _trans_forward

            trans.forward = types.MethodType(
                _make_trans_forward(), trans
            )

        print(f"    Patched 3 Transition forwards for QAT")

        # ── Step 3: Set QAT qconfig and propagate ──
        qat_qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        wrapped.qconfig = qat_qconfig
        torch.quantization.propagate_qconfig_(wrapped)

        # Disable qconfig on modules that must stay float
        wrapped.norm5.qconfig = None
        wrapped.classifier.qconfig = None

        for block_name in ['denseblock1', 'denseblock2', 'denseblock3', 'denseblock4']:
            block = getattr(wrapped.features_body, block_name, None)
            if block is None:
                continue
            for layer_name, layer_module in block.named_children():
                if 'DenseLayer' not in type(layer_module).__name__:
                    continue
                for attr in ['norm1', 'relu1', 'norm2', 'relu2']:
                    sub = getattr(layer_module, attr, None)
                    if sub is not None:
                        sub.qconfig = None

        for i in [1, 2, 3]:
            trans = getattr(wrapped.features_body, f'transition{i}', None)
            if trans is None:
                continue
            for attr in ['norm', 'relu']:
                sub = getattr(trans, attr, None)
                if sub is not None:
                    sub.qconfig = None

        # ── Step 4: Fuse stem conv+bn+relu ──
        try:
            torch.quantization.fuse_modules(
                wrapped.features_body,
                [['conv0', 'norm0', 'relu0']],
                inplace=True
            )
            print("    Fused stem conv+bn+relu")
        except Exception as e:
            print(f"    Stem fusion failed (non-fatal): {e}")

        # ── Step 5: prepare_qat ──
        torch.quantization.prepare_qat(wrapped, inplace=True)
        print("    QAT preparation successful (DenseNet-aware eager mode)")

        return wrapped

    def run(self):
        """
        Run QAT and return the quantised INT8 model.

        Returns:
            dict with 'model' (quantised model), 'avg_ms', 'max_diff_pct', etc.
        """
        print("\n" + "=" * 60)
        print("QUANTIZATION-AWARE TRAINING (QAT)")
        print("=" * 60)

        torch.backends.quantized.engine = 'fbgemm'
        print("  Building DenseNet-aware QAT model...")

        try:
            wrapped = self._build_qat_model()
        except Exception as e:
            print(f"  QAT model build failed: {e}")
            return None

        if wrapped is None:
            return None

        # ── QAT training loop ──
        print(f"\n  Training: {self.epochs} epochs, LR={self.lr}, batch={self.batch_size}")
        criterion = nn.BCEWithLogitsLoss()
        optimiser = torch.optim.Adam(
            [p for p in wrapped.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=1e-5
        )

        n = len(self.train_images)
        rng = np.random.RandomState(42)
        indices = np.arange(n)

        for epoch in range(self.epochs):
            wrapped.train()
            rng.shuffle(indices)
            epoch_loss = 0
            n_batches = 0

            for i in range(0, n, self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                batch_imgs = torch.cat([self.train_images[j] for j in batch_idx], dim=0)
                batch_labels = torch.tensor(
                    self.train_labels[batch_idx], dtype=torch.float32
                )

                # DenseNetQuantWrapper outputs 18 classes; take first 14
                logits = wrapped(batch_imgs.float())[:, :len(self.pathology_names)]
                loss = criterion(logits, batch_labels)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  QAT Epoch {epoch + 1}/{self.epochs}  loss={avg_loss:.4f}")

        # ── Convert to true INT8 ──
        print("  Converting QAT model to INT8...")
        wrapped.eval()
        try:
            model_int8 = torch.quantization.convert(wrapped)
            print("  QAT conversion successful")
        except Exception as e:
            print(f"  QAT conversion failed: {e}")
            return None

        # Count quantised layers
        n_quantized = sum(
            1 for m in model_int8.modules()
            if type(m).__module__.startswith('torch.ao.nn.quantized')
        )
        print(f"  Quantised layers: {n_quantized}")

        # Sanity check
        try:
            with torch.no_grad():
                test_out = model_int8(self.train_images[0].float())
            print(f"  Sanity check passed — output shape: {test_out.shape}")
        except Exception as e:
            print(f"  QAT model sanity check failed: {e}")
            return None

        # ── Benchmark ──
        print("  Benchmarking QAT INT8 model...")
        times = []
        outputs = []

        # Warmup
        with torch.no_grad():
            for img in self.train_images[:30]:
                _ = model_int8(img.float())

        with torch.no_grad():
            for img in self.train_images:
                start = time.perf_counter()
                out = model_int8(img.float())
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                outputs.append(out.float().cpu())

        all_outputs = torch.cat(outputs, dim=0)
        avg_ms = np.mean(times)
        mean_est, ci_lo, ci_hi = StatisticalAnalysis.bootstrap_ci(times)
        throughput = 1000.0 / avg_ms if avg_ms > 0 else 0

        print(f"  QAT INT8: {avg_ms:.1f} +/- {np.std(times):.1f} ms  "
              f"[95% CI: {ci_lo:.1f} - {ci_hi:.1f}]  throughput: {throughput:.1f} img/s")

        return {
            'model': model_int8,
            'outputs': all_outputs,
            'times': times,
            'avg_ms': avg_ms,
            'std_ms': float(np.std(times)),
            'ci_lower': ci_lo,
            'ci_upper': ci_hi,
            'throughput_ips': throughput,
            'n_quantized_layers': n_quantized,
            'epochs': self.epochs,
            'lr': self.lr,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MIXED-PRECISION FORWARD PASS
# ═══════════════════════════════════════════════════════════════════════════════

class MixedPrecisionFramework:
    """
    Implements distance-based and empirical mixed-precision inference.

    This is the core novel contribution of the dissertation.
    Instead of uniform quantisation (all INT8 or all FP32), we assign
    different precisions to different connections based on either:
      - distance: how many layers apart the source and destination are
      - empirical: measured sensitivity of each block

    Method:
      We override each dense block's forward pass. Before concatenating
      features for input to layer L, we quantise each feature map from
      layer K based on the distance L - K (or the source block's
      measured sensitivity).
    """

    def __init__(self, model, test_images, layer_sensitivity=None, pathology_col_map=None):
        self.model = model
        self.test_images = test_images
        self.layer_sensitivity = layer_sensitivity or {}
        self.pathology_col_map = pathology_col_map or {}

        # Pre-compute entropy-based sensitivity ranking for each dense block
        self._entropy_ranking = self._compute_entropy_ranking()

    def _compute_entropy_ranking(self):
        """
        Compute a sensitivity score for each dense layer based on weight
        entropy. Higher entropy → more information content → more sensitive
        to quantisation → needs higher precision.

        Inspired by entropy-based mixed-precision methods (GMPQ-TE, HAWQ).
        Uses Shannon entropy of the weight distribution as a proxy for
        layer importance, which avoids the costly Hessian computation
        that methods like HAWQ require.

        Returns:
            dict mapping (block_num, layer_idx) → 'high'/'medium'/'low' sensitivity
        """
        ranking = {}

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue

            # Parse block and layer position
            block_num = None
            layer_num = None
            for part in name.split('.'):
                if 'denseblock' in part:
                    block_num = int(part.replace('denseblock', ''))
                elif 'denselayer' in part:
                    layer_num = int(part.replace('denselayer', ''))

            if block_num is None or layer_num is None:
                continue

            # Compute weight entropy (discretise to 256 bins)
            weights = module.weight.data.cpu().float().numpy().flatten()
            hist, _ = np.histogram(weights, bins=256, density=True)
            hist = hist[hist > 0]  # remove zero bins
            # Normalise to probability
            hist = hist / hist.sum()
            entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))

            # Also factor in weight magnitude range (wider range = more quantisation error)
            weight_range = float(weights.max() - weights.min())

            # Combined score: entropy × range gives a good proxy
            score = entropy * weight_range

            ranking[(block_num, layer_num)] = {
                'entropy': entropy,
                'weight_range': weight_range,
                'score': score,
                'name': name,
            }

        # Classify into thirds: high / medium / low
        if ranking:
            scores = [v['score'] for v in ranking.values()]
            p33 = np.percentile(scores, 33)
            p66 = np.percentile(scores, 66)

            for key in ranking:
                s = ranking[key]['score']
                if s >= p66:
                    ranking[key]['tier'] = 'high'      # sensitive → FP32
                elif s >= p33:
                    ranking[key]['tier'] = 'medium'     # moderate → FP16
                else:
                    ranking[key]['tier'] = 'low'        # robust → INT8

        return ranking

    def _get_layer_tier(self, block_num, layer_idx):
        """Get the entropy-based precision tier for a specific layer."""
        key = (block_num, layer_idx)
        if key in self._entropy_ranking:
            return self._entropy_ranking[key].get('tier', 'medium')
        return 'medium'  # default

    def _quantise_tensor_fp16(self, tensor):
        """Simulate FP16 precision loss."""
        return tensor.half().float()

    def _quantise_tensor_bf16(self, tensor):
        """Simulate BF16 precision loss."""
        return tensor.to(torch.bfloat16).float()

    def _quantise_tensor_int8(self, tensor):
        """Simulate INT8 quantisation (per-tensor)."""
        scale = tensor.abs().max() / 127.0
        if scale > 0:
            return (tensor / scale).round().clamp(-128, 127) * scale
        return tensor

    def _mixed_dense_block_forward(self, block, init_features, strategy, block_num=None):
        """
        Custom forward pass for a DenseNet dense block with mixed precision.

        In xrv's DenseNet, _DenseBlock is nn.Sequential and _DenseLayer
        concatenates input with output internally. Features accumulate via
        channel concatenation: [init | layer1_out | layer2_out | ...].

        We split the accumulated tensor back into per-source channel groups,
        quantise each based on distance from the current layer, re-concatenate,
        then run through the layer's submodules (bypassing _DenseLayer.forward
        which would re-concatenate).

        Args:
            block: a DenseNet _DenseBlock module (nn.Sequential)
            init_features: input features to this block
            strategy: 'distance' or 'empirical'
            block_num: which block this is (needed for empirical strategy)
        """
        # Determine growth rate from the first layer's conv2
        growth_rate = None
        layers = [(name, layer) for name, layer in block.named_children()
                  if name.startswith('denselayer')]

        for _, layer in layers:
            for sub_name, sub_module in layer.named_modules():
                if 'conv2' in sub_name and isinstance(sub_module, nn.Conv2d):
                    growth_rate = sub_module.out_channels
                    break
            if growth_rate is not None:
                break

        if growth_rate is None:
            # Fallback: run block normally without mixed precision
            return block(init_features)

        init_channels = init_features.shape[1]

        # Track channel groups: (start_ch, end_ch, source_layer_idx)
        # source_layer_idx 0 = init_features, 1 = layer 1, etc.
        channel_groups = [(0, init_channels, 0)]

        accumulated = init_features

        for layer_idx, (name, layer) in enumerate(layers):
            current_layer = layer_idx + 1  # 1-indexed

            # Split accumulated features into channel groups and quantise each
            quantised_parts = []
            for start_ch, end_ch, source_idx in channel_groups:
                feat = accumulated[:, start_ch:end_ch, :, :]
                distance = current_layer - source_idx

                if strategy == 'distance':
                    if distance <= 1:
                        q_feat = feat  # FP32 — most recent
                    elif distance <= 3:
                        q_feat = self._quantise_tensor_fp16(feat)  # FP16 — recent
                    else:
                        q_feat = self._quantise_tensor_int8(feat)  # INT8 — distant
                elif strategy == 'empirical':
                    # Based on measured block sensitivity:
                    #   Block 2: least sensitive → aggressive quantisation OK
                    #   Block 3: medium → FP16
                    #   Blocks 1, 4: most sensitive → keep FP32
                    if block_num in [1, 4]:
                        q_feat = feat  # Sensitive blocks: keep FP32
                    elif block_num == 3:
                        q_feat = self._quantise_tensor_fp16(feat) if distance > 1 else feat
                    else:  # Block 2
                        if distance <= 1:
                            q_feat = feat
                        elif distance <= 3:
                            q_feat = self._quantise_tensor_fp16(feat)
                        else:
                            q_feat = self._quantise_tensor_int8(feat)
                elif strategy == 'entropy':
                    # Entropy-based: assign precision per-layer based on
                    # weight entropy ranking (GMPQ-TE inspired).
                    # Each source feature's precision depends on the tier
                    # of the layer that produced it.
                    source_tier = self._get_layer_tier(block_num, source_idx)
                    if source_idx == 0:
                        # Init features (from previous block/stem) — keep FP32
                        q_feat = feat
                    elif source_tier == 'high':
                        q_feat = feat  # Sensitive layer → FP32
                    elif source_tier == 'medium':
                        q_feat = self._quantise_tensor_fp16(feat)  # → FP16
                    else:
                        q_feat = self._quantise_tensor_int8(feat)  # → INT8
                else:
                    q_feat = feat

                quantised_parts.append(q_feat)

            quantised_input = torch.cat(quantised_parts, 1)

            # Run through the layer's submodules directly (norm1→relu1→conv1→norm2→relu2→conv2)
            # bypassing _DenseLayer.forward() which would re-concatenate
            new_features = quantised_input
            for sub_name, sub_module in layer.named_children():
                new_features = sub_module(new_features)

            # Apply dropout if applicable
            if hasattr(layer, 'drop_rate') and layer.drop_rate > 0:
                new_features = F.dropout(new_features, p=layer.drop_rate, training=layer.training)

            # Concatenate new features onto accumulated (like _DenseLayer.forward does)
            accumulated = torch.cat([accumulated, new_features], 1)

            # Track new channel group
            new_start = channel_groups[-1][1]
            channel_groups.append((new_start, new_start + growth_rate, current_layer))

        return accumulated

    def run_mixed_precision_inference(self, strategy='distance'):
        """
        Run full model inference with mixed-precision dense blocks.

        Returns outputs comparable to standard inference for accuracy
        comparison.
        """
        model = copy.deepcopy(self.model).float()
        model.eval()

        # Identify the dense blocks
        # xrv's _DenseBlock inherits from nn.Sequential (not nn.ModuleDict),
        # so we detect blocks by checking for denselayer children
        dense_blocks = {}
        for name, module in model.named_modules():
            child_names = [n for n, _ in module.named_children()]
            if any('denselayer' in n for n in child_names):
                for part in name.split('.'):
                    if 'denseblock' in part:
                        block_num = int(part.replace('denseblock', ''))
                        dense_blocks[block_num] = (name, module)

        if not dense_blocks:
            print("  Warning: could not identify dense blocks")
            return None

        # Save original forward methods and replace
        originals = {}
        for block_num, (name, block) in dense_blocks.items():
            originals[block_num] = block.forward
            # Create closure to capture block_num
            def make_forward(b, bn, strat):
                def new_forward(init_features):
                    return self._mixed_dense_block_forward(b, init_features, strat, bn)
                return new_forward
            block.forward = make_forward(block, block_num, strategy)

        # Run inference
        times = []
        outputs = []
        with torch.no_grad():
            for img in self.test_images:
                start = time.perf_counter()
                out = model(img.float())
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                outputs.append(out.float().cpu())

        # Restore original forward methods
        for block_num, (name, block) in dense_blocks.items():
            block.forward = originals[block_num]

        all_outputs = torch.cat(outputs, dim=0)
        avg_time = np.mean(times)
        std_time = np.std(times)

        return {
            'avg_ms': avg_time, 'std_ms': std_time,
            'times': times, 'outputs': all_outputs
        }

    def compare_strategies(self, fp32_outputs):
        """
        Compare distance-based, empirical, and entropy-based mixed-precision
        against uniform FP32 and uniform INT8.
        """
        print("\n" + "=" * 60)
        print("MIXED-PRECISION FRAMEWORK COMPARISON")
        print("=" * 60)

        # Print entropy ranking summary
        if self._entropy_ranking:
            tiers = defaultdict(int)
            for v in self._entropy_ranking.values():
                tiers[v.get('tier', 'unknown')] += 1
            print(f"\n  Entropy ranking: {dict(tiers)}")

        results = {}

        for strategy in ['distance', 'empirical', 'entropy']:
            print(f"\n  Running {strategy}-based mixed precision...")
            res = self.run_mixed_precision_inference(strategy)
            if res is None:
                continue

            # Compare accuracy vs FP32
            diff = torch.abs(
                torch.sigmoid(res['outputs']) - torch.sigmoid(fp32_outputs)
            )
            mean_diff = float(diff.mean()) * 100
            max_diff = float(diff.max()) * 100

            mean_est, ci_lo, ci_hi = StatisticalAnalysis.bootstrap_ci(res['times'])

            print(f"    Time: {res['avg_ms']:.1f} +/- {res['std_ms']:.1f} ms  "
                  f"[CI: {ci_lo:.1f} - {ci_hi:.1f}]")
            print(f"    Accuracy: mean diff {mean_diff:.4f}%  max diff {max_diff:.4f}%")

            # Per-pathology (using model column mapping)
            per_path = {}
            for i, name in enumerate(PATHOLOGIES):
                model_col = self.pathology_col_map.get(i)
                if model_col is not None and model_col < diff.shape[1]:
                    per_path[name] = {
                        'mean_diff_pct': float(diff[:, model_col].mean()) * 100,
                        'max_diff_pct': float(diff[:, model_col].max()) * 100,
                    }

            # Check clinical safety
            safe = all(
                per_path.get(cp, {}).get('max_diff_pct', 999) < SAFETY_THRESHOLD_PERCENT
                for cp in CRITICAL_PATHOLOGIES
            )
            print(f"    Clinical safety: {'PASS' if safe else 'FAIL'}")

            results[strategy] = {
                'avg_ms': res['avg_ms'], 'std_ms': res['std_ms'],
                'ci_lower': ci_lo, 'ci_upper': ci_hi,
                'mean_diff_pct': mean_diff, 'max_diff_pct': max_diff,
                'per_pathology': per_path, 'clinically_safe': safe
            }

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLICATION-QUALITY VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

class ResultsVisualiser:
    """Creates publication-quality figures from experiment results."""

    def __init__(self, results, model_size_mb):
        self.results = results
        self.model_size_mb = model_size_mb

    def create_main_figure(self, filename='results_main.png'):
        """Figure 1: Core precision comparison results (2x2 grid)."""
        if not HAS_MPL:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})

        # (a) Inference latency with error bars — including native + static
        ax = axes[0, 0]
        prec_data = self.results.get('precision', {})
        native_data = self.results.get('native_precision', {})
        static_data = self.results.get('static_int8', {})

        # Build ordered label list
        labels = [l for l in ['FP32', 'BF16', 'FP16', 'INT8', 'INT8_PerChannel',
                               'Dynamic_INT8', 'Native_BF16', 'Native_FP16', 'Static_INT8']
                  if (l in prec_data or l in native_data or
                      (l == 'Static_INT8' and static_data))]

        display_labels = [l.replace('_', '\n') for l in labels]

        times = []
        ci_lo = []
        ci_hi = []
        colours = []
        for l in labels:
            if l in prec_data:
                d = prec_data[l]
            elif l in native_data:
                d = native_data[l]
            elif l == 'Static_INT8' and static_data:
                d = static_data
            else:
                continue
            times.append(d['avg_ms'])
            ci_lo.append(d['avg_ms'] - d['ci_lower'])
            ci_hi.append(d['ci_upper'] - d['avg_ms'])
            colours.append(COLOURS.get(l, '#888888'))

        if times:
            bars = ax.bar(display_labels, times, color=colours,
                          yerr=[ci_lo, ci_hi], capsize=4, error_kw={'linewidth': 1})
            ax.set_ylabel('Latency (ms)')
            ax.set_title('(a) Inference Latency with 95% CI')
            ax.tick_params(axis='x', rotation=45, labelsize=7)
            for bar, t in zip(bars, times):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(ci_hi) * 0.3,
                        f'{t:.1f}', ha='center', va='bottom', fontsize=7)

        # (b) Speedup comparison
        ax = axes[0, 1]
        speedup_labels = []
        speedups = []
        sp_colours = []
        fp32_time = prec_data.get('FP32', {}).get('avg_ms', 1)

        for l in labels:
            if l == 'FP32':
                continue
            if l in prec_data:
                sp = prec_data[l].get('speedup', fp32_time / prec_data[l]['avg_ms'])
            elif l in native_data:
                sp = native_data[l].get('speedup', fp32_time / native_data[l]['avg_ms'])
            elif l == 'Static_INT8' and static_data:
                sp = static_data.get('speedup', fp32_time / static_data['avg_ms'])
            else:
                continue
            speedup_labels.append(l.replace('_', '\n'))
            speedups.append(sp)
            sp_colours.append(COLOURS.get(l, '#888888'))

        if speedups:
            bars = ax.bar(speedup_labels, speedups, color=sp_colours)
            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='FP32 baseline')
            ax.set_ylabel('Speedup vs FP32')
            ax.set_title('(b) Speedup Factor')
            ax.tick_params(axis='x', rotation=45, labelsize=7)
            for bar, s in zip(bars, speedups):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{s:.2f}x', ha='center', va='bottom', fontsize=7)
            ax.legend(fontsize=8)

        # (c) Block sensitivity
        ax = axes[1, 0]
        block_sens = self.results.get('block_sensitivity', {})
        if block_sens:
            components = sorted([k for k in block_sens.keys()],
                                key=lambda x: ('0' if 'denseblock' in x else '1') + x)
            mses = [block_sens[c]['mse'] for c in components]
            max_mse = max(mses) if mses else 1
            normalised = [m / max_mse * 100 for m in mses]

            c_colours = []
            for m in mses:
                if m == max(mses):
                    c_colours.append('#c0392b')
                elif m == min(mses):
                    c_colours.append('#27ae60')
                else:
                    c_colours.append('#2E86AB')

            bars = ax.bar(range(len(components)), normalised, color=c_colours)
            ax.set_xticks(range(len(components)))
            ax.set_xticklabels([c.replace('denseblock', 'DB').replace('transition', 'T')
                                for c in components], fontsize=8, rotation=45)
            ax.set_ylabel('Relative Sensitivity (%)')
            ax.set_title('(c) Component Sensitivity to INT8')

        # (d) Clinical safety — critical pathology differences
        ax = axes[1, 1]
        clinical = self.results.get('clinical', {})
        if clinical:
            crit_labels = []
            crit_diffs = []
            crit_colours = []
            for prec_label in ['BF16', 'FP16', 'INT8', 'INT8_PerChannel']:
                if prec_label not in clinical:
                    continue
                for cp in CRITICAL_PATHOLOGIES:
                    pp = clinical[prec_label].get('per_pathology', {}).get(cp, {})
                    crit_labels.append(f"{cp[:6]}\n{prec_label[:4]}")
                    val = pp.get('max_diff_pct', 0)
                    crit_diffs.append(val)
                    crit_colours.append('#27ae60' if val < SAFETY_THRESHOLD_PERCENT else '#c0392b')

            if crit_diffs:
                bars = ax.bar(range(len(crit_labels)), crit_diffs, color=crit_colours)
                ax.set_xticks(range(len(crit_labels)))
                ax.set_xticklabels(crit_labels, fontsize=7)
                ax.axhline(y=SAFETY_THRESHOLD_PERCENT, color='red', linestyle='--',
                           linewidth=1, label=f'{SAFETY_THRESHOLD_PERCENT}% threshold')
                ax.set_ylabel('Max Probability Diff (%)')
                ax.set_title('(d) Critical Pathology Safety')
                ax.legend(fontsize=8)

        plt.suptitle('Low-Precision DenseNet-121 for Chest X-ray Diagnosis\n'
                      '(Simulated + Native Precision + Static INT8)',
                      fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()

    def create_detailed_figure(self, filename='results_detailed.png'):
        """Figure 2: Detailed analysis (2x2 grid)."""
        if not HAS_MPL:
            return

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # (a) Per-pathology heatmap (INT8 vs FP32)
        ax = axes[0, 0]
        acc = self.results.get('accuracy', {}).get('INT8', {}).get('per_pathology', {})
        if acc:
            path_names = [p for p in PATHOLOGIES if p in acc]
            diffs = [acc[p]['max_diff_pct'] for p in path_names]

            cmap = LinearSegmentedColormap.from_list('safety',
                                                     ['#27ae60', '#f1c40f', '#c0392b'])
            bars = ax.barh(range(len(path_names)), diffs, color=[
                cmap(d / max(max(diffs), 0.01)) for d in diffs
            ])
            ax.set_yticks(range(len(path_names)))
            ax.set_yticklabels(path_names, fontsize=8)
            ax.set_xlabel('Max Probability Diff (%)')
            ax.set_title('(a) INT8 vs FP32 by Pathology')
            ax.axvline(x=SAFETY_THRESHOLD_PERCENT, color='red', linestyle='--',
                       linewidth=1, label=f'{SAFETY_THRESHOLD_PERCENT}% threshold')
            ax.legend(fontsize=8)

            # Highlight critical pathologies
            for i, p in enumerate(path_names):
                if p in CRITICAL_PATHOLOGIES:
                    ax.get_yticklabels()[i].set_fontweight('bold')

        # (b) Connection distance distribution
        ax = axes[0, 1]
        conn = self.results.get('connections', {}).get('distance_distribution', {})
        if conn:
            distances = sorted(conn.keys(), key=int)
            counts = [conn[d] for d in distances]
            dist_colours = []
            for d in distances:
                d_int = int(d)
                if d_int <= 1:
                    dist_colours.append(COLOURS['FP32'])
                elif d_int <= 3:
                    dist_colours.append(COLOURS['FP16'])
                else:
                    dist_colours.append(COLOURS['INT8'])

            ax.bar([int(d) for d in distances], counts, color=dist_colours)
            ax.set_xlabel('Connection Distance')
            ax.set_ylabel('Number of Connections')
            ax.set_title('(b) Connection Distance Distribution')

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=COLOURS['FP32'], label='FP32 (d=1)'),
                Patch(facecolor=COLOURS['FP16'], label='FP16 (d=2-3)'),
                Patch(facecolor=COLOURS['INT8'], label='INT8 (d>3)'),
            ]
            ax.legend(handles=legend_elements, fontsize=8)

        # (c) Mixed-precision comparison
        ax = axes[1, 0]
        mixed = self.results.get('mixed_precision', {})
        prec = self.results.get('precision', {})
        if mixed or prec:
            comp_labels = []
            comp_times = []
            comp_diffs = []
            comp_colours = []

            if 'FP32' in prec:
                comp_labels.append('Uniform\nFP32')
                comp_times.append(prec['FP32']['avg_ms'])
                comp_diffs.append(0)
                comp_colours.append(COLOURS['FP32'])
            if 'INT8' in prec:
                comp_labels.append('Uniform\nINT8')
                comp_times.append(prec['INT8']['avg_ms'])
                comp_diffs.append(
                    self.results.get('accuracy', {}).get('INT8', {}).get('max_diff_pct', 0))
                comp_colours.append(COLOURS['INT8'])
            if 'distance' in mixed:
                comp_labels.append('Mixed\n(Distance)')
                comp_times.append(mixed['distance']['avg_ms'])
                comp_diffs.append(mixed['distance']['max_diff_pct'])
                comp_colours.append(COLOURS['Mixed (Distance)'])
            if 'empirical' in mixed:
                comp_labels.append('Mixed\n(Empirical)')
                comp_times.append(mixed['empirical']['avg_ms'])
                comp_diffs.append(mixed['empirical']['max_diff_pct'])
                comp_colours.append(COLOURS['Mixed (Empirical)'])

            if comp_labels:
                x = np.arange(len(comp_labels))
                width = 0.35
                ax.bar(x - width / 2, comp_times, width, color=comp_colours,
                       alpha=0.8, label='Latency (ms)')
                ax2 = ax.twinx()
                ax2.bar(x + width / 2, comp_diffs, width, color=comp_colours,
                        alpha=0.4, hatch='//', label='Max Diff (%)')
                ax.set_xticks(x)
                ax.set_xticklabels(comp_labels, fontsize=8)
                ax.set_ylabel('Latency (ms)')
                ax2.set_ylabel('Max Probability Diff (%)')
                ax.set_title('(c) Mixed-Precision Strategies')
                ax.legend(loc='upper left', fontsize=8)
                ax2.legend(loc='upper right', fontsize=8)

        # (d) Layer sensitivity by position
        ax = axes[1, 1]
        layer_sens = self.results.get('layer_sensitivity', {}).get('layers', [])
        if layer_sens:
            positions = []
            mses = []
            block_nums = []
            for r in layer_sens:
                if r.get('int8_mse') is not None and r['position'].get('layer_num'):
                    positions.append(r['position']['layer_num'])
                    mses.append(r['int8_mse'])
                    block_nums.append(r['position'].get('block', 0))

            if positions:
                scatter_colours = [
                    ['#2E86AB', '#A23B72', '#F18F01', '#27AE60'][
                        (b - 1) % 4] if b else '#888888'
                    for b in block_nums
                ]
                ax.scatter(positions, mses, c=scatter_colours, alpha=0.6, s=30)
                ax.set_xlabel('Layer Position Within Block')
                ax.set_ylabel('INT8 Sensitivity (MSE)')
                ax.set_yscale('log')
                ax.set_title('(d) Layer Sensitivity by Position')

                from matplotlib.patches import Patch
                legend_el = [Patch(facecolor=c, label=f'Block {i}')
                             for i, c in enumerate(['#2E86AB', '#A23B72', '#F18F01', '#27AE60'], 1)]
                ax.legend(handles=legend_el, fontsize=8)

        plt.suptitle('Detailed Analysis: Quantisation Sensitivity & Mixed Precision',
                      fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()

    def create_hardware_figure(self, filename='results_hardware.png'):
        """
        Figure 3: Hardware-accelerated results comparison.

        Compares simulated vs native vs static quantisation — the key
        dissertation figure showing that real hardware acceleration
        delivers meaningful speedups on server-class CPUs.
        """
        if not HAS_MPL:
            return

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        prec = self.results.get('precision', {})
        native = self.results.get('native_precision', {})
        static = self.results.get('static_int8', {})
        fp32_time = prec.get('FP32', {}).get('avg_ms', 1)

        # (a) Simulated vs Native vs Static — latency bars
        ax = axes[0]
        groups = []
        group_times = []
        group_colours = []

        comparisons = [
            ('FP32', prec.get('FP32', {}), COLOURS['FP32']),
            ('BF16\n(sim)', prec.get('BF16', {}), COLOURS['BF16']),
            ('BF16\n(native)', native.get('Native_BF16', {}), COLOURS['Native_BF16']),
            ('FP16\n(sim)', prec.get('FP16', {}), COLOURS['FP16']),
            ('FP16\n(native)', native.get('Native_FP16', {}), COLOURS['Native_FP16']),
            ('INT8\n(sim)', prec.get('INT8', {}), COLOURS['INT8']),
            ('INT8\n(static)', static if static else {}, COLOURS['Static_INT8']),
        ]

        for label, data, colour in comparisons:
            if data and 'avg_ms' in data:
                groups.append(label)
                group_times.append(data['avg_ms'])
                group_colours.append(colour)

        if groups:
            bars = ax.bar(groups, group_times, color=group_colours)
            ax.set_ylabel('Latency (ms)')
            ax.set_title('(a) Simulated vs Hardware-Accelerated')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            for bar, t in zip(bars, group_times):
                sp = fp32_time / t if t > 0 else 0
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{t:.1f}ms\n{sp:.2f}x', ha='center', va='bottom', fontsize=7)

        # (b) Accuracy vs Speedup scatter
        ax = axes[1]
        scatter_data = []

        for label, data_src, colour in [
            ('BF16 (sim)', ('accuracy', 'BF16'), COLOURS['BF16']),
            ('FP16 (sim)', ('accuracy', 'FP16'), COLOURS['FP16']),
            ('INT8 (sim)', ('accuracy', 'INT8'), COLOURS['INT8']),
            ('Native BF16', ('native_precision', 'Native_BF16'), COLOURS['Native_BF16']),
            ('Native FP16', ('native_precision', 'Native_FP16'), COLOURS['Native_FP16']),
            ('Static INT8', ('static_int8', None), COLOURS['Static_INT8']),
        ]:
            if data_src[1] is None:
                data = self.results.get(data_src[0], {})
            else:
                data = self.results.get(data_src[0], {}).get(data_src[1], {})

            if data and 'max_diff_pct' in data:
                sp_key = 'speedup'
                if sp_key in data:
                    sp = data[sp_key]
                elif 'avg_ms' in data:
                    sp = fp32_time / data['avg_ms']
                else:
                    continue
                scatter_data.append((label, sp, data['max_diff_pct'], colour))

        if scatter_data:
            for label, sp, diff, colour in scatter_data:
                ax.scatter(sp, diff, c=colour, s=100, zorder=5, edgecolors='black', linewidth=0.5)
                ax.annotate(label, (sp, diff), textcoords="offset points",
                            xytext=(5, 5), fontsize=7)

            ax.axhline(y=SAFETY_THRESHOLD_PERCENT, color='red', linestyle='--',
                       linewidth=1, label=f'{SAFETY_THRESHOLD_PERCENT}% safety threshold')
            ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.5)
            ax.set_xlabel('Speedup vs FP32')
            ax.set_ylabel('Max Probability Diff (%)')
            ax.set_title('(b) Accuracy–Speed Trade-off')
            ax.legend(fontsize=8)

        # (c) Per-pathology comparison: Static INT8 vs Simulated INT8
        ax = axes[2]
        sim_acc = self.results.get('accuracy', {}).get('INT8', {}).get('per_pathology', {})
        static_acc = static.get('per_pathology', {}) if static else {}

        if sim_acc and static_acc:
            path_names = [p for p in PATHOLOGIES if p in sim_acc and p in static_acc]
            sim_diffs = [sim_acc[p]['max_diff_pct'] for p in path_names]
            static_diffs = [static_acc[p]['max_diff_pct'] for p in path_names]

            x = np.arange(len(path_names))
            width = 0.35
            ax.barh(x - width / 2, sim_diffs, width, color=COLOURS['INT8'],
                    alpha=0.8, label='Simulated INT8')
            ax.barh(x + width / 2, static_diffs, width, color=COLOURS['Static_INT8'],
                    alpha=0.8, label='Static INT8')
            ax.set_yticks(x)
            ax.set_yticklabels(path_names, fontsize=7)
            ax.set_xlabel('Max Probability Diff (%)')
            ax.set_title('(c) Simulated vs Static INT8 by Pathology')
            ax.axvline(x=SAFETY_THRESHOLD_PERCENT, color='red', linestyle='--',
                       linewidth=1)
            ax.legend(fontsize=8)

            for i, p in enumerate(path_names):
                if p in CRITICAL_PATHOLOGIES:
                    ax.get_yticklabels()[i].set_fontweight('bold')
        elif sim_acc:
            path_names = [p for p in PATHOLOGIES if p in sim_acc]
            diffs = [sim_acc[p]['max_diff_pct'] for p in path_names]
            ax.barh(range(len(path_names)), diffs, color=COLOURS['INT8'])
            ax.set_yticks(range(len(path_names)))
            ax.set_yticklabels(path_names, fontsize=7)
            ax.set_xlabel('Max Probability Diff (%)')
            ax.set_title('(c) INT8 per Pathology (simulated only)')

        plt.suptitle('Hardware-Accelerated Low-Precision Inference\n'
                      'AMD EPYC 9005 — AVX-512 VNNI / BF16 / FP16',
                      fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()

    def create_energy_figure(self, filename='results_energy.png'):
        """
        Figure 4: Energy consumption and carbon footprint comparison.

        Bar chart showing joules per image and gCO2 per 1000 images
        for each precision format, derived from TDP-based estimation
        or RAPL measurements.
        """
        if not HAS_MPL:
            return

        energy = self.results.get('energy', {})
        # Filter to entries that have actual data (dicts with joules_per_image)
        plot_data = {k: v for k, v in energy.items()
                     if isinstance(v, dict) and 'joules_per_image' in v}

        if not plot_data:
            print("  No energy data to plot")
            return

        # Sort by energy consumption (descending)
        sorted_labels = sorted(plot_data.keys(),
                               key=lambda k: plot_data[k]['joules_per_image'],
                               reverse=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Assign colours
        colours = []
        for label in sorted_labels:
            c = COLOURS.get(label, '#888888')
            colours.append(c)

        # (a) Joules per image
        j_values = [plot_data[l]['joules_per_image'] for l in sorted_labels]
        display_labels = [l.replace('_', '\n') for l in sorted_labels]

        bars1 = ax1.barh(display_labels, j_values, color=colours)
        ax1.set_xlabel('Energy per Image (Joules)')
        ax1.set_title('(a) Energy Consumption per Inference')
        ax1.invert_yaxis()

        # Annotate with values
        for bar, val in zip(bars1, j_values):
            ax1.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height() / 2,
                     f'{val:.4f}J', ha='left', va='center', fontsize=8)

        # (b) gCO2 per 1000 images
        co2_values = [plot_data[l]['gco2_per_1000_images'] for l in sorted_labels]

        bars2 = ax2.barh(display_labels, co2_values, color=colours)
        ax2.set_xlabel('Carbon Footprint (gCO2 per 1000 images)')
        ax2.set_title('(b) Carbon Footprint per 1000 Inferences')
        ax2.invert_yaxis()

        for bar, val in zip(bars2, co2_values):
            ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     f'{val:.2f}g', ha='left', va='center', fontsize=8)

        # Add FP32 savings reference line
        if 'FP32' in plot_data:
            fp32_co2 = plot_data['FP32']['gco2_per_1000_images']
            ax2.axvline(x=fp32_co2, color='red', linestyle='--',
                        linewidth=1, alpha=0.7, label='FP32 baseline')
            ax2.legend(fontsize=8)

        method = energy.get('method', 'unknown')
        method_str = 'RAPL (measured)' if method == 'rapl' else 'TDP-based (estimated)'
        plt.suptitle(f'Energy & Carbon Footprint — {method_str}\n'
                      f'UK grid: 233 gCO2/kWh',
                      fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(results, filename='demo_results.json'):
    """Save results to JSON, converting non-serialisable types."""

    def clean(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        return obj

    cleaned = clean(results)

    with open(filename, 'w') as f:
        json.dump(cleaned, f, indent=2)

    print(f"\n  Results saved to {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Low-Precision CNNs on Server-Class CPUs — Experiment Suite"
    )
    parser.add_argument('--images', type=int, default=1000,
                        help='Number of images to test (default: 1000)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 100 images, fewer layer tests')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune DenseNet-121 on ChestX-ray14 labels before experiments')
    parser.add_argument('--finetune-epochs', type=int, default=5,
                        help='Number of fine-tuning epochs (default: 5)')
    parser.add_argument('--qat', action='store_true',
                        help='Run Quantization-Aware Training after static INT8')
    parser.add_argument('--qat-epochs', type=int, default=3,
                        help='Number of QAT epochs (default: 3)')
    parser.add_argument('--onnx', action='store_true',
                        help='Run ONNX Runtime comparison (requires onnxruntime)')
    parser.add_argument('--data', type=str, default=DATA_PATH,
                        help='Path to chest X-ray images')
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory for results')
    args = parser.parse_args()

    if args.quick:
        args.images = 100
        max_layers = 30
    else:
        max_layers = 120

    # ── Thread optimisation for server CPUs ──────────────────────────────
    # Thread sweep (see thread_count_sweep()) showed 2 threads is optimal
    # for single-image DenseNet-121 inference on 4-vCPU Azure VM
    # (Standard_F4s_v2, EPYC 9005). At 2 threads the workload fits in
    # L1/L2 without cross-CCX synchronisation overhead.
    n_threads = 2
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(1)  # single inter-op thread avoids contention

    # Force fbgemm backend for x86 (AVX-512 VNNI on EPYC 9005)
    try:
        torch.backends.quantized.engine = 'fbgemm'
    except Exception:
        pass

    print("=" * 60)
    print("Low-Precision CNNs on Server-Class CPUs")
    print("DenseNet-121 Chest X-ray Diagnosis")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Images: {args.images}  Mode: {'quick' if args.quick else 'full'}")
    print(f"Threads: {n_threads}  Backend: {torch.backends.quantized.engine}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 60)

    # Create output directory
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Setup ──────────────────────────────────────────────────────────
    exp = PrecisionExperiment(data_path=args.data, max_images=args.images)
    exp.load_model_and_data()

    # ── 1b. Fine-tune if requested ─────────────────────────────────────
    if args.finetune and exp.ground_truth_labels is not None:
        finetuner = FineTuner(
            exp.model, exp.test_images, exp.ground_truth_labels,
            epochs=args.finetune_epochs, batch_size=16, lr=1e-4
        )
        ft_result = finetuner.run()
        if ft_result and 'model' in ft_result:
            # Replace the experiment model with the fine-tuned version
            exp.model = ft_result['model']
            exp.model.eval()
            exp.results['fine_tuning'] = ft_result.get('history', {})
            print("  Model replaced with fine-tuned version for all subsequent experiments.")
    elif args.finetune and exp.ground_truth_labels is None:
        print("\n  WARNING: --finetune requires ground-truth labels (Data_Entry CSV).")
        print("  Skipping fine-tuning.")

    # ── 2. Precision comparison (simulated) ──────────────────────────────
    exp.run_precision_comparison()

    # ── 3. Block sensitivity ──────────────────────────────────────────────
    exp.analyse_block_sensitivity()

    # ── 4. Layer sensitivity ──────────────────────────────────────────────
    exp.analyse_layer_sensitivity(max_layers=max_layers)

    # ── 5. Clinical accuracy ──────────────────────────────────────────────
    exp.analyse_clinical_accuracy()

    # ── 5b. Calibration and AUC-ROC analysis ─────────────────────────────
    # Pass ground-truth labels if available — this enables real AUC-ROC
    # computation instead of pseudo-label agreement with FP32.
    exp.analyse_calibration_and_auc(labels_matrix=exp.ground_truth_labels)

    # ── 5c. McNemar's test — classification decision equivalence ─────────
    # Tests whether quantised formats make systematically different binary
    # classification decisions compared to FP32 (per pathology).
    exp.mcnemar_test(labels_matrix=exp.ground_truth_labels)

    # ── 6. Distance connections ───────────────────────────────────────────
    exp.analyse_distance_connections()

    # ── 7. Mixed-precision framework (now with 3 strategies) ─────────────
    print("\n" + "=" * 60)
    print("MIXED-PRECISION FRAMEWORK")
    print("=" * 60)

    layer_sens = exp.results.get('layer_sensitivity', {})
    mp = MixedPrecisionFramework(exp.model, exp.test_images, layer_sens, exp.pathology_col_map)
    mixed_results = mp.compare_strategies(exp._outputs['FP32'])
    exp.results['mixed_precision'] = mixed_results

    # Store entropy ranking for reference
    exp.results['entropy_ranking'] = {
        f"block{k[0]}_layer{k[1]}": {
            'entropy': v['entropy'], 'weight_range': v['weight_range'],
            'score': v['score'], 'tier': v['tier'], 'name': v['name']
        }
        for k, v in mp._entropy_ranking.items()
    }

    # ── 8. Native BF16 / FP16 (hardware-accelerated) ─────────────────────
    # Force garbage collection before native precision benchmarks so that
    # leftover deep-copied models from previous steps don't pollute the
    # CPU cache or cause memory pressure that inflates latency numbers.
    import gc
    gc.collect()
    exp.run_native_precision()

    # ── 9. Static INT8 quantisation (hardware-accelerated) ────────────────
    exp.run_static_quantisation()

    # ── 9b. QAT if requested ────────────────────────────────────────────
    if args.qat and exp.ground_truth_labels is not None:
        qat_trainer = QATTrainer(
            exp.model, exp.test_images, exp.ground_truth_labels,
            epochs=args.qat_epochs, batch_size=16, lr=1e-5
        )
        qat_result = qat_trainer.run()
        if qat_result:
            # Compare QAT INT8 vs FP32
            if 'FP32' in exp._outputs:
                fp32_probs = torch.sigmoid(exp._outputs['FP32'])
                qat_probs = torch.sigmoid(qat_result['outputs'])
                diff = torch.abs(qat_probs - fp32_probs)
                mean_diff_pct = float(diff.mean()) * 100
                max_diff_pct = float(diff.max()) * 100
                fp32_time = exp.results.get('precision', {}).get('FP32', {}).get('avg_ms', 1)
                speedup = fp32_time / qat_result['avg_ms'] if qat_result['avg_ms'] > 0 else 0

                # Clinical safety check (using model column mapping)
                per_path = {}
                for i, pathology in enumerate(PATHOLOGIES):
                    model_col = exp.pathology_col_map.get(i)
                    if model_col is not None and model_col < diff.shape[1]:
                        per_path[pathology] = {
                            'mean_diff_pct': float(diff[:, model_col].mean()) * 100,
                            'max_diff_pct': float(diff[:, model_col].max()) * 100,
                        }
                safe = all(
                    per_path.get(cp, {}).get('max_diff_pct', 999) < SAFETY_THRESHOLD_PERCENT
                    for cp in CRITICAL_PATHOLOGIES
                )

                print(f"\n  QAT INT8 Results:")
                print(f"    Speedup vs FP32: {speedup:.2f}x")
                print(f"    Mean prob diff:  {mean_diff_pct:.4f}%")
                print(f"    Max prob diff:   {max_diff_pct:.4f}%")
                print(f"    Clinical safety: {'PASS' if safe else 'FAIL'}")

                qat_result['speedup'] = speedup
                qat_result['mean_diff_pct'] = mean_diff_pct
                qat_result['max_diff_pct'] = max_diff_pct
                qat_result['per_pathology'] = per_path
                qat_result['clinically_safe'] = safe

            exp.results['qat_int8'] = {k: v for k, v in qat_result.items()
                                        if k not in ('model', 'outputs')}
            exp._outputs['QAT_INT8'] = qat_result['outputs']
            exp._times['QAT_INT8'] = qat_result['times']
    elif args.qat and exp.ground_truth_labels is None:
        print("\n  WARNING: --qat requires ground-truth labels. Skipping QAT.")

    # ── 9c. ONNX Runtime comparison ──────────────────────────────────────
    if args.onnx:
        exp.run_onnx_runtime_comparison()

    # ── 10. Model size analysis (run after benchmarks to avoid cache effects) ──
    exp.report_model_sizes()

    # ── 11. Weight distribution histograms ────────────────────────────────
    exp.plot_weight_distributions(
        filename=os.path.join(out_dir, 'weight_distributions.png')
    )

    # ── 11. Energy measurement ────────────────────────────────────────────
    exp.measure_energy()

    # ── 11b. Tail latency analysis ──────────────────────────────────────
    exp.analyse_tail_latencies()
    exp.plot_latency_histogram(filename=os.path.join(out_dir, 'latency_histogram.png'))

    # ── 11c. NUMA topology ───────────────────────────────────────────────
    exp.detect_numa_topology()

    # ── 12. Thread count sweep ────────────────────────────────────────────
    if not args.quick:
        exp.thread_count_sweep()

    # ── 13. Batch size sweep ──────────────────────────────────────────────
    if not args.quick:
        exp.batch_size_sweep()

    # ── 14. Latency drift plot ────────────────────────────────────────────
    exp.plot_latency_drift(filename=os.path.join(out_dir, 'latency_drift.png'))
    
    # ── 14b. Additions (CAM comparison, calibration sweep, idle baseline) ──
    try:
        from additions import run_additions
        additions_results = run_additions(exp, output_dir=out_dir)
        exp.results['additions'] = additions_results
    except Exception as e:
        print(f"  Additions skipped: {e}")

    # ── 15. Save results ─────────────────────────────────────────────────
    exp.results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'images_tested': len(exp.test_images),
        'using_real_images': exp.using_real_images,
        'pytorch_version': torch.__version__,
        'numpy_version': np.__version__,
        'python_version': platform.python_version(),
        'cpu_model': platform.processor() or 'unknown',
        'quantisation_backend': exp.backend,
        'cpu_threads': n_threads,
        'torch_num_threads': torch.get_num_threads(),
        'random_seed': 42,
        'mode': 'quick' if args.quick else 'full',
        'has_scipy': HAS_SCIPY,
        'has_sklearn': HAS_SKLEARN,
        'has_onnxruntime': HAS_ORT,
        'ground_truth_labels': exp.ground_truth_labels is not None,
        'fine_tuned': args.finetune and exp.ground_truth_labels is not None,
        'qat': args.qat and exp.ground_truth_labels is not None,
    }

    # Try to get detailed CPU info on Linux
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if 'model name' in line:
                    exp.results['metadata']['cpu_model_name'] = line.split(':')[1].strip()
                    break
    except (IOError, IndexError):
        pass

    results_path = os.path.join(out_dir, 'demo_results.json')
    save_results(exp.results, results_path)

    # ── 16. Figures ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    viz = ResultsVisualiser(exp.results, exp.model_size_mb)
    viz.create_main_figure(os.path.join(out_dir, 'results_main.png'))
    viz.create_detailed_figure(os.path.join(out_dir, 'results_detailed.png'))
    viz.create_hardware_figure(os.path.join(out_dir, 'results_hardware.png'))
    viz.create_energy_figure(os.path.join(out_dir, 'results_energy.png'))

    # ── 17. Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    n = len(exp.test_images)
    p = exp.results.get('precision', {})
    a = exp.results.get('accuracy', {})
    c = exp.results.get('clinical', {})
    m = exp.results.get('mixed_precision', {})
    nat = exp.results.get('native_precision', {})
    sta = exp.results.get('static_int8', {})

    print(f"\n  Images tested: {n} ({'real' if exp.using_real_images else 'synthetic'})")
    print(f"  CPU threads:   {n_threads}")
    print(f"  Backend:       {exp.backend}")

    # ── Consolidated summary table ──────────────────────────────────────
    # Mirrors the simulated summary format but adds real hardware results
    fp32_ms = p.get('FP32', {}).get('avg_ms', None)

    print(f"\n  --- Simulated Precision (weight-cast only) ---")
    for label in ['FP64', 'BF16', 'FP16', 'INT8', 'INT8_PerChannel', 'Dynamic_INT8']:
        if label not in p:
            continue
        res = p[label]
        sp = res.get('speedup', 0)
        md = a.get(label, {}).get('max_diff_pct', 0)
        clinical = c.get(label, {})
        safe_str = ""
        if clinical:
            safe_str = "  SAFE" if clinical.get('all_critical_safe', False) else "  REVIEW"
        print(f"  {label:<20} {res['avg_ms']:.1f} ms  speedup {sp:.2f}x  "
              f"max diff {md:.4f}%{safe_str}")

    print(f"\n  --- Real Hardware-Accelerated Precision ---")
    any_hw = False
    if 'Native_BF16' in nat and fp32_ms:
        r = nat['Native_BF16']
        sp = r.get('speedup', fp32_ms / r['avg_ms'] if r['avg_ms'] > 0 else 0)
        md = r.get('max_diff_pct', 0)
        safe_str = "  SAFE" if r.get('clinically_safe', True) else "  REVIEW"
        print(f"  {'Native BF16':<20} {r['avg_ms']:.1f} ms  speedup {sp:.2f}x  "
              f"max diff {md:.4f}%{safe_str}")
        any_hw = True
    if 'Native_FP16' in nat and fp32_ms:
        r = nat['Native_FP16']
        sp = r.get('speedup', fp32_ms / r['avg_ms'] if r['avg_ms'] > 0 else 0)
        md = r.get('max_diff_pct', 0)
        print(f"  {'Native FP16':<20} {r['avg_ms']:.1f} ms  speedup {sp:.2f}x  "
              f"max diff {md:.4f}%  (no HW support on x86)")
        any_hw = True
    if sta and fp32_ms:
        sp = sta.get('speedup', fp32_ms / sta['avg_ms'] if sta['avg_ms'] > 0 else 0)
        md = sta.get('max_diff_pct', 0)
        safe_str = ""
        if 'clinically_safe' in sta:
            safe_str = "  SAFE" if sta['clinically_safe'] else "  REVIEW"
        print(f"  {'Static INT8 (VNNI)':<20} {sta['avg_ms']:.1f} ms  speedup {sp:.2f}x  "
              f"max diff {md:.4f}%{safe_str}")
        any_hw = True
    # QAT INT8
    qat = exp.results.get('qat_int8', {})
    if qat and fp32_ms:
        sp = qat.get('speedup', fp32_ms / qat['avg_ms'] if qat.get('avg_ms', 0) > 0 else 0)
        md = qat.get('max_diff_pct', 0)
        safe_str = ""
        if 'clinically_safe' in qat:
            safe_str = "  SAFE" if qat['clinically_safe'] else "  REVIEW"
        print(f"  {'QAT INT8':<20} {qat['avg_ms']:.1f} ms  speedup {sp:.2f}x  "
              f"max diff {md:.4f}%{safe_str}")
        any_hw = True

    # ONNX Runtime
    ort_results = exp.results.get('onnx_runtime', {})
    if ort_results:
        for label, r in ort_results.items():
            sp = r.get('speedup_vs_pytorch_fp32', fp32_ms / r['avg_ms'] if r.get('avg_ms', 0) > 0 else 0)
            md = r.get('max_diff_pct', 0)
            print(f"  {label:<20} {r['avg_ms']:.1f} ms  speedup {sp:.2f}x  "
                  f"max diff {md:.4f}%")
        any_hw = True

    if not any_hw:
        print(f"  (no hardware-accelerated results available)")

    print(f"\n  --- Mixed-Precision ---")
    for strat_name, strat_label in [('distance', 'Distance-based'),
                                     ('empirical', 'Empirical-based'),
                                     ('entropy', 'Entropy-based')]:
        if strat_name in m:
            s = m[strat_name]
            time_str = f"  {s['avg_ms']:.1f} ms" if 'avg_ms' in s else ""
            print(f"  {strat_label:<20}{time_str}  max diff {s['max_diff_pct']:.4f}%  "
                  f"{'SAFE' if s.get('clinically_safe', True) else 'REVIEW'}")

    # Best result highlight
    print(f"\n  --- Best Results ---")
    best_candidates = {}
    if 'Native_BF16' in nat:
        best_candidates['Native BF16'] = nat['Native_BF16']
    if sta:
        best_candidates['Static INT8 (VNNI)'] = sta
    for label in ['BF16', 'INT8']:
        if label in p:
            best_candidates[f'{label} (simulated)'] = p[label]
    if best_candidates:
        fastest = min(best_candidates.items(), key=lambda x: x[1].get('avg_ms', 9999))
        print(f"  Fastest:       {fastest[0]} at {fastest[1]['avg_ms']:.1f} ms")
    if fp32_ms:
        print(f"  FP32 baseline: {fp32_ms:.1f} ms")

    # ── Calibration summary ──────────────────────────────────────────────
    cal = exp.results.get('calibration', {})
    if cal:
        print(f"\n  --- Calibration (ECE / MCE) ---")
        for label in ['FP32', 'BF16', 'FP16', 'INT8', 'Native_BF16', 'Static_INT8']:
            if label in cal:
                ece = cal[label].get('mean_ece')
                mce = cal[label].get('max_mce')
                ece_str = f"ECE={ece:.4f}" if ece is not None else "ECE=N/A"
                mce_str = f"MCE={mce:.4f}" if mce is not None else "MCE=N/A"
                print(f"  {label:<20} {ece_str}  {mce_str}")

    # ── Consolidated table ──────────────────────────────────────────────
    print(f"\n  {'='*85}")
    print(f"  {'Method':<22} {'Latency':>8} {'Speed':>6} {'Tput':>7} "
          f"{'MaxDiff':>8} {'ECE':>7} {'Safe':>5}")
    print(f"  {'-'*85}")

    def _print_row(label, data, acc_data=None, cal_data=None):
        ms = data.get('avg_ms', 0)
        sp = data.get('speedup', fp32_ms / ms if fp32_ms and ms > 0 else 0)
        tp = data.get('throughput_ips', 1000.0 / ms if ms > 0 else 0)
        md = 0
        if acc_data:
            md = acc_data.get('max_diff_pct', 0)
        elif 'max_diff_pct' in data:
            md = data['max_diff_pct']
        ece = cal_data.get('mean_ece', None) if cal_data else None
        ece_str = f"{ece:.4f}" if ece is not None else "  -"
        safe = "  -"
        if 'clinically_safe' in data:
            safe = " YES" if data['clinically_safe'] else "  NO"
        elif acc_data and 'all_critical_safe' in (c.get(label.replace(' (sim)', ''), {}) or {}):
            pass  # handled above
        print(f"  {label:<22} {ms:>7.1f}ms {sp:>5.2f}x {tp:>6.1f} "
              f"{md:>7.4f}% {ece_str:>7} {safe:>5}")

    # Simulated
    for label in ['FP32', 'BF16', 'FP16', 'INT8', 'INT8_PerChannel', 'Dynamic_INT8']:
        if label in p:
            _print_row(label, p[label], a.get(label), cal.get(label))

    # Native
    for label in ['Native_BF16', 'Native_FP16']:
        if label in nat:
            _print_row(label, nat[label], cal_data=cal.get(label))

    # Static INT8
    if sta:
        _print_row('Static INT8', sta, cal_data=cal.get('Static_INT8'))

    # QAT INT8
    if qat:
        _print_row('QAT INT8', qat)

    # ONNX Runtime
    for label in ['ORT_FP32', 'ORT_INT8']:
        if label in ort_results:
            _print_row(label, ort_results[label])

    # Mixed precision
    for strat_name, strat_label in [('distance', 'Mixed (Distance)'),
                                     ('empirical', 'Mixed (Empirical)'),
                                     ('entropy', 'Mixed (Entropy)')]:
        if strat_name in m:
            _print_row(strat_label, m[strat_name])

    print(f"  {'='*85}")

    print(f"\n  Output files:")
    print(f"    {results_path}")
    for fig_name in ['results_main.png', 'results_detailed.png',
                     'results_hardware.png', 'results_energy.png',
                     'weight_distributions.png', 'latency_drift.png',
                     'latency_histogram.png']:
        fig_path = os.path.join(out_dir, fig_name)
        if os.path.exists(fig_path):
            print(f"    {fig_path}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
