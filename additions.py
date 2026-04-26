"""
================================================================================
Dissertation additions — three missing pieces identified during code review
================================================================================

This module adds three things that are NOT already in newmain.py:

  1. GradCAMAnalyser   — FP32-vs-INT8 class activation map comparison with
                         IoU + Pearson agreement metrics. Supports the
                         clinical-safety argument visually.

  2. calibration_size_sweep  — ablation over INT8 calibration sample size
                               (32, 128, 512, 1024). Shows AUC stability
                               wrt observer statistics.

  3. idle_power_baseline  — measures baseline package power with the model
                            idle, so per-inference Joules can be reported
                            as (total - idle) rather than total. Follows
                            UPM (2024) methodology.

All three import from newmain.py and reuse its models / images / helpers.
Usage:
    from additions import run_additions
    run_additions(exp)   # exp is a populated PrecisionExperiment instance
"""

import os
import glob
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GRAD-CAM / CAM COMPARISON  (FP32 vs INT8 vs Mixed)
# ═══════════════════════════════════════════════════════════════════════════════

class GradCAMAnalyser:
    """
    Class Activation Map comparison between precision formats.

    Uses the Zhou et al. (2016) CAM formulation rather than Grad-CAM because
    DenseNet-121 ends in global-average-pool + single linear classifier —
    the architecture CAM was designed for. Advantages over Grad-CAM here:

      * No gradient required  →  works on static-INT8 quantised models
        where autograd is unavailable.
      * Deterministic         →  no numerical noise from backward pass.
      * Identical formulation across precisions   →  any heatmap difference
        is attributable to quantisation, not to implementation drift.

    CAM_c(h, w) = sum_k  w_{c,k} * A_k(h, w)

    where A_k are the last-block feature maps (before GAP) and w_{c,k} are
    the classifier weights for class c, channel k.

    Reported per (image, pathology) pair:
      * Pearson correlation of the raw heatmaps
      * IoU of top-20% thresholded regions
      * Centre-of-mass shift (pixels)

    Supports the dissertation's clinical-safety argument: if CAMs agree,
    the INT8 model is attending to the same anatomical regions as FP32.
    Caveat: labels on ChestX-ray14 are known-noisy (Oakden-Rayner 2017),
    so we claim preservation of attention, not clinical correctness.
    """

    def __init__(self, pathologies, target_layer_name='features'):
        self.pathologies = pathologies
        self.target_layer_name = target_layer_name
        self._activations = {}
        self._hooks = []

    def _register_hook(self, model, key):
        """Attach a forward hook on the last feature map (pre-GAP)."""
        # torchxrayvision DenseNet exposes .features (Sequential ending in norm5).
        # After eager-mode quantisation the model may be wrapped (QuantWrapper,
        # custom wrapper, etc.), so we try several strategies:

        target = None

        # Strategy 1: direct attribute access (most reliable — survives wrapping)
        target = getattr(model, self.target_layer_name, None)
        if target is None:
            # Common wrapper patterns: model.module.features, model.model.features
            for wrapper_attr in ('module', 'model', '_model'):
                inner = getattr(model, wrapper_attr, None)
                if inner is not None:
                    target = getattr(inner, self.target_layer_name, None)
                    if target is not None:
                        break

        # Strategy 2: named_modules search (handles deeper nesting)
        if target is None:
            for name, module in model.named_modules():
                if (name == self.target_layer_name
                        or name.endswith('.' + self.target_layer_name)):
                    target = module
                    break

        # Strategy 3: find the last BatchNorm / conv whose channel count
        # matches the classifier's in_features (1024 for DenseNet-121).
        # This is far more robust than "last Conv2d" which grabs a 32-ch
        # growth-rate conv inside a DenseLayer.
        if target is None:
            try:
                W = self._classifier_weights(model)
                expected_C = W.size(1)  # 1024
                candidates = []
                for _name, m in model.named_modules():
                    out_c = getattr(m, 'num_features', None) \
                            or getattr(m, 'out_channels', None)
                    if out_c == expected_C:
                        candidates.append(m)
                if candidates:
                    target = candidates[-1]
            except Exception:
                pass

        # Strategy 4: ultimate fallback — last conv-like layer
        if target is None:
            conv_type_names = {
                'Conv2d', 'QuantizedConv2d', 'QuantizedConvReLU2d',
                'ConvReLU2d', 'ConvBn2d', 'ConvBnReLU2d',
            }
            convs = [m for m in model.modules()
                     if isinstance(m, nn.Conv2d)
                     or type(m).__name__ in conv_type_names]
            if not convs:
                raise RuntimeError("No Conv2d layer found for CAM hook")
            target = convs[-1]

        def hook(_mod, _inp, out):
            # Store a dequantised float copy (works for both float and quint8)
            if hasattr(out, 'dequantize'):
                try:
                    out = out.dequantize()
                except Exception:
                    pass
            self._activations[key] = out.detach().float().cpu()

        h = target.register_forward_hook(hook)
        self._hooks.append(h)
        return target

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._activations = {}

    def _classifier_weights(self, model):
        """
        Locate the final Linear classifier and return its weight matrix
        (n_classes, n_channels). Handles torchxrayvision's .classifier
        attribute and quantised-linear variants.
        """
        cls = getattr(model, 'classifier', None)
        if cls is None:
            # Search for the final Linear
            linears = [m for m in model.modules()
                       if isinstance(m, (nn.Linear,))
                       or type(m).__name__ in ('LinearPackedParams', 'Linear')]
            if not linears:
                raise RuntimeError("No classifier Linear found")
            cls = linears[-1]

        # Quantised linear exposes weight via _packed_params
        if hasattr(cls, 'weight') and callable(getattr(cls, 'weight', None)) is False \
           and isinstance(cls.weight, torch.Tensor):
            W = cls.weight.detach().float().cpu()
        elif hasattr(cls, '_packed_params'):
            w_packed = cls._packed_params._packed_params
            # Returns (weight, bias) for quantised Linear
            W = w_packed[0].dequantize().detach().float().cpu()
        elif callable(getattr(cls, 'weight', None)):
            W = cls.weight().dequantize().detach().float().cpu()
        else:
            W = cls.weight.detach().float().cpu()
        return W

    def compute_cam(self, model, img_tensor, class_idx):
        """
        Compute the CAM for one image, one class.
        Returns a (H, W) numpy array normalised to [0, 1].
        """
        self._clear_hooks()
        self._register_hook(model, 'A')

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                _ = model(img_tensor.float())
        finally:
            if was_training:
                model.train()

        if 'A' not in self._activations:
            self._clear_hooks()
            raise RuntimeError("Forward hook captured no activations")

        A = self._activations['A']  # (1, C, H, W) on CPU float32
        self._clear_hooks()

        # Pass through any post-features ops (norm5 + relu) if the hook was on
        # 'features' (which is pre-norm5 in torchxrayvision).
        # Apply a ReLU as a conservative proxy for the post-feature activation.
        A = F.relu(A)

        W = self._classifier_weights(model)  # (n_classes, C)
        if A.size(1) != W.size(1):
            # Mismatch — try transposing or bailing gracefully
            raise RuntimeError(
                f"CAM channel mismatch: features C={A.size(1)}, "
                f"classifier in_features={W.size(1)}"
            )

        w_c = W[class_idx]  # (C,)
        cam = torch.einsum('c,bchw->bhw', w_c, A)  # (1, H, W)
        cam = cam[0].numpy()

        # Normalise to [0, 1] for visual / metric comparison
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    @staticmethod
    def agreement_metrics(cam_a, cam_b, threshold_pct=0.80):
        """
        Compute agreement between two CAMs.

        Returns dict with:
          pearson:    correlation of flattened heatmaps
          iou:        IoU of regions above the threshold_pct quantile
          com_shift:  centre-of-mass shift in pixels (L2)
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

    def compare_models(self, models_dict, images, pathology_indices=None,
                       max_images=20, verbose=True):
        """
        Compute CAMs for each (model, image, pathology) triple and report
        pairwise agreement against the FP32 baseline.

        Args:
          models_dict: {'FP32': model_fp32, 'INT8': model_int8, ...}
                       FP32 MUST be present as the reference key.
          images: list of image tensors (same shape the models expect)
          pathology_indices: list of class indices to probe; defaults to
                             [Cardiomegaly, Pneumothorax, Effusion] which
                             tend to have clear spatial signatures.
          max_images: cap on images to process (these are slow)

        Returns:
          summary dict:
            {'per_pair': {'INT8_vs_FP32': {'pearson_mean': ..., 'iou_mean': ...}},
             'per_image': [...]}
        """
        if 'FP32' not in models_dict:
            raise ValueError("models_dict must contain key 'FP32'")

        if pathology_indices is None:
            # Default picks with reasonably clear spatial patterns
            defaults = ['Cardiomegaly', 'Pneumothorax', 'Effusion']
            pathology_indices = [self.pathologies.index(p)
                                 for p in defaults if p in self.pathologies]
            if not pathology_indices:
                pathology_indices = [0, 1, 2]

        n = min(max_images, len(images))
        per_image = []
        per_pair_acc = {}

        other_keys = [k for k in models_dict if k != 'FP32']

        if verbose:
            print("\n" + "=" * 60)
            print("GRAD-CAM / CAM COMPARISON")
            print("=" * 60)
            print(f"  Images: {n}    "
                  f"Pathologies: {[self.pathologies[i] for i in pathology_indices]}")
            print(f"  Models: FP32 (ref) vs {', '.join(other_keys)}")

        for img_i in range(n):
            img = images[img_i]
            if img.dim() == 3:
                img = img.unsqueeze(0)

            for cls_idx in pathology_indices:
                # FP32 reference CAM
                try:
                    cam_ref = self.compute_cam(models_dict['FP32'], img, cls_idx)
                except Exception as e:
                    if verbose:
                        print(f"  [img {img_i} cls {cls_idx}] FP32 CAM failed: {e}")
                    continue

                for key in other_keys:
                    try:
                        cam_k = self.compute_cam(models_dict[key], img, cls_idx)
                    except Exception as e:
                        if verbose:
                            print(f"  [img {img_i} cls {cls_idx}] {key} CAM failed: {e}")
                        continue

                    # Resize to match if needed (INT8 paths may have slightly
                    # different feature-map resolutions in edge cases)
                    if cam_k.shape != cam_ref.shape:
                        cam_k_t = torch.from_numpy(cam_k).unsqueeze(0).unsqueeze(0)
                        cam_k = F.interpolate(
                            cam_k_t, size=cam_ref.shape, mode='bilinear',
                            align_corners=False
                        )[0, 0].numpy()

                    metrics = self.agreement_metrics(cam_ref, cam_k)
                    pair_key = f"{key}_vs_FP32"
                    per_pair_acc.setdefault(pair_key, []).append(metrics)

                    per_image.append({
                        'image_idx': img_i,
                        'pathology': self.pathologies[cls_idx],
                        'precision': key,
                        **metrics,
                    })

        # Aggregate
        summary = {'per_pair': {}, 'per_image': per_image}
        for pair_key, metrics_list in per_pair_acc.items():
            if not metrics_list:
                continue
            pearsons = [m['pearson'] for m in metrics_list
                        if not np.isnan(m['pearson'])]
            ious = [m['iou'] for m in metrics_list
                    if not np.isnan(m['iou'])]
            coms = [m['com_shift_px'] for m in metrics_list]
            summary['per_pair'][pair_key] = {
                'n': len(metrics_list),
                'pearson_mean': float(np.mean(pearsons)) if pearsons else float('nan'),
                'pearson_std': float(np.std(pearsons)) if pearsons else float('nan'),
                'iou_mean': float(np.mean(ious)) if ious else float('nan'),
                'iou_std': float(np.std(ious)) if ious else float('nan'),
                'com_shift_px_mean': float(np.mean(coms)) if coms else float('nan'),
            }
            if verbose:
                s = summary['per_pair'][pair_key]
                print(f"  {pair_key:<20} n={s['n']:>3}  "
                      f"Pearson={s['pearson_mean']:.3f}±{s['pearson_std']:.3f}  "
                      f"IoU@80%={s['iou_mean']:.3f}±{s['iou_std']:.3f}  "
                      f"COM_shift={s['com_shift_px_mean']:.2f}px")

        return summary

    def save_comparison_figure(self, models_dict, images, image_idx,
                               pathology_idx, out_path):
        """
        Save a side-by-side figure: original image + overlaid CAMs per
        precision. One image, one pathology.
        """
        if not HAS_MPL:
            print("  matplotlib not available, skipping figure")
            return
        img = images[image_idx]
        if img.dim() == 3:
            img = img.unsqueeze(0)

        cams = {}
        for key, model in models_dict.items():
            try:
                cams[key] = self.compute_cam(model, img, pathology_idx)
            except Exception as e:
                print(f"  {key} CAM failed: {e}")

        n_panels = 1 + len(cams)
        fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.4))
        if n_panels == 1:
            axes = [axes]

        # Original X-ray (denormalised rough display)
        img_np = img[0, 0].detach().float().cpu().numpy()
        img_np = (img_np - img_np.min()) / (np.ptp(img_np) + 1e-9)
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title('X-ray')
        axes[0].axis('off')

        # CAM overlays
        for i, (key, cam) in enumerate(cams.items(), start=1):
            # Upsample CAM to image size
            cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
            cam_up = F.interpolate(
                cam_t, size=img_np.shape, mode='bilinear', align_corners=False
            )[0, 0].numpy()
            axes[i].imshow(img_np, cmap='gray')
            axes[i].imshow(cam_up, cmap='jet', alpha=0.45)
            axes[i].set_title(f'{key}')
            axes[i].axis('off')

        path_name = self.pathologies[pathology_idx]
        fig.suptitle(f"CAM comparison — {path_name} (image #{image_idx})")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved CAM figure → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CALIBRATION-SIZE SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def calibration_size_sweep(exp, calib_sizes=(32, 128, 256, 512, 1024),
                           eval_images=None, labels_matrix=None,
                           verbose=True):
    """
    Ablate INT8 static-quantisation calibration sample size.

    For each N in calib_sizes:
      - Prepare a fresh INT8 model using N calibration images
      - Run inference on the evaluation set
      - Report per-pathology AUC (mean) and max-probability-diff vs FP32

    Uses exp's existing quantisation machinery (_try_eager_quantisation and
    _try_fx_standard) so the calibration logic is identical to the main
    pipeline — only the sample count varies.

    Supports examiner questions of the form "how sensitive are your INT8
    results to the calibration set size?" which is a standard PTQ ablation
    (see Gholami 2021 survey, Nagel 2020).
    """
    if verbose:
        print("\n" + "=" * 60)
        print("CALIBRATION SIZE SWEEP")
        print("=" * 60)

    torch.backends.quantized.engine = 'fbgemm'
    eval_images = eval_images if eval_images is not None else exp.test_images
    n_eval = len(eval_images)

    # Precompute FP32 reference probabilities
    fp32_model = copy.deepcopy(exp.model).float().eval()
    with torch.no_grad():
        fp32_outs = torch.stack(
            [fp32_model(img.float())[0] for img in eval_images]
        )
    fp32_probs = torch.sigmoid(fp32_outs).numpy()

    results = {}
    available_calib = len(exp.test_images)

    for n_calib in calib_sizes:
        if n_calib > available_calib:
            if verbose:
                print(f"  Skipping N={n_calib} (only {available_calib} images available)")
            continue
        if verbose:
            print(f"\n  Calibrating with N={n_calib} images...")

        model_fp32 = copy.deepcopy(exp.model).float().eval()
        prepared = None
        mode = None
        for fn in [exp._try_fx_standard, exp._try_fx_safe_tracer,
                   exp._try_eager_quantisation]:
            try:
                out = fn(model_fp32)
                if out is not None:
                    prepared, mode = out
                    break
            except Exception:
                continue
        if prepared is None:
            if verbose:
                print(f"    Quantisation prep failed at N={n_calib}")
            continue

        with torch.no_grad():
            for img in exp.test_images[:n_calib]:
                prepared(img.float())

        try:
            if mode == 'fx_standard':
                from torch.ao.quantization import quantize_fx
                int8_model = quantize_fx.convert_fx(prepared)
            else:
                int8_model = torch.quantization.convert(prepared)
        except Exception as e:
            if verbose:
                print(f"    Convert failed: {e}")
            continue

        # Inference
        with torch.no_grad():
            int8_outs = []
            for img in eval_images:
                try:
                    int8_outs.append(int8_model(img.float())[0])
                except Exception as e:
                    if verbose:
                        print(f"    Inference failed: {e}")
                    int8_outs = None
                    break
        if int8_outs is None:
            continue
        int8_outs = torch.stack(int8_outs)
        int8_probs = torch.sigmoid(int8_outs).numpy()

        max_diff = float(np.max(np.abs(fp32_probs - int8_probs)))
        mean_diff = float(np.mean(np.abs(fp32_probs - int8_probs)))

        # Per-pathology AUC (if labels available)
        auc_mean = None
        if labels_matrix is not None and HAS_SKLEARN:
            aucs = []
            n_paths = min(int8_probs.shape[1], labels_matrix.shape[1])
            for j in range(n_paths):
                y = labels_matrix[:n_eval, j]
                if y.sum() < 2 or y.sum() > len(y) - 2:
                    continue
                try:
                    aucs.append(roc_auc_score(y, int8_probs[:n_eval, j]))
                except Exception:
                    continue
            auc_mean = float(np.mean(aucs)) if aucs else None

        results[n_calib] = {
            'n_calib': n_calib,
            'max_prob_diff': max_diff,
            'mean_prob_diff': mean_diff,
            'auc_mean': auc_mean,
            'quant_mode': mode,
        }
        if verbose:
            auc_str = f"{auc_mean:.4f}" if auc_mean is not None else "n/a"
            print(f"    N={n_calib:>4}  max|Δp|={max_diff:.4f}  "
                  f"mean|Δp|={mean_diff:.4f}  AUC={auc_str}  mode={mode}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. IDLE-POWER BASELINE MEASUREMENT (for RAPL)
# ═══════════════════════════════════════════════════════════════════════════════

def measure_idle_baseline(duration_s=3.0, verbose=True):
    """
    Measure package power while the CPU is idle (no model inference).

    Follows UPM (2024) methodology: per-inference Joules should be reported
    as (busy_power - idle_power) * time, so that the figure captures the
    marginal cost of inference rather than the machine's resting draw.

    Returns:
      dict with:
        'idle_watts': mean package power during idle (W), or None if RAPL
                      is unavailable.
        'method': 'rapl' or 'unavailable'
        'duration_s': measurement window

    If RAPL is virtualised away (common on Azure/AWS/GCP VMs), returns
    None for idle_watts and prints a note. Callers should then fall back
    to TDP estimation without idle subtraction.
    """
    rapl_base = '/sys/class/powercap/intel-rapl'

    def read_total_uj():
        files = glob.glob(os.path.join(rapl_base, '*/energy_uj'))
        total = 0
        for f in files:
            try:
                with open(f, 'r') as fh:
                    total += int(fh.read().strip())
            except (IOError, ValueError):
                pass
        return total

    if not os.path.exists(rapl_base):
        if verbose:
            print("  RAPL not available — idle subtraction skipped")
        return {'idle_watts': None, 'method': 'unavailable', 'duration_s': 0.0}

    # Try a read; if it fails (perm errors on VM), give up cleanly
    try:
        e0 = read_total_uj()
        if e0 == 0:
            if verbose:
                print("  RAPL counters read as 0 — likely virtualised")
            return {'idle_watts': None, 'method': 'unavailable',
                    'duration_s': 0.0}
    except Exception:
        return {'idle_watts': None, 'method': 'unavailable', 'duration_s': 0.0}

    if verbose:
        print(f"\n  Measuring idle baseline ({duration_s:.1f}s)...")

    t0 = time.time()
    time.sleep(duration_s)
    e1 = read_total_uj()
    elapsed = time.time() - t0
    joules = (e1 - e0) / 1e6
    watts = joules / elapsed if elapsed > 0 else 0.0

    if verbose:
        print(f"    Idle package power: {watts:.2f}W over {elapsed:.2f}s "
              f"({joules:.3f}J)")

    return {
        'idle_watts': float(watts),
        'idle_joules': float(joules),
        'duration_s': float(elapsed),
        'method': 'rapl',
    }


def marginal_energy_per_inference(total_joules, time_s, idle_watts):
    """
    Subtract idle baseline from a measured energy window.

    E_marginal = E_total - (idle_watts * time_s)

    Returns joules (may be negative if the model is cooler than idle,
    which would indicate a measurement artefact; we clamp at 0).
    """
    if idle_watts is None:
        return total_joules
    marginal = total_joules - idle_watts * time_s
    return max(0.0, marginal)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_additions(exp, do_gradcam=True, do_calib_sweep=True,
                  do_idle_baseline=True, output_dir='.'):
    """
    Run all three additions against a populated PrecisionExperiment instance.

    Args:
      exp: PrecisionExperiment after it has loaded images and run
           run_static_quantisation() / run_native_precision().
      output_dir: where to drop CAM figures.
    """
    additions_results = {}

    if do_idle_baseline:
        additions_results['idle_baseline'] = measure_idle_baseline(duration_s=3.0)

    if do_calib_sweep:
        labels = getattr(exp, 'ground_truth_labels', None)
        additions_results['calibration_sweep'] = calibration_size_sweep(
            exp, eval_images=exp.test_images, labels_matrix=labels
        )

    if do_gradcam:
        # Assemble the models we have available
        models = {'FP32': copy.deepcopy(exp.model).float().eval()}

        # Prefer a real static INT8 model if newmain.py exposed one
        static_int8 = exp.results.get('static_int8', {}).get('_model')
        if static_int8 is None:
            static_int8 = getattr(exp, 'static_int8_model', None)

        # Otherwise, build one on the fly using exp's own prep path
        if static_int8 is None:
            try:
                torch.backends.quantized.engine = 'fbgemm'
                model_fp32 = copy.deepcopy(exp.model).float().eval()
                prepared, mode = None, None
                for fn in [getattr(exp, '_try_fx_standard', None),
                           getattr(exp, '_try_fx_safe_tracer', None),
                           getattr(exp, '_try_eager_quantisation', None)]:
                    if fn is None:
                        continue
                    try:
                        out = fn(model_fp32)
                        if out is not None:
                            prepared, mode = out
                            break
                    except Exception:
                        continue
                if prepared is not None:
                    with torch.no_grad():
                        for img in exp.test_images[:128]:
                            prepared(img.float())
                    if mode == 'fx_standard':
                        from torch.ao.quantization import quantize_fx
                        static_int8 = quantize_fx.convert_fx(prepared)
                    else:
                        static_int8 = torch.quantization.convert(prepared)
            except Exception as e:
                print(f"  Could not build INT8 model for CAM: {e}")

        if static_int8 is not None:
            models['INT8'] = static_int8
        # A simulated BF16 / INT8 path (force_precision) as alternative
        try:
            if hasattr(exp, '_simulated_int8_model'):
                models['INT8_simulated'] = exp._simulated_int8_model
        except Exception:
            pass

        analyser = GradCAMAnalyser(exp.pathologies if hasattr(exp, 'pathologies')
                                   else [
                                       'Atelectasis', 'Cardiomegaly',
                                       'Consolidation', 'Edema', 'Effusion',
                                       'Emphysema', 'Fibrosis', 'Hernia',
                                       'Infiltration', 'Mass', 'Nodule',
                                       'Pleural_Thickening', 'Pneumonia',
                                       'Pneumothorax'
                                   ])
        additions_results['cam_comparison'] = analyser.compare_models(
            models, exp.test_images, max_images=20
        )
        # Save a figure for the first image, Cardiomegaly
        try:
            card_idx = analyser.pathologies.index('Cardiomegaly')
            analyser.save_comparison_figure(
                models, exp.test_images,
                image_idx=0, pathology_idx=card_idx,
                out_path=os.path.join(output_dir, 'cam_comparison.png')
            )
        except Exception as e:
            print(f"  CAM figure failed: {e}")

    return additions_results


if __name__ == '__main__':
    print("This module is imported by newmain.py — see run_additions().")
