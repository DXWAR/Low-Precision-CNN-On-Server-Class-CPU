# Low-Precision  CNN  on Server-Class CPUs

## Overview

This repository contains the full experimental code and results for a dissertation investigating whether low-precision numerical formats (BF16, FP16, INT8) can accelerate DenseNet-121 chest X-ray inference on server-class CPUs without compromising diagnostic accuracy.

The study targets **AMD EPYC 9005 (Zen 5)** processors with AVX-512 VNNI and AVX-512 BF16 instruction support, using PyTorch's `fbgemm` quantisation backend. The clinical workload is multi-label thoracic pathology classification on the NIH ChestX-ray14 dataset via a pretrained `torchxrayvision` DenseNet-121 checkpoint.

## Key Findings

- **Native BF16** inference delivers a **1.41× speedup** with probability deviations well within the 1.0% clinical-safety threshold.
- **Dynamic INT8** quantisation achieves a **2.13× speedup** but exceeds the safety threshold for certain pathologies.
- **Energy savings** of up to 29% are achievable with BF16 at no diagnostic cost.
- Mixed-precision strategies (per-layer/per-block) do not outperform uniform BF16 — block-level granularity is the practical sweet spot.

## Repository Structure

```
.
├── newmain.py                      # Full 13-experiment pipeline (main script)
├── additions.py                    # Supplementary analyses (Grad-CAM, calibration sweep, idle power)
├── Data_Entry_2017_v2020.csv       # NIH ChestX-ray14 ground-truth labels
├── densenet121_sample/             # 1,000 sample chest X-ray images (one per patient)
├── tests/                          # Pytest test suite (T1–T24 unit + I1–I3 integration)
│   ├── conftest.py                 # Shared fixtures (tiny DenseNet, dummy inputs, seeds)
│   ├── test_quantisation_wrapper.py
│   ├── test_statistical_analysis.py
│   ├── test_mixed_precision.py
│   ├── test_rapl_energy.py
│   ├── test_agreement_metrics.py
│   ├── test_gradcam.py
│   ├── test_calibration_determinism.py
│   └── test_integration.py
├── Results/                        # Output figures and raw JSON data
│   ├── demo_results.json
│   ├── results_main.png
│   ├── results_detailed.png
│   ├── results_hardware.png
│   ├── results_energy.png
│   ├── weight_distributions.png
│   ├── latency_drift.png
│   ├── latency_histogram.png
│   └── cam_comparison.png
└── README.md
```

## Experiments

`newmain.py` runs a 13-step experimental pipeline:

1. **Precision spectrum** — FP64, FP32, BF16, FP16, INT8 simulated baselines
2. **Dynamic INT8 quantisation** — `torch.quantization.quantize_dynamic`
3. **Per-channel vs per-tensor INT8** — granularity comparison
4. **Block sensitivity analysis** — which DenseNet dense blocks tolerate low precision
5. **Layer sensitivity analysis** — per-layer sensitivity with type breakdown
6. **Clinical accuracy evaluation** — per-pathology probability differences
7. **McNemar's test** — binary classification equivalence testing
8. **Mixed-precision forward pass** — distance-based and entropy-based strategies
9. **Statistical analysis** — bootstrap CIs, p-values, Cohen's d effect sizes
10. **Native BF16/FP16 inference** — `torch.cpu.amp.autocast` hardware paths
11. **Static INT8 quantisation** — `torch.quantization` prepare/convert via VNNI
12. **Energy and carbon footprint** — TDP-based and RAPL measurement
13. **figures** — error bars, heatmaps, energy plots

`additions.py` provides three supplementary analyses:

- **Grad-CAM comparison** — FP32 vs INT8 class activation maps with IoU and Pearson agreement
- **Calibration size sweep** — INT8 calibration over N = 32, 128, 512, 1024 samples
- **Idle power baseline** — package-level idle power measurement (UPM 2024 methodology)

## Requirements

- Python 3.9+
- An x86-64 CPU with AVX-512 support (for INT8/BF16 hardware-accelerated paths)

### Install dependencies

```bash
pip install torch torchvision torchxrayvision numpy matplotlib Pillow scipy scikit-learn tqdm
```

## Usage

### Running experiments

```bash
# Full pipeline (default: 1,000 images)
python newmain.py

# Quick run (100 images, fewer layers analysed)
python newmain.py --quick

# Custom image count
python newmain.py --images 500
```

### Running supplementary analyses

```python
from newmain import PrecisionExperiment
from additions import run_additions

exp = PrecisionExperiment()
exp.load_model_and_data()
# ... run main experiments ...
run_additions(exp)
```

### Running the test suite

```bash
# Run all tests
pytest tests/ -v

# Run a specific test module
pytest tests/test_quantisation_wrapper.py -v
```

Tests use a lightweight toy DenseNet fixture (no dataset download or GPU required) and complete in under 30 seconds.

## Data

The `densenet121_sample/` directory contains 1,000 images sampled from the [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) dataset (one image per patient). `Data_Entry_2017_v2020.csv` provides the corresponding ground-truth pathology labels.

## Hardware

All experiments were run on:

- **CPU**: AMD EPYC 9V45 96-Core Processor (Azure Standard_F4s_v2, 4 vCPUs / 2 physical cores)
- **ISA**: AVX-512 VNNI (INT8), AVX-512 BF16
- **Backend**: PyTorch 2.10.0 fbgemm (vendor-agnostic — works on both Intel and AMD AVX-512 CPUs)

## Licence

This project is part of an academic dissertation. Please cite appropriately if referencing this work.

## Author

Dawar Mohammadi 
