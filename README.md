# CPDS-AD Detection Methods

Detection and validation code for the paper:

**A Cyber-Attack Detection Dataset for Distribution Systems from a Scaled Cyber-physical Testbed**

This folder contains the code used to evaluate the `CPDS-AD` dataset on cyber-attack detection tasks in cyber-physical distribution systems.

## Overview

The repository covers two tasks:

- `FDI attack detection` with unsupervised deep learning models:
  `Transformer`, `LSTM`, and `TCN`
- `DoS attack detection` with traffic-based anomaly detection methods:
  `Z-score`, `Isolation Forest`, and `KNN`

The dataset is derived from a laboratory-scale cyber-physical distribution system testbed that emulates a 13-node three-phase unbalanced radial distribution network and a three-layer cyber architecture.

## Dataset Files

Processed datasets are already included in `CPDS-AD_dataset/`.

| File | Description | Shape |
|---|---|---:|
| `train_data.xlsx` | Normal-operation training set | `14400 x 132` |
| `test_data_A.xlsx` | Additive FDI test set | `2880 x 133` |
| `test_data_S.xlsx` | Subtractive FDI test set | `2880 x 133` |
| `test_data_R.xlsx` | Replay FDI test set | `2880 x 133` |
| `test_data_D.xlsx` | DoS traffic test set | `1440 x 2` |

Notes:

- FDI test files use the last column as the ground-truth label.
- DoS traffic data contains `Bytes` and `Labels`.
- Raw scenario folders are also included for traceability and reprocessing.

## Repository Structure

```text
detection_methods/
├─ my_main.py                  # Main entry for FDI training/testing
├─ DoS_detection.py            # DoS detection
├─ plot_umap.py                # UMAP visualization for FDI windows
├─ models/                     # Model definitions
├─ solvers/                    # Training/testing logic
├─ utils/                      # Data loading and processing utilities
├─ checkpoints/                # Saved model checkpoints
└─ CPDS-AD_dataset/            # Processed and raw dataset files
```

## Requirements

Recommended environment:

- Python `3.10+` or `3.11`

Tested package versions:

- `pandas==2.3.3`
- `numpy==1.26.2`
- `scikit-learn==1.8.0`
- `matplotlib==3.7.2`
- `seaborn==0.13.0`
- `openpyxl==3.1.2`
- `umap-learn==0.5.11`

Install dependencies with:

```bash
pip install torch pandas==2.3.3 numpy==1.26.2 scikit-learn==1.8.0 matplotlib==3.7.2 seaborn==0.13.0 openpyxl==3.1.2 umap-learn==0.5.11
```

## Quick Start

Run all FDI evaluations:

```bash
python my_main.py
```

This runs `Transformer`, `LSTM`, and `TCN` on the three FDI scenarios and saves:

- `FDI_detection_performances.xlsx`

Run DoS detection:

```bash
python DoS_detection.py
```

This saves:

- `DoS_detection_performances.xlsx`

Generate the UMAP figures used for technical validation:

```bash
python plot_umap.py
```

This generates three figures for the additive, subtractive, and replay FDI scenarios:

- `UMAP_windows_visualization_A.png`
- `UMAP_windows_visualization_S.png`
- `UMAP_windows_visualization_R.png`

Because UMAP involves stochastic optimization and device-dependent floating-point computation, the exact plotted layout may vary across different machines, but the class separation patterns and the overall conclusion are not affected.

## FDI Model Usage

### Test a single model

Transformer:

```bash
python my_main.py --solver solver_transformer --transformer_mode test
```

LSTM:

```bash
python my_main.py --solver solver_LSTM --LSTM_mode test
```

TCN:

```bash
python my_main.py --solver solver_TCN --TCN_mode test
```

### Train a single model

Transformer:

```bash
python my_main.py --solver solver_transformer --transformer_mode train
```

LSTM:

```bash
python my_main.py --solver solver_LSTM --LSTM_mode train
```

TCN:

```bash
python my_main.py --solver solver_TCN --TCN_mode train
```

### Change the test file

Example for testing TCN on additive FDI data:

```bash
python my_main.py --solver solver_TCN --TCN_mode test --TCN_testdata_path CPDS-AD_dataset/test_data_A.xlsx
```

## Outputs

Typical outputs include:

- `checkpoints/transformer/checkpoint.pth`
- `checkpoints/LSTM/checkpoint.pth`
- `checkpoints/TCN/checkpoint.pth`
- `FDI_detection_performances.xlsx`
- `DoS_detection_performances.xlsx`
- `UMAP_windows_visualization_*.png`

## Reproducibility

FDI testing for `Transformer`, `LSTM`, and `TCN` has been updated to be reproducible for the same checkpoint and the same test file.

Available seed arguments:

- `--transformer_seed`
- `--LSTM_seed`
- `--TCN_seed`

Default seed:

```bash
42
```

## Notes

- Run scripts from this folder to avoid relative path issues.
- Preprocessed datasets are already provided, so reprocessing is usually not necessary.
- Some data processing scripts contain hard-coded paths from the original experimental environment and may need adjustment before reuse.

## Citation

If this code or dataset is useful in your work, please cite:

```text
Yulin Liu, Zhaojun Ruan, Libao Shi.
A Cyber-Attack Detection Dataset for Distribution Systems from a Scaled Cyber-physical Testbed.
```
