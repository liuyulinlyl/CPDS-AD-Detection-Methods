# CPDS-AD Detection Methods

[CPDS-AD dataset (figshare)](https://doi.org/10.6084/m9.figshare.31989933) | [Archived code (Zenodo)](https://doi.org/10.5281/zenodo.19562168) | [GitHub repository](https://github.com/liuyulinlyl/CPDS-AD-Detection-Methods)

This repository contains the data-processing, visualization, and attack-detection code accompanying the manuscript:

> **A Cyber-Attack Detection Dataset for Distribution Systems Generated from a Scaled Cyber-Physical Testbed**  
> Yulin Liu, Zhaojun Ruan, and Libao Shi

The included benchmarks cover false data injection (FDI) detection with unsupervised deep reconstruction models and denial-of-service (DoS) detection from communication-traffic time series. Preprocessed datasets and pretrained FDI checkpoints are included so that the validation experiments can be run directly.

## About CPDS-AD

CPDS-AD was collected from a laboratory-level cyber-physical distribution-system testbed with:

- a scaled 13-node, three-phase unbalanced, medium-voltage-type radial feeder;
- line-impedance, three-phase controllable-load, and single-phase controllable-load simulation cabinets;
- a three-layer cyber architecture comprising application, gateway, and perception layers;
- MQTT communication between the master-station system and intelligent gateway, followed by Modbus communication with field devices;
- voltage, current, active-power, reactive-power, raw-message, and communication-traffic records.

The laboratory feeder uses a 380 V source for safe operation. It preserves the topology, phase configuration, and steady-state relationships of the emulated feeder; it is not intended as a one-to-one model of a practical 380 V low-voltage network. The electrical quantities in the dataset are actual hardware measurements without post-processing scale factors.

Measurements were polled at 5 s intervals. Each 24 h load profile was compressed by a factor of 30 into a 48 min physical experiment. The load profiles were derived from selected SMART-DS daily curves and cover different weekdays, holidays, and seasons.

## Attack scenarios

| Family | Code | Scenarios | Description |
|---|---|---:|---|
| Additive FDI | `A` | 5 | Applies a small upward proportional bias to selected measurements. |
| Subtractive FDI | `S` | 5 | Applies a small downward proportional bias to selected measurements. |
| Replay FDI | `R` | 5 | Replaces current measurements with previously recorded normal values. |
| Ramp-drift FDI | `RD` | 5 | Uses a smooth rise-hold-fall perturbation profile with a maximum relative deviation of 1%. |
| Local model-consistent FDI | `LMC` | 5 | Coordinates voltage, current, active-power, and reactive-power changes to preserve local physical consistency. |
| Adversarial-learning FDI | `AL` | 5 | Constructs stealth-oriented perturbations from learned normal-data patterns. |
| Polling-aware DoS | `D` | 6 | Injects spoofed traffic at low, medium, or high intensity while disrupting polling responses. |

The baseline additive and subtractive attacks use nominal proportional biases of approximately 1%, with limited variation between attacked channels. The DoS data are divided into two scenarios at each intensity; the mean injected spoofed-message rates reported in the paper are 230, 370, and 460 bytes/s for the low-, medium-, and high-intensity cases, respectively.

## Repository structure

```text
detection_methods/
|-- my_main.py                     # FDI training and evaluation entry point
|-- DoS_detection.py               # Traffic-based DoS evaluation
|-- plot_umap.py                   # Six per-scenario FDI UMAP figures
|-- combine_umap_figures.py        # Combined 2 x 3 UMAP figure
|-- models/                        # Transformer, LSTM, and TCN definitions
|-- solvers/                       # FDI training and point-level evaluation logic
|-- utils/                         # Time-series data loading and windowing
|-- checkpoints/                   # Supplied pretrained FDI checkpoints
|-- CPDS-AD_dataset/
|   |-- merged_datasets/           # Analysis-ready training and test workbooks
|   `-- scenario_records/          # Per-experiment raw logs and processed records
|-- FDI_detection_performances.xlsx
`-- DoS_detection_performances.xlsx
```

## Dataset organization

### Analysis-ready merged datasets

All benchmark scripts read from `CPDS-AD_dataset/merged_datasets/`.

| File(s) | Samples per file | Analysis columns | Purpose |
|---|---:|---|---|
| `train_data.xlsx` | 14,400 | 132 measurement features | Normal-operation training data |
| `test_data_A.xlsx` | 2,880 | 132 features + `labels` | Additive FDI test data |
| `test_data_S.xlsx` | 2,880 | 132 features + `labels` | Subtractive FDI test data |
| `test_data_R.xlsx` | 2,880 | 132 features + `labels` | Replay FDI test data |
| `test_data_RD.xlsx` | 2,880 | 132 features + `labels` | Ramp-drift FDI test data |
| `test_data_LMC.xlsx` | 2,880 | 132 features + `labels` | Local model-consistent FDI test data |
| `test_data_AL.xlsx` | 2,880 | 132 features + `labels` | Adversarial-learning FDI test data |
| `test_data_D_low.xlsx` | 1,152 | `Time`, `Traffic_volume`, `Labels` | Low-intensity DoS test data |
| `test_data_D_medium.xlsx` | 1,152 | `Time`, `Traffic_volume`, `Labels` | Medium-intensity DoS test data |
| `test_data_D_high.xlsx` | 1,152 | `Time`, `Traffic_volume`, `Labels` | High-intensity DoS test data |

The Excel workbooks preserve a serialized row-index column. The FDI loader reads it with `index_col=0`, so the logical dimensions shown above exclude that index. For both FDI and DoS data, label `0` denotes normal operation and label `1` denotes an attacked sample. `Traffic_volume` is the number of message bytes aggregated in each consecutive 5 s window.

The 132 FDI features combine the source meter identifier and measurement variable. Common variable names and units are:

| Columns | Meaning | Unit |
|---|---|---|
| `U_a`, `U_b`, `U_c` | Phase-voltage magnitudes | V |
| `U_ab`, `U_bc`, `U_ac` | Line-to-line voltage magnitudes | V |
| `I_a`, `I_b`, `I_c` | Phase-current magnitudes | A |
| `P_a`, `P_b`, `P_c` | Phase active power | W |
| `Q_a`, `Q_b`, `Q_c` | Phase reactive power | Var |
| `U`, `I`, `P`, `Q` | Corresponding single-phase measurements | V, A, W, Var |

### Per-scenario records

```text
CPDS-AD_dataset/scenario_records/
|-- normal_operation/
|   `-- train_data_1/ ... train_data_25/
|-- FDI_attacks/
|   |-- additive/test_data_A_1/ ... test_data_A_5/
|   |-- subtractive/test_data_S_1/ ... test_data_S_5/
|   |-- replay/test_data_R_1/ ... test_data_R_5/
|   |-- ramp_drift/test_data_RD_1/ ... test_data_RD_5/
|   |-- local_model_consistent/test_data_LMC_1/ ... test_data_LMC_5/
|   `-- adversarial_learning/test_data_AL_1/ ... test_data_AL_5/
`-- DoS_attacks/
    |-- low_intensity/test_data_D_1/ ... test_data_D_2/
    |-- medium_intensity/test_data_D_3/ ... test_data_D_4/
    `-- high_intensity/test_data_D_5/ ... test_data_D_6/
```

A normal-operation directory contains the primary `log_YYYYMMDD_HH.log`, extracted messages, device-level Excel files, `traffic_data.xlsx`, and the corresponding parsing/alignment scripts. The device files comprise one line-cabinet meter, one load-cabinet meter, five three-phase meters, and nine single-phase meters.

Each FDI directory has the same consistent core layout: one attacked log, `message_received.txt`, `message_received.npy`, `received_message_index.npy`, 16 device-level measurement files, `get_received_message.py`, `get_data.py`, `utlis.py`, and `attack_info.npy`. The `attack_info.npy` file stores the indices of records modified by the attack.

Each DoS directory contains exactly four files:

- `log_YYYYMMDD_HH.log`: attacked raw communication log;
- `attack_time_ranges.csv`: start and end times of attack intervals;
- `cal_traffic.py`: 5 s traffic-aggregation script;
- `traffic_data.xlsx`: scenario-level traffic time series.

## Detection methods

### FDI detection

`my_main.py` implements three unsupervised reconstruction models: Transformer, LSTM, and TCN. Models are trained only on normal measurements. The data loader standardizes all 132 features using statistics fitted on the normal training set, divides each experiment into 576-sample subsequences, and creates overlapping time windows. At test time, the largest feature-wise reconstruction error at each time point is used as its anomaly score.

### DoS detection

`DoS_detection.py` compares:

- a three-sigma Z-score detector;
- Isolation Forest;
- K-nearest-neighbor distance (`KNN`).

Isolation Forest and KNN use `Traffic_volume` as their only feature and fit their reference models on the independent normal traffic in `train_data_1` through `train_data_25`. All three intensity levels are evaluated by default, and precision, recall, and F1-score are reported.

### UMAP validation

`plot_umap.py` creates window-level UMAP visualizations for all six FDI attacks. It standardizes flattened windows, applies deterministic full-SVD PCA to 20 components, and then projects the data to two dimensions with UMAP. `combine_umap_figures.py` arranges the six images into the 2 x 3 panel used for comparison.

## Requirements

Python 3.11 is recommended. The current experiments were verified with Python 3.11.5 and the following package versions:

| Package | Version |
|---|---:|
| PyTorch | 2.11.0 (CPU build) |
| NumPy | 1.26.2 |
| pandas | 2.3.3 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.7.2 |
| openpyxl | 3.1.2 |
| umap-learn | 0.5.11 |
| Pillow | 9.5.0 |

Install the appropriate PyTorch build for your platform, followed by the remaining dependencies. For a CPU-only environment:

```bash
python -m pip install torch numpy==1.26.2 pandas==2.3.3 scikit-learn==1.8.0 matplotlib==3.7.2 openpyxl==3.1.2 umap-learn==0.5.11 Pillow==9.5.0
```

## Quick start

Run commands from the repository root (`detection_methods/`).

### Evaluate every FDI model and scenario

```bash
python my_main.py
```

This loads the supplied checkpoints, evaluates Transformer, LSTM, and TCN on `A`, `S`, `R`, `RD`, `LMC`, and `AL`, and writes `FDI_detection_performances.xlsx`. The workbook contains `Metrics` and `Configuration` sheets.

To evaluate a subset or write to another location:

```bash
python my_main.py --test_datasets A S R --results_path outputs/fdi_baseline.xlsx
```

### Evaluate one FDI model

Always provide the intended test workbook when running a single solver. For example:

```bash
python my_main.py --solver solver_transformer --transformer_mode test --transformer_testdata_path CPDS-AD_dataset/merged_datasets/test_data_A.xlsx

python my_main.py --solver solver_LSTM --LSTM_mode test --LSTM_testdata_path CPDS-AD_dataset/merged_datasets/test_data_LMC.xlsx

python my_main.py --solver solver_TCN --TCN_mode test --TCN_testdata_path CPDS-AD_dataset/merged_datasets/test_data_RD.xlsx
```

### Train an FDI model

```bash
python my_main.py --solver solver_transformer --transformer_mode train
python my_main.py --solver solver_LSTM --LSTM_mode train
python my_main.py --solver solver_TCN --TCN_mode train
```

Training writes `checkpoint.pth` and `log_vali_loss.xlsx` into the corresponding checkpoint directory. It overwrites the supplied checkpoint for that model, so copy the original file or select another directory with `--transformer_checkpoint_path`, `--LSTM_checkpoint_path`, or `--TCN_checkpoint_path` before retraining.

### Evaluate DoS attacks

```bash
python DoS_detection.py
```

The default run evaluates the high-, medium-, and low-intensity workbooks and writes `DoS_detection_performances.xlsx`. Custom input and output paths can be supplied as follows:

```bash
python DoS_detection.py --inputs CPDS-AD_dataset/merged_datasets/test_data_D_low.xlsx --output outputs/dos_low.xlsx
```

### Generate UMAP figures

```bash
python plot_umap.py
python combine_umap_figures.py
```

The first command writes `UMAP_windows_visualization_A.png`, `S.png`, `R.png`, `RD.png`, `LMC.png`, and `AL.png`; the second writes `UMAP_windows_visualization_combined.png`.

Use `python my_main.py --help`, `python DoS_detection.py --help`, or `python combine_umap_figures.py --help` for the complete command-line options.

## Outputs and reproducibility

| Output | Description |
|---|---|
| `FDI_detection_performances.xlsx` | Point-level results for all selected FDI datasets and models, plus the evaluation configuration |
| `DoS_detection_performances.xlsx` | Precision, recall, and F1-score for each DoS intensity and detector |
| `checkpoints/*/checkpoint.pth` | Supplied or newly trained FDI model parameters |
| `UMAP_windows_visualization_*.png` | Individual and combined FDI distribution visualizations |

FDI evaluation uses fixed Python, NumPy, and PyTorch seeds and enables deterministic PyTorch algorithms. The default global and per-model seeds are all `42`. UMAP also uses seed `42` and single-threaded execution. Small numerical or layout differences may still occur across hardware, operating systems, PyTorch builds, or UMAP versions.


## Data and code availability

- Dataset: [CPDS-AD dataset on figshare](https://doi.org/10.6084/m9.figshare.31989933)
- Archived source code: [CPDS-AD-Detection-Methods on Zenodo](https://doi.org/10.5281/zenodo.19562168)
- Development repository: [GitHub](https://github.com/liuyulinlyl/CPDS-AD-Detection-Methods)

## Citation

When using CPDS-AD, cite the dataset and code archive:

```text
Liu, Y., Ruan, Z., & Shi, L. (2026). CPDS-AD dataset. figshare.
https://doi.org/10.6084/m9.figshare.31989933

Liu, Y., Ruan, Z., & Shi, L. (2026). CPDS-AD-Detection-Methods. Zenodo.
https://doi.org/10.5281/zenodo.19562168
```

Please also cite the associated manuscript when bibliographic publication details become available:

```text
Yulin Liu, Zhaojun Ruan, and Libao Shi.
A Cyber-Attack Detection Dataset for Distribution Systems Generated from a
Scaled Cyber-Physical Testbed.
```
