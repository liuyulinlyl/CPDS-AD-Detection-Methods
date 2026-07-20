import os

# Limit thread-based nondeterminism before importing numerical libraries.
for env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(env_var, "1")

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ===============================
# Reproducibility configuration
# ===============================
RANDOM_SEED = 42


def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)


# ===============================
# Paper figure font settings
# ===============================
FONT_FAMILY = "Times New Roman"
plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

font = FontProperties(family=FONT_FAMILY, size=16)

# ===============================
# Parameters
# ===============================
MAIN_DIR = Path(__file__).resolve().parent
DATASET_DIR = MAIN_DIR / "CPDS-AD_dataset"
MERGED_DATASET_DIR = DATASET_DIR / "merged_datasets"
TRAIN_FILE = MERGED_DATASET_DIR / "train_data.xlsx"
ATTACK_DATASETS = ("RD", "A", "AL", "LMC", "R", "S")
RD_NORMAL_SAMPLE_SIZE = 700

# Default UMAP parameters for AL, LMC, R, and S.
DEFAULT_UMAP_PARAMS = {
    "umap_n_components": 2,
    "umap_n_neighbors": 10,
    "umap_min_dist": 1,
    "umap_metric": "euclidean",
    "umap_init": "spectral",
}

# RD-only UMAP parameters. Change these values when tuning RD; the other
# datasets will continue to use their own parameters or DEFAULT_UMAP_PARAMS.
RD_UMAP_PARAMS = {
    "umap_n_components": 2,
    "umap_n_neighbors": 7,
    "umap_min_dist": 1,
    "umap_metric": "euclidean",
    "umap_init": "spectral",
}

# A-only UMAP parameters. Change these values when tuning A; the other
# datasets will continue to use their own parameters or DEFAULT_UMAP_PARAMS.
A_UMAP_PARAMS = {
    "umap_n_components": 2,
    "umap_n_neighbors": 7,
    "umap_min_dist": 1,
    "umap_metric": "euclidean",
    "umap_init": "spectral",
}

UMAP_PARAMS_BY_DATASET = {
    "RD": RD_UMAP_PARAMS,
    "A": A_UMAP_PARAMS,
}

PLOT_CONFIGS = [
    {
        "name": name,
        "window_size": 10,
        "sub_seq_len": 576,
        "pca_components": 20,
        "test_file": MERGED_DATASET_DIR / f"test_data_{name}.xlsx",
        "save_fig": MAIN_DIR / f"UMAP_windows_visualization_{name}.png",
        **UMAP_PARAMS_BY_DATASET.get(name, DEFAULT_UMAP_PARAMS),
    }
    for name in ATTACK_DATASETS
]


def load_normal_training_features():
    if not TRAIN_FILE.is_file():
        raise FileNotFoundError(TRAIN_FILE)
    normal_data = pd.read_excel(TRAIN_FILE)
    feature_columns = [
        column
        for column in normal_data.columns
        if not str(column).startswith("Unnamed:")
        and str(column).lower() != "labels"
    ]
    if not feature_columns:
        raise ValueError(f"No measurement features found in {TRAIN_FILE}")
    print(
        f"Loaded {len(normal_data)} normal samples with "
        f"{len(feature_columns)} measurement features from {TRAIN_FILE}"
    )
    return normal_data[feature_columns].to_numpy(dtype=np.float64), feature_columns


def extract_anomaly_windows(features, labels, window_size, subseq_len):
    windows = []
    num_subseq = len(features) // subseq_len

    for subseq_idx in range(num_subseq):
        start = subseq_idx * subseq_len
        end = start + subseq_len

        feature_subseq = features[start:end]
        label_subseq = labels[start:end]

        for window_idx in range(subseq_len - window_size + 1):
            data_window = feature_subseq[window_idx:window_idx + window_size]
            label_window = label_subseq[window_idx:window_idx + window_size]

            if np.any(label_window == 1):
                windows.append(data_window.flatten())

    return np.asarray(windows, dtype=np.float64)


def extract_normal_windows(features, window_size, subseq_len):
    windows = []
    num_subseq = len(features) // subseq_len

    for subseq_idx in range(num_subseq):
        start = subseq_idx * subseq_len
        end = start + subseq_len

        feature_subseq = features[start:end]

        for window_idx in range(subseq_len - window_size + 1):
            data_window = feature_subseq[window_idx:window_idx + window_size]
            windows.append(data_window.flatten())

    return np.asarray(windows, dtype=np.float64)


def canonicalize_embedding(embedding, labels):
    """Reduce axis-swap and mirror differences across platforms."""
    embedding = np.asarray(embedding, dtype=np.float64).copy()
    embedding -= embedding.mean(axis=0, keepdims=True)

    axis_order = np.argsort(-embedding.var(axis=0))
    embedding = embedding[:, axis_order]

    normal_mask = labels == 0
    anomaly_mask = labels == 1

    if np.any(normal_mask) and np.any(anomaly_mask):
        class_delta = embedding[anomaly_mask].mean(axis=0) - embedding[normal_mask].mean(axis=0)
        for axis in range(embedding.shape[1]):
            if class_delta[axis] < 0:
                embedding[:, axis] *= -1

    return embedding


def generate_plot(config, x_train, feature_columns):
    set_reproducibility(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # ===============================
    # Read test data with labels
    # ===============================
    test_data = pd.read_excel(config["test_file"])
    label_columns = [
        column for column in test_data.columns if str(column).lower() == "labels"
    ]
    if len(label_columns) != 1:
        raise ValueError(
            f'{config["test_file"]} must contain exactly one labels column'
        )
    label_column = label_columns[0]
    test_feature_columns = [
        column
        for column in test_data.columns
        if not str(column).startswith("Unnamed:") and column != label_column
    ]
    if test_feature_columns != feature_columns:
        raise ValueError(
            f'{config["test_file"]} measurement features do not match {TRAIN_FILE}'
        )
    x_test = test_data[feature_columns].to_numpy(dtype=np.float64)
    y_test = test_data[label_column].to_numpy(dtype=np.int64)

    anomaly_windows = extract_anomaly_windows(
        features=x_test,
        labels=y_test,
        window_size=config["window_size"],
        subseq_len=config["sub_seq_len"],
    )
    print(f'[{config["name"]}] Number of anomaly windows:', anomaly_windows.shape[0])

    normal_windows = extract_normal_windows(
        features=x_train,
        window_size=config["window_size"],
        subseq_len=config["sub_seq_len"],
    )
    print(f'[{config["name"]}] Number of normal windows:', normal_windows.shape[0])

    # ===============================
    # Sample windows. Keep the original 1:1 balance for all datasets except
    # RD, which uses a fixed number of normal windows.
    # ===============================
    sample_size = min(len(normal_windows), len(anomaly_windows))
    normal_sample_size = (
        min(RD_NORMAL_SAMPLE_SIZE, len(normal_windows))
        if config["name"] == "RD"
        else sample_size
    )
    anomaly_sample_size = sample_size

    normal_idx = rng.choice(len(normal_windows), normal_sample_size, replace=False)
    anomaly_idx = rng.choice(
        len(anomaly_windows), anomaly_sample_size, replace=False
    )

    normal_windows = normal_windows[normal_idx]
    anomaly_windows = anomaly_windows[anomaly_idx]

    windows = np.vstack([normal_windows, anomaly_windows])
    labels = np.array(
        [0] * normal_sample_size + [1] * anomaly_sample_size,
        dtype=np.int64,
    )

    print(f'[{config["name"]}] Final sample shape:', windows.shape)

    # ===============================
    # Standardization
    # ===============================
    scaler = StandardScaler()
    windows_scaled = scaler.fit_transform(windows)

    # ===============================
    # PCA
    # Use a deterministic full SVD solver.
    # ===============================
    pca_components = min(
        config["pca_components"],
        windows_scaled.shape[0],
        windows_scaled.shape[1],
    )
    print(f'[{config["name"]}] PCA components:', pca_components)
    pca = PCA(n_components=pca_components, svd_solver="full")
    windows_pca = pca.fit_transform(windows_scaled)

    # ===============================
    # UMAP
    # Use single-threaded execution and random init with a fixed seed.
    # ===============================
    umap_model = umap.UMAP(
        n_components=config["umap_n_components"],
        n_neighbors=config["umap_n_neighbors"],
        min_dist=config["umap_min_dist"],
        metric=config["umap_metric"],
        init=config["umap_init"],
        random_state=RANDOM_SEED,
        transform_seed=RANDOM_SEED,
        n_jobs=1,
        low_memory=True,
    )
    x_umap = umap_model.fit_transform(windows_pca)
    x_umap = canonicalize_embedding(x_umap, labels)

    # ===============================
    # Plotting
    # ===============================
    fig, ax = plt.subplots(figsize=(7, 6))

    normal_mask = labels == 0
    anomaly_mask = labels == 1

    ax.scatter(
        x_umap[normal_mask, 0],
        x_umap[normal_mask, 1],
        alpha=0.5,
        s=10,
        label="Normal",
    )
    ax.scatter(
        x_umap[anomaly_mask, 0],
        x_umap[anomaly_mask, 1],
        alpha=0.7,
        s=10,
        label="FDI",
    )

    ax.set_xlabel("UMAP Dimension 1", fontproperties=font)
    ax.set_ylabel("UMAP Dimension 2", fontproperties=font)

    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontname(FONT_FAMILY)
        tick_label.set_fontsize(14)

    ax.legend(loc="lower right", prop=font)
    ax.tick_params(direction="in")

    plt.savefig(
        config["save_fig"],
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main():
    x_train, feature_columns = load_normal_training_features()
    for config in PLOT_CONFIGS:
        generate_plot(config, x_train, feature_columns)


if __name__ == "__main__":
    main()
