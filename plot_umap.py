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
DATASET_DIR = os.path.join(os.path.dirname(__file__), "CPDS-AD_dataset")
TRAIN_FILE = os.path.join(DATASET_DIR, "train_data.xlsx")

PLOT_CONFIGS = [
    {
        "name": "S",
        "window_size": 10,
        "sub_seq_len": 576,
        "pca_components": 20,
        "test_file": os.path.join(DATASET_DIR, "test_data_S.xlsx"),
        "save_fig": "UMAP_windows_visualization_S.png",
        "umap_n_components": 2,
        "umap_n_neighbors": 10,
        "umap_min_dist": 1,
        "umap_init": "spectral",
    },
    {
        "name": "R",
        "window_size": 10,
        "sub_seq_len": 576,
        "pca_components": 20,
        "test_file": os.path.join(DATASET_DIR, "test_data_R.xlsx"),
        "save_fig": "UMAP_windows_visualization_R.png",
        "umap_n_components": 2,
        "umap_n_neighbors": 10,
        "umap_min_dist": 1,
        "umap_init": "spectral",
    },
    {
        "name": "A",
        "window_size": 10,
        "sub_seq_len": 576,
        "pca_components": 20,
        "test_file": os.path.join(DATASET_DIR, "test_data_A.xlsx"),
        "save_fig": "UMAP_windows_visualization_A.png",
        "umap_n_components": 2,
        "umap_n_neighbors": 10,
        "umap_min_dist": 1,
        "umap_init": "spectral",
    },
]


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


def generate_plot(config):
    set_reproducibility(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # ===============================
    # Read test data with labels
    # ===============================
    test_data = pd.read_excel(config["test_file"])
    x_test = test_data.iloc[:, :-1].to_numpy(dtype=np.float64)
    y_test = test_data.iloc[:, -1].to_numpy(dtype=np.int64)

    anomaly_windows = extract_anomaly_windows(
        features=x_test,
        labels=y_test,
        window_size=config["window_size"],
        subseq_len=config["sub_seq_len"],
    )
    print(f'[{config["name"]}] Number of anomaly windows:', anomaly_windows.shape[0])

    # ===============================
    # Read train data
    # ===============================
    train_data = pd.read_excel(TRAIN_FILE)
    x_train = train_data.to_numpy(dtype=np.float64)

    normal_windows = extract_normal_windows(
        features=x_train,
        window_size=config["window_size"],
        subseq_len=config["sub_seq_len"],
    )
    print(f'[{config["name"]}] Number of normal windows:', normal_windows.shape[0])

    # ===============================
    # Balance samples
    # ===============================
    sample_size = min(len(normal_windows), len(anomaly_windows))

    normal_idx = rng.choice(len(normal_windows), sample_size, replace=False)
    anomaly_idx = rng.choice(len(anomaly_windows), sample_size, replace=False)

    normal_windows = normal_windows[normal_idx]
    anomaly_windows = anomaly_windows[anomaly_idx]

    windows = np.vstack([normal_windows, anomaly_windows])
    labels = np.array([0] * sample_size + [1] * sample_size, dtype=np.int64)

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
    pca = PCA(n_components=config["pca_components"], svd_solver="full")
    windows_pca = pca.fit_transform(windows_scaled)

    # ===============================
    # UMAP
    # Use single-threaded execution and random init with a fixed seed.
    # ===============================
    umap_model = umap.UMAP(
        n_components=config["umap_n_components"],
        n_neighbors=config["umap_n_neighbors"],
        min_dist=config["umap_min_dist"],
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
    for config in PLOT_CONFIGS:
        generate_plot(config)


if __name__ == "__main__":
    main()
