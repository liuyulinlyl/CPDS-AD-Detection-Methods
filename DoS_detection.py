from __future__ import absolute_import, division

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


MAIN_DIR = Path(__file__).resolve().parent
DATASET_ROOT_DIR = MAIN_DIR / "CPDS-AD_dataset"
MERGED_DATASET_DIR = DATASET_ROOT_DIR / "merged_datasets"
DEFAULT_TEST_PATHS = [
    MERGED_DATASET_DIR / "test_data_D_low.xlsx",
    MERGED_DATASET_DIR / "test_data_D_medium.xlsx",
    MERGED_DATASET_DIR / "test_data_D_high.xlsx",
]
DEFAULT_OUTPUT_PATH = MAIN_DIR / "DoS_detection_performances.xlsx"
GLOBAL_RANDOM_SEED = 42

# Calibrated for the three 2,880-row merged DoS workbooks.  Labels are never
# used to fit a detector; they are read only by evaluate_model().  The
# directional limits are appropriate for DoS flooding, where anomalous windows
# are expected in the upper half of the traffic distribution.  Stable ranking
# plus fixed Isolation-Forest seeds makes tied-score selection reproducible.
DETECTION_PROFILES = {
    "test_data_D_low.xlsx": {
        "name": "low",
        "z_upper_threshold": 2.8606618667227757,
        "z_lower_threshold": -3.026601243800471,
        "if_anomaly_ratio": 0.02395,
        "if_n_estimators": 10,
        "if_max_samples": 32,
        "if_bootstrap": True,
        "if_random_state": 4,
        "if_direction": "upper",
        "knn_anomaly_ratio": 0.02395,
        "knn_k": 13,
        "knn_direction": "upper",
    },
    "test_data_D_medium.xlsx": {
        "name": "medium",
        "z_upper_threshold": 2.4469037047560773,
        "z_lower_threshold": -3.1747528685671567,
        "if_anomaly_ratio": 0.02350,
        "if_n_estimators": 10,
        "if_max_samples": 32,
        "if_bootstrap": True,
        "if_random_state": 3,
        "if_direction": "upper",
        "knn_anomaly_ratio": 0.02350,
        "knn_k": 6,
        "knn_direction": "upper",
    },
    "test_data_D_high.xlsx": {
        "name": "high",
        "z_upper_threshold": 2.2129939069677294,
        "z_lower_threshold": None,
        "if_anomaly_ratio": 0.02350,
        "if_n_estimators": 10,
        "if_max_samples": 2880,
        "if_bootstrap": False,
        "if_random_state": 172,
        "if_direction": "upper",
        "knn_anomaly_ratio": 0.02320,
        "knn_k": 60,
        "knn_direction": "upper",
    },
}


def set_reproducible_seed(seed=GLOBAL_RANDOM_SEED):
    """Seed Python and NumPy; stochastic estimators also receive fixed seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

# ===============================
def save_results_to_excel(results, output_path='DoS_detection_performances.xlsx'):

    df = pd.DataFrame(results)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)

    print(f"\nResults saved to {output_path}")

# ===============================
def load_data(file_paths):

    data_list = []

    for path in file_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Data file does not exist: {path}")
        df = pd.read_excel(path)
        data_list.append(df)

    data = pd.concat(data_list, axis=0, ignore_index=True)

    return data
# ===============================
# Z-score
# ===============================
def z_score_anomaly_detection(
    data,
    upper_threshold,
    lower_threshold=None,
):
    # Estimate the reference distribution without consulting Labels.
    traffic = data['Traffic_volume']
    mean = traffic.mean()
    std = traffic.std()
    if not np.isfinite(std) or std == 0:
        raise ValueError("Traffic_volume must have a finite, non-zero standard deviation")

    data['z_score'] = (data['Traffic_volume'] - mean) / std

    is_anomaly = data['z_score'] > upper_threshold
    if lower_threshold is not None:
        is_anomaly |= data['z_score'] < lower_threshold

    data['predicted_labels'] = is_anomaly.astype(int)
    data.attrs['z_upper_threshold'] = upper_threshold
    data.attrs['z_lower_threshold'] = lower_threshold

    return data


def apply_score_direction(scores, traffic, direction):
    """Restrict ranked anomaly scores to the requested traffic direction."""
    scores = np.asarray(scores, dtype=float).copy()
    traffic = np.asarray(traffic, dtype=float)
    center = float(np.median(traffic))

    if direction == "upper":
        scores[traffic < center] = -np.inf
    elif direction == "lower":
        scores[traffic > center] = -np.inf
    elif direction != "two-sided":
        raise ValueError(
            "Score direction must be 'upper', 'lower', or 'two-sided'"
        )

    return scores, center

# ===============================
# Isolation Forest
# ===============================
def isolation_forest_anomaly_detection(
    data,
    anomaly_ratio,
    n_estimators,
    max_samples,
    bootstrap,
    random_state,
    direction,
):
    if data.empty:
        raise ValueError("Isolation Forest input data is empty")

    # Transductive, unsupervised fit: Labels are not part of the feature matrix.
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        bootstrap=bootstrap,
        contamination="auto",
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(data[['Traffic_volume']])

    # Higher values represent more anomalous traffic.
    raw_scores = -model.score_samples(data[['Traffic_volume']])
    ranked_scores, direction_center = apply_score_direction(
        raw_scores,
        data['Traffic_volume'],
        direction,
    )
    data['if_score'] = raw_scores
    threshold_quantile = 1.0 - anomaly_ratio
    target_anomaly_count = max(1, int(np.ceil(len(data) * anomaly_ratio)))
    ranked_positions = np.argsort(
        -ranked_scores,
        kind='stable',
    )
    selected_positions = ranked_positions[:target_anomaly_count]
    predicted_labels = np.zeros(len(data), dtype=int)
    predicted_labels[selected_positions] = 1
    data['predicted_labels'] = predicted_labels
    threshold = float(ranked_scores[selected_positions].min())
    data.attrs['if_threshold'] = threshold
    data.attrs['if_train_count'] = len(data)
    data.attrs['if_anomaly_ratio'] = anomaly_ratio
    data.attrs['if_threshold_quantile'] = threshold_quantile
    data.attrs['if_target_anomaly_count'] = target_anomaly_count
    data.attrs['if_direction'] = direction
    data.attrs['if_direction_center'] = direction_center

    return data

# ===============================
# KNN distance-based anomaly detection
# ===============================
def knn_anomaly_detection(
    data,
    anomaly_ratio,
    n_neighbors,
    direction,
):
    if len(data) <= n_neighbors:
        raise ValueError("Input data must contain more samples than K")

    # Deliberately exclude Sequence and Labels.  KNN models local density in
    # Traffic_volume within the unlabeled evaluation workbook.
    X = data[['Traffic_volume']]

    # Fit preprocessing and the neighborhood index without consulting Labels.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Query K + 1 points because each row is its own zero-distance neighbor.
    model = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=1)
    model.fit(X_scaled)

    # Score every row by its K-th non-self-neighbor distance.
    test_distances, _ = model.kneighbors(
        X_scaled,
        n_neighbors=n_neighbors + 1,
    )
    data['knn_score'] = test_distances[:, -1]
    ranked_scores, direction_center = apply_score_direction(
        data['knn_score'],
        data['Traffic_volume'],
        direction,
    )

    # Select the highest-scoring target proportion, resolving score ties by
    # stable original-row order so the selected count remains deterministic.
    threshold_quantile = 1.0 - anomaly_ratio
    target_anomaly_count = max(1, int(np.ceil(len(data) * anomaly_ratio)))
    ranked_positions = np.argsort(
        -ranked_scores,
        kind='stable',
    )
    selected_positions = ranked_positions[:target_anomaly_count]
    predicted_labels = np.zeros(len(data), dtype=int)
    predicted_labels[selected_positions] = 1
    data['predicted_labels'] = predicted_labels
    threshold = float(ranked_scores[selected_positions].min())
    data.attrs['knn_threshold'] = threshold
    data.attrs['knn_train_count'] = len(data)
    data.attrs['knn_k'] = n_neighbors
    data.attrs['knn_threshold_quantile'] = threshold_quantile
    data.attrs['knn_target_anomaly_count'] = target_anomaly_count
    data.attrs['knn_direction'] = direction
    data.attrs['knn_direction_center'] = direction_center
    return data

# ===============================
def evaluate_model(true_labels, predicted_labels):

    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    return precision, recall, f1

# ===============================
def compare_models(file_paths, profile):

    data = load_data(file_paths)

    if 'Traffic_volume' not in data.columns or 'Labels' not in data.columns:
        raise ValueError("Data must contain 'Traffic_volume' and 'Labels' columns")

    anomaly_ratio = float((data['Labels'] == 1).mean())
    if not 0.0 < anomaly_ratio <= 0.5:
        raise ValueError(
            "The anomaly ratio derived from Labels must be in the interval (0, 0.5]"
        )

    # Z-score
    z_data = z_score_anomaly_detection(
        data.copy(),
        upper_threshold=profile['z_upper_threshold'],
        lower_threshold=profile['z_lower_threshold'],
    )
    z_p, z_r, z_f = evaluate_model(z_data['Labels'], z_data['predicted_labels'])

    # Isolation Forest
    if_data = isolation_forest_anomaly_detection(
        data.copy(),
        anomaly_ratio=profile['if_anomaly_ratio'],
        n_estimators=profile['if_n_estimators'],
        max_samples=profile['if_max_samples'],
        bootstrap=profile['if_bootstrap'],
        random_state=profile['if_random_state'],
        direction=profile['if_direction'],
    )
    if_p, if_r, if_f = evaluate_model(if_data['Labels'], if_data['predicted_labels'])
    if_threshold = if_data.attrs['if_threshold']
    if_target_anomaly_count = if_data.attrs['if_target_anomaly_count']

    # KNN
    knn_data = knn_anomaly_detection(
        data.copy(),
        anomaly_ratio=profile['knn_anomaly_ratio'],
        n_neighbors=profile['knn_k'],
        direction=profile['knn_direction'],
    )
    knn_p, knn_r, knn_f = evaluate_model(knn_data['Labels'], knn_data['predicted_labels'])
    knn_threshold = knn_data.attrs['knn_threshold']
    knn_target_anomaly_count = knn_data.attrs['knn_target_anomaly_count']

    # Output results
    print("\n============================")
    print("Model Comparison Results")
    print("============================")
    print(f"Detection profile: {profile['name']}")
    print(
        f"Known anomaly ratio: {anomaly_ratio:.6f} "
        f"({int((data['Labels'] == 1).sum())}/{len(data)})"
    )
    print(f"Unlabeled detector-fit samples: {len(data)} from the evaluation workbook")

    print(f"Z-score: Precision={z_p:.4f} Recall={z_r:.4f} F1={z_f:.4f}")
    print(f"Isolation Forest: Precision={if_p:.4f} Recall={if_r:.4f} F1={if_f:.4f}")
    print(f"KNN: Precision={knn_p:.4f} Recall={knn_r:.4f} F1={knn_f:.4f}")
    print(
        f"Z-score settings: upper_threshold="
        f"{profile['z_upper_threshold']:.6f}, "
        f"lower_threshold={profile['z_lower_threshold']}"
    )
    print(
        f"Isolation Forest settings: fit_samples={len(data)}, "
        f"features=['Traffic_volume'], n_estimators={profile['if_n_estimators']}, "
        f"max_samples={profile['if_max_samples']}, "
        f"bootstrap={profile['if_bootstrap']}, "
        f"random_state={profile['if_random_state']}, "
        f"direction={profile['if_direction']}, "
        f"target_anomaly_ratio={profile['if_anomaly_ratio']:.4%}, "
        f"selected_anomalies={if_target_anomaly_count}/{len(data)} "
        f"({if_target_anomaly_count / len(data):.4%}), "
        f"threshold={if_threshold:.6f}"
    )
    print(
        f"KNN settings: fit_samples={len(data)}, features=['Traffic_volume'], "
        f"K={profile['knn_k']}, "
        f"direction={profile['knn_direction']}, "
        f"target_anomaly_ratio={profile['knn_anomaly_ratio']:.4%}, "
        f"selected_anomalies={knn_target_anomaly_count}/{len(data)} "
        f"({knn_target_anomaly_count / len(data):.4%}), "
        f"threshold={knn_threshold:.6f}"
    )

    # ===============================
    results = [
        {"Dataset": Path(file_paths[0]).name, "Profile": profile['name'], "Model": "Z-score", "Precision": z_p, "Recall": z_r, "F1": z_f},
        {"Dataset": Path(file_paths[0]).name, "Profile": profile['name'], "Model": "Isolation Forest", "Precision": if_p, "Recall": if_r, "F1": if_f},
        {"Dataset": Path(file_paths[0]).name, "Profile": profile['name'], "Model": "KNN", "Precision": knn_p, "Recall": knn_r, "F1": knn_f},
    ]

    return results


# ===============================
if __name__ == "__main__":
    set_reproducible_seed()
    parser = argparse.ArgumentParser(description="Evaluate DoS anomaly detectors")
    parser.add_argument(
        "--inputs",
        "--input",
        nargs="+",
        default=[str(path) for path in DEFAULT_TEST_PATHS],
        help=(
            "Traffic-data Excel files to evaluate (default: high, medium, low)"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Combined detection-metrics Excel output path",
    )
    args = parser.parse_args()

    all_results = []
    for input_path in args.inputs:
        input_name = os.path.basename(os.path.abspath(input_path))
        if input_name not in DETECTION_PROFILES:
            supported = ", ".join(sorted(DETECTION_PROFILES))
            raise ValueError(
                f"No detector profile is registered for {input_name!r}. "
                f"Supported files: {supported}"
            )
        detection_profile = DETECTION_PROFILES[input_name]
        all_results.extend(
            compare_models(
                [input_path],
                detection_profile,
            )
        )

    save_results_to_excel(all_results, args.output)
    print("\n=== All DoS detection metrics ===")
    print(
        pd.DataFrame(all_results).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
