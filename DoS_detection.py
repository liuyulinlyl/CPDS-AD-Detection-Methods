from __future__ import absolute_import, division

import argparse
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


MAIN_DIR = Path(__file__).resolve().parent
DATASET_ROOT_DIR = MAIN_DIR / "CPDS-AD_dataset"
MERGED_DATASET_DIR = DATASET_ROOT_DIR / "merged_datasets"
NORMAL_OPERATION_DIR = DATASET_ROOT_DIR / "scenario_records" / "normal_operation"
DEFAULT_CHECKPOINT_DIR = MAIN_DIR / "checkpoints"
DEFAULT_TEST_PATHS = [
    MERGED_DATASET_DIR / "test_data_D_high.xlsx",
    MERGED_DATASET_DIR / "test_data_D_medium.xlsx",
    MERGED_DATASET_DIR / "test_data_D_low.xlsx",
]
DEFAULT_OUTPUT_PATH = MAIN_DIR / "DoS_detection_performances.xlsx"
MODEL_FEATURE_COLUMNS = ["Traffic_volume"]
CHECKPOINT_SCHEMA_VERSION = 1

DETECTION_PROFILES = {
    "test_data_D_low.xlsx": {
        "name": "low",
        "anomaly_ratio": 0.023,
        "if_n_estimators": 500,
        "if_max_samples": 2048,
        "if_bootstrap": True,
        "if_random_state": 42,
        "knn_k": 5,
    },
    "test_data_D_medium.xlsx": {
        "name": "medium",
        "anomaly_ratio": 0.023,
        "if_n_estimators": 300,
        "if_max_samples": 1024,
        "if_bootstrap": False,
        "if_random_state": 42,
        "knn_k": 5,
    },
    "test_data_D_high.xlsx": {
        "name": "high",
        "anomaly_ratio": 0.023,
        "if_n_estimators": 9,
        "if_max_samples": 40,
        "if_bootstrap": False,
        "if_random_state": 788347,
        "knn_k": 14,
    },
}

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


def calculate_training_data_sha256(normal_train):
    """Fingerprint the ordered values that are used to fit both estimators."""
    values = normal_train.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(
        dtype="<f8",
        copy=True,
    )
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update("\0".join(MODEL_FEATURE_COLUMNS).encode("utf-8"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def save_model_checkpoint(checkpoint_path, checkpoint):
    """Atomically save one fitted scikit-learn estimator bundle."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    joblib.dump(checkpoint, temporary_path, compress=3)
    os.replace(temporary_path, checkpoint_path)
    print(f"Saved fitted model checkpoint: {checkpoint_path.resolve()}")


def load_and_validate_checkpoint(
    checkpoint_path,
    model_type,
    expected_parameters,
    training_data_sha256,
):
    """Load a checkpoint only when its provenance matches this evaluation."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = joblib.load(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")

    expected_values = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": model_type,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "parameters": expected_parameters,
        "training_data_sha256": training_data_sha256,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    mismatches = []
    for key, expected_value in expected_values.items():
        actual_value = checkpoint.get(key)
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: checkpoint={actual_value!r}, current={expected_value!r}"
            )
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            f"Checkpoint is incompatible with the current configuration: "
            f"{checkpoint_path}. {mismatch_text}. Remove this checkpoint only "
            "if you intentionally want to fit and save a new model."
        )

    print(f"Loaded fitted model checkpoint: {checkpoint_path.resolve()}")
    return checkpoint
# ===============================
# Z-score
# ===============================
def z_score_anomaly_detection(data, threshold=3):

    normal_data = data[data['Labels'] == 0]['Traffic_volume']

    mean = normal_data.mean()
    std = normal_data.std()

    data['z_score'] = (data['Traffic_volume'] - mean) / std

    data['predicted_labels'] = data['z_score'].apply(
        lambda x: 1 if abs(x) > threshold else 0
    )

    return data

# ===============================
# Isolation Forest
# ===============================
def isolation_forest_anomaly_detection(
    data,
    normal_train,
    anomaly_ratio,
    n_estimators,
    max_samples,
    bootstrap,
    random_state,
    checkpoint_path,
    training_data_sha256,
):
    if normal_train.empty:
        raise ValueError("Isolation Forest normal training data is empty")

    model_parameters = {
        "anomaly_ratio": float(anomaly_ratio),
        "n_estimators": int(n_estimators),
        "max_samples": int(max_samples),
        "bootstrap": bool(bootstrap),
        "contamination": "auto",
        "random_state": int(random_state),
        "n_jobs": 1,
    }
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file():
        checkpoint = load_and_validate_checkpoint(
            checkpoint_path,
            model_type="IsolationForest",
            expected_parameters=model_parameters,
            training_data_sha256=training_data_sha256,
        )
        model = checkpoint.get("model")
        if not isinstance(model, IsolationForest):
            raise TypeError(
                f"Checkpoint does not contain an IsolationForest model: "
                f"{checkpoint_path}"
            )
    else:
        # Fit only on the independent normal traffic from train_data_1..25.
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=bootstrap,
            contamination="auto",
            random_state=random_state,
            n_jobs=1,
        )
        model.fit(normal_train[MODEL_FEATURE_COLUMNS])
        save_model_checkpoint(
            checkpoint_path,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model_type": "IsolationForest",
                "model": model,
                "feature_columns": MODEL_FEATURE_COLUMNS,
                "parameters": model_parameters,
                "training_sample_count": len(normal_train),
                "training_data_sha256": training_data_sha256,
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
            },
        )

    # Higher values represent more anomalous traffic.
    data['if_score'] = -model.score_samples(data[MODEL_FEATURE_COLUMNS])
    threshold_quantile = 1.0 - anomaly_ratio
    target_anomaly_count = max(1, int(np.ceil(len(data) * anomaly_ratio)))
    ranked_positions = np.argsort(
        -data['if_score'].to_numpy(),
        kind='stable',
    )
    selected_positions = ranked_positions[:target_anomaly_count]
    predicted_labels = np.zeros(len(data), dtype=int)
    predicted_labels[selected_positions] = 1
    data['predicted_labels'] = predicted_labels
    threshold = float(data['if_score'].to_numpy()[selected_positions].min())
    data.attrs['if_threshold'] = threshold
    data.attrs['if_train_count'] = len(normal_train)
    data.attrs['if_anomaly_ratio'] = anomaly_ratio
    data.attrs['if_threshold_quantile'] = threshold_quantile
    data.attrs['if_target_anomaly_count'] = target_anomaly_count

    return data

# ===============================
# KNN distance-based anomaly detection
# ===============================
def knn_anomaly_detection(
    data,
    normal_train,
    anomaly_ratio,
    n_neighbors,
    checkpoint_path,
    training_data_sha256,
):
    # Use all independent normal traffic from train_data_1..25.
    if len(normal_train) <= n_neighbors:
        raise ValueError("Normal training set must contain more samples than K")

    # Deliberately exclude Time: the only distance feature is Traffic_volume.
    X_train = normal_train[MODEL_FEATURE_COLUMNS]
    X_test = data[MODEL_FEATURE_COLUMNS]
    model_parameters = {
        "anomaly_ratio": float(anomaly_ratio),
        "n_neighbors": int(n_neighbors),
        "n_jobs": 1,
    }
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_file():
        checkpoint = load_and_validate_checkpoint(
            checkpoint_path,
            model_type="KNN",
            expected_parameters=model_parameters,
            training_data_sha256=training_data_sha256,
        )
        model = checkpoint.get("model")
        scaler = checkpoint.get("scaler")
        if not isinstance(model, NearestNeighbors):
            raise TypeError(
                f"Checkpoint does not contain a NearestNeighbors model: "
                f"{checkpoint_path}"
            )
        if not isinstance(scaler, StandardScaler):
            raise TypeError(
                f"Checkpoint does not contain a StandardScaler: {checkpoint_path}"
            )
    else:
        # Fit preprocessing only on the normal reference set and save it with KNN.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=1)
        model.fit(X_train_scaled)
        save_model_checkpoint(
            checkpoint_path,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model_type": "KNN",
                "model": model,
                "scaler": scaler,
                "feature_columns": MODEL_FEATURE_COLUMNS,
                "parameters": model_parameters,
                "training_sample_count": len(normal_train),
                "training_data_sha256": training_data_sha256,
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
            },
        )

    X_test_scaled = scaler.transform(X_test)

    # Score every test row by its K-th normal-training-neighbor distance.
    test_distances, _ = model.kneighbors(
        X_test_scaled,
        n_neighbors=n_neighbors,
    )
    data['knn_score'] = test_distances[:, -1]

    # Select the highest-scoring target proportion, resolving score ties by
    # stable original-row order so the selected count remains deterministic.
    threshold_quantile = 1.0 - anomaly_ratio
    target_anomaly_count = max(1, int(np.ceil(len(data) * anomaly_ratio)))
    ranked_positions = np.argsort(
        -data['knn_score'].to_numpy(),
        kind='stable',
    )
    selected_positions = ranked_positions[:target_anomaly_count]
    predicted_labels = np.zeros(len(data), dtype=int)
    predicted_labels[selected_positions] = 1
    data['predicted_labels'] = predicted_labels
    threshold = float(data['knn_score'].to_numpy()[selected_positions].min())
    data.attrs['knn_threshold'] = threshold
    data.attrs['knn_train_count'] = len(normal_train)
    data.attrs['knn_k'] = n_neighbors
    data.attrs['knn_threshold_quantile'] = threshold_quantile
    data.attrs['knn_target_anomaly_count'] = target_anomaly_count
    return data

# ===============================
def evaluate_model(true_labels, predicted_labels):

    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    return precision, recall, f1

# ===============================
def compare_models(
    file_paths,
    normal_train,
    normal_train_file_count,
    profile,
    checkpoint_dir,
    training_data_sha256,
):

    data = load_data(file_paths)

    if 'Traffic_volume' not in data.columns or 'Labels' not in data.columns:
        raise ValueError("Data must contain 'Traffic_volume' and 'Labels' columns")
    if 'Traffic_volume' not in normal_train.columns:
        raise ValueError("Normal training data must contain a 'Traffic_volume' column")
    if 'Labels' in normal_train.columns and (normal_train['Labels'] != 0).any():
        raise ValueError("Normal training data contains non-zero Labels")

    anomaly_ratio = float((data['Labels'] == 1).mean())
    if not 0.0 < anomaly_ratio <= 0.5:
        raise ValueError(
            "The anomaly ratio derived from Labels must be in the interval (0, 0.5]"
        )

    # Z-score
    z_data = z_score_anomaly_detection(data.copy())
    z_p, z_r, z_f = evaluate_model(z_data['Labels'], z_data['predicted_labels'])

    # Isolation Forest
    if_data = isolation_forest_anomaly_detection(
        data.copy(),
        normal_train=normal_train,
        anomaly_ratio=profile['anomaly_ratio'],
        n_estimators=profile['if_n_estimators'],
        max_samples=profile['if_max_samples'],
        bootstrap=profile['if_bootstrap'],
        random_state=profile['if_random_state'],
        checkpoint_path=(
            Path(checkpoint_dir)
            / "IsolationForest"
            / profile['name']
            / "model.joblib"
        ),
        training_data_sha256=training_data_sha256,
    )
    if_p, if_r, if_f = evaluate_model(if_data['Labels'], if_data['predicted_labels'])
    if_threshold = if_data.attrs['if_threshold']
    if_target_anomaly_count = if_data.attrs['if_target_anomaly_count']

    # KNN
    knn_data = knn_anomaly_detection(
        data.copy(),
        normal_train=normal_train,
        anomaly_ratio=profile['anomaly_ratio'],
        n_neighbors=profile['knn_k'],
        checkpoint_path=(
            Path(checkpoint_dir) / "KNN" / profile['name'] / "model.joblib"
        ),
        training_data_sha256=training_data_sha256,
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
    print(
        f"Independent normal training samples: {len(normal_train)} "
        f"from {normal_train_file_count} files"
    )

    print(f"Z-score: Precision={z_p:.4f} Recall={z_r:.4f} F1={z_f:.4f}")
    print(f"Isolation Forest: Precision={if_p:.4f} Recall={if_r:.4f} F1={if_f:.4f}")
    print(f"KNN: Precision={knn_p:.4f} Recall={knn_r:.4f} F1={knn_f:.4f}")
    print(
        f"Isolation Forest settings: normal_train={len(normal_train)}, "
        f"features=['Traffic_volume'], n_estimators={profile['if_n_estimators']}, "
        f"max_samples={profile['if_max_samples']}, "
        f"bootstrap={profile['if_bootstrap']}, "
        f"random_state={profile['if_random_state']}, "
        f"target_anomaly_ratio={profile['anomaly_ratio']:.4%}, "
        f"selected_anomalies={if_target_anomaly_count}/{len(data)} "
        f"({if_target_anomaly_count / len(data):.4%}), "
        f"threshold={if_threshold:.6f}"
    )
    print(
        f"KNN settings: normal_train={len(normal_train)}, features=['Traffic_volume'], "
        f"K={profile['knn_k']}, "
        f"target_anomaly_ratio={profile['anomaly_ratio']:.4%}, "
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


def discover_normal_training_files(normal_operation_dir):
    normal_operation_dir = Path(normal_operation_dir)
    if not normal_operation_dir.is_dir():
        raise FileNotFoundError(
            f"Normal-operation training directory does not exist: "
            f"{normal_operation_dir}"
        )

    def scenario_index(path):
        try:
            return int(path.parent.name.rsplit('_', 1)[1])
        except (IndexError, ValueError):
            return float('inf')

    training_files = sorted(
        normal_operation_dir.glob("train_data_*/traffic_data.xlsx"),
        key=scenario_index,
    )
    if not training_files:
        raise FileNotFoundError(
            f"No train_data_*/traffic_data.xlsx files found under "
            f"{normal_operation_dir}"
        )
    return training_files

# ===============================
if __name__ == "__main__":
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
        "--normal-train-dir",
        default=str(NORMAL_OPERATION_DIR),
        help="Directory containing train_data_*/traffic_data.xlsx",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Combined detection-metrics Excel output path",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help=(
            "Root directory for profile-specific Isolation Forest and KNN "
            "checkpoints (default: checkpoints)"
        ),
    )
    args = parser.parse_args()

    normal_train_file_paths = discover_normal_training_files(
        args.normal_train_dir
    )
    normal_training_data = load_data(normal_train_file_paths)
    training_data_sha256 = calculate_training_data_sha256(normal_training_data)
    print(
        f"Loaded {len(normal_training_data)} independent normal training samples "
        f"from {len(normal_train_file_paths)} files under "
        f"{Path(args.normal_train_dir).resolve()}"
    )
    print(f"Normal training data SHA256: {training_data_sha256}")

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
                normal_training_data,
                len(normal_train_file_paths),
                detection_profile,
                args.checkpoint_dir,
                training_data_sha256,
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
