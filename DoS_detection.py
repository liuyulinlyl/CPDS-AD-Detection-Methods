import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
# ===============================
def save_results_to_excel(results, output_path='DoS_detection_performances.xlsx'):

    df = pd.DataFrame(results)

    df.to_excel(output_path, index=False)

    print(f"\nResults saved to {output_path}")

# ===============================
def load_data(file_paths):

    data_list = []

    for path in file_paths:
        df = pd.read_excel(path)
        data_list.append(df)

    data = pd.concat(data_list, axis=0, ignore_index=True)

    return data
# ===============================
# Z-score
# ===============================
def z_score_anomaly_detection(data, threshold=3):

    normal_data = data[data['Labels'] == 0]['Bytes']

    mean = normal_data.mean()
    std = normal_data.std()

    data['z_score'] = (data['Bytes'] - mean) / std

    data['predicted_labels'] = data['z_score'].apply(
        lambda x: 1 if abs(x) > threshold else 0
    )

    return data

# ===============================
# Isolation Forest
# ===============================
def isolation_forest_anomaly_detection(data, contamination=0.1):

    model = IsolationForest(contamination=contamination, random_state=42)

    data['predicted_labels'] = model.fit_predict(data[['Bytes']])

    data['predicted_labels'] = data['predicted_labels'].apply(
        lambda x: 1 if x == -1 else 0
    )

    return data

# ===============================
# KNN 
# ===============================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def knn_anomaly_detection(data, n_neighbors=5, use_distance_weights=True):
    # Drop non-numeric columns (e.g., datetime columns)
    data_without_labels = data.drop(columns=['Labels'])
    
    # Convert datetime columns to numeric (e.g., Unix timestamp)
    for column in data_without_labels.select_dtypes(include=['datetime64']).columns:
        data_without_labels[column] = data_without_labels[column].astype('int64') / 10**9  # Convert to Unix timestamp in seconds
    
    # Ensure there are no non-numeric columns
    data_without_labels = data_without_labels.select_dtypes(include=[float, int])

    # Scale data (standardization) to make features comparable
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_without_labels)
    
    # Fit KNN model with adjusted parameters
    labels = data['Labels']
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance' if use_distance_weights else 'uniform')
    
    # GridSearchCV for hyperparameter tuning (optional)
    param_grid = {
        'n_neighbors': [3, 5, 7, 10, 15],
        'weights': ['uniform', 'distance']
    }
    
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1')
    grid_search.fit(scaled_data, labels)
    
    # Best model after tuning
    best_knn = grid_search.best_estimator_

    # Predict labels using the best model
    data['predicted_labels'] = best_knn.predict(scaled_data)
    
    return data

# ===============================
def evaluate_model(true_labels, predicted_labels):

    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return precision, recall, f1

# ===============================
def compare_models(file_paths):

    data = load_data(file_paths)

    if 'Bytes' not in data.columns or 'Labels' not in data.columns:
        raise ValueError("Data must contain 'Bytes' and 'Labels' columns")

    # Z-score
    z_data = z_score_anomaly_detection(data.copy())
    z_p, z_r, z_f = evaluate_model(z_data['Labels'], z_data['predicted_labels'])

    # Isolation Forest
    if_data = isolation_forest_anomaly_detection(data.copy())
    if_p, if_r, if_f = evaluate_model(if_data['Labels'], if_data['predicted_labels'])

    # KNN
    knn_data = knn_anomaly_detection(data.copy())
    knn_p, knn_r, knn_f = evaluate_model(knn_data['Labels'], knn_data['predicted_labels'])

    # Output results
    print("\n============================")
    print("Model Comparison Results")
    print("============================")

    print(f"Z-score: Precision={z_p:.4f} Recall={z_r:.4f} F1={z_f:.4f}")
    print(f"Isolation Forest: Precision={if_p:.4f} Recall={if_r:.4f} F1={if_f:.4f}")
    print(f"KNN: Precision={knn_p:.4f} Recall={knn_r:.4f} F1={knn_f:.4f}")

    # ===============================
    results = [
        {"Model": "Z-score", "Precision": z_p, "Recall": z_r, "F1": z_f},
        {"Model": "Isolation Forest", "Precision": if_p, "Recall": if_r, "F1": if_f},
        {"Model": "KNN", "Precision": knn_p, "Recall": knn_r, "F1": knn_f},
    ]

    save_results_to_excel(results)

# ===============================
if __name__ == "__main__":

    dataset_dir = os.path.join(os.path.dirname(__file__), 'CPDS-AD_dataset')

    file_paths = [
        os.path.join(dataset_dir, 'test_data_D.xlsx')
    ]

    compare_models(file_paths)
