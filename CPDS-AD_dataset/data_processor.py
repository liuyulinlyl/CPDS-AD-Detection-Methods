import os
import pandas as pd
import pickle

def merge_excel_files_without_first_column(folder_path, skip_time_column=True):
    """
    Merge all Excel files in a folder and drop the first column from each file.

    Args:
    folder_path: Folder path containing Excel files to merge.
    skip_time_column: Whether to skip the time column. Defaults to True.

    Returns:
    merged_df: Merged DataFrame.
    """
    # Ensure the folder path exists.
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹路径不存在: {folder_path}")

    # Get all Excel files in the folder.
    excel_files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]

    if not excel_files:
        raise ValueError(f"文件夹中没有找到Excel文件: {folder_path}")

    print(f"Found {len(excel_files)} Excel files: {excel_files}")

    # Store all DataFrames and source file names.
    dataframes = []
    file_names = []

    # Read all Excel files.
    for file in excel_files:
        file_path = os.path.join(folder_path, file)

        # Use the file stem as the column prefix.
        file_name = os.path.splitext(file)[0]
        file_names.append(file_name)

        # Read the Excel file.
        try:
            df = pd.read_excel(file_path)
            print(f"Read file: {file}, shape: {df.shape}")
            dataframes.append((file_name, df))
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    # Validate that all DataFrames have the same row count.
    first_shape = dataframes[0][1].shape[0]
    for file_name, df in dataframes:
        if df.shape[0] != first_shape:
            raise ValueError(f"文件 {file_name} 的行数 ({df.shape[0]}) 与其他文件不一致")

    # Create the merged DataFrame.
    merged_df = pd.DataFrame()

    # Merge columns from all DataFrames, skipping the first column.
    for file_name, df in dataframes:
        # Rename columns as file_name_original_column.
        for i, col in enumerate(df.columns):
            # Skip the first column.
            if i == 0:
                print(f"Skipping first column in file {file_name}: {col}")
                continue

            # Skip the time column when requested.
            if skip_time_column and col.lower() == 'time':
                print(f"Skipping time column in file {file_name}: {col}")
                continue

            # Create the new column name.
            new_col_name = f"{file_name}_{col}"

            # Add the column to the merged DataFrame.
            merged_df[new_col_name] = df[col]

    print(f"Merge completed, final DataFrame shape: {merged_df.shape}")
    print(f"Number of columns: {len(merged_df.columns)}")

    return merged_df


def merge_excel_files_without_first_column_test(folder_path, skip_time_column=True):
    """
    Merge all Excel files in a folder, drop each first column,
    and create an overall labels column.
    The overall label is 0 only when all file labels are 0; otherwise it is 1.
    """

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹路径不存在: {folder_path}")

    excel_files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]

    if not excel_files:
        raise ValueError(f"文件夹中没有找到Excel文件: {folder_path}")

    print(f"Found {len(excel_files)} Excel files: {excel_files}")

    dataframes = []
    label_columns = []   # Store labels columns from all files.

    # Read all Excel files.
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        file_name = os.path.splitext(file)[0]

        try:
            df = pd.read_excel(file_path)
            print(f"Read file: {file}, shape: {df.shape}")
            dataframes.append((file_name, df))

            # Collect the labels column.
            if 'labels' in df.columns:
                label_columns.append(df['labels'])
            else:
                raise ValueError(f"文件 {file} 中没有labels列")

        except Exception as e:
            print(f"Error reading file {file}: {e}")

    # Validate matching row counts.
    first_shape = dataframes[0][1].shape[0]
    for file_name, df in dataframes:
        if df.shape[0] != first_shape:
            raise ValueError(f"文件 {file_name} 的行数 ({df.shape[0]}) 与其他文件不一致")

    merged_df = pd.DataFrame()

    # Merge feature columns.
    for file_name, df in dataframes:
        for i, col in enumerate(df.columns):

            if i == 0:
                continue

            if skip_time_column and col.lower() == 'time':
                continue

            if col.lower() == 'labels':  # Handle labels separately.
                continue

            new_col_name = f"{file_name}_{col}"
            merged_df[new_col_name] = df[col]

    # Compute overall labels.
    labels_matrix = pd.concat(label_columns, axis=1)

    # Set label to 1 if any source label is 1.
    overall_labels = (labels_matrix.sum(axis=1) > 0).astype(int)

    merged_df['labels'] = overall_labels

    print(f"Merge completed, final DataFrame shape: {merged_df.shape}")
    print(f"Anomaly ratio in overall labels: {merged_df['labels'].mean():.4f}")

    return merged_df


def concat_traffic_data_folders(dataset_dir, folder_prefix, start_index, end_index):
    """
    Concatenate traffic_data.xlsx files from numbered folders in order.
    """
    dataframes = []
    expected_columns = None

    for i in range(start_index, end_index + 1):
        folder_name = f'{folder_prefix}_{i}'
        traffic_data_path = os.path.join(dataset_dir, folder_name, 'traffic_data.xlsx')

        print(f'Processing traffic data: {traffic_data_path}')

        if not os.path.exists(traffic_data_path):
            raise FileNotFoundError(f"traffic_data.xlsx not found: {traffic_data_path}")

        df = pd.read_excel(traffic_data_path)
        print(f"Read traffic data: {folder_name}, shape: {df.shape}")

        if expected_columns is None:
            expected_columns = list(df.columns)
        elif list(df.columns) != expected_columns:
            raise ValueError(
                f"Columns in {traffic_data_path} do not match the first traffic_data.xlsx file."
            )

        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"Traffic data concat completed, final DataFrame shape: {merged_df.shape}")

    return merged_df


REFERENCE_FEATURE_ORDER = ['data_three_phase_meter_4_U_a', 'data_three_phase_meter_4_U_b', 'data_three_phase_meter_4_U_c', 'data_three_phase_meter_4_U_ab', 'data_three_phase_meter_4_U_bc', 'data_three_phase_meter_4_U_ac', 'data_three_phase_meter_4_I_a', 'data_three_phase_meter_4_I_b', 'data_three_phase_meter_4_I_c', 'data_three_phase_meter_1_U_a', 'data_three_phase_meter_1_U_b', 'data_three_phase_meter_1_U_c', 'data_three_phase_meter_1_U_ab', 'data_three_phase_meter_1_U_bc', 'data_three_phase_meter_1_U_ac', 'data_three_phase_meter_1_I_a', 'data_three_phase_meter_1_I_b', 'data_three_phase_meter_1_I_c', 'data_single_phase_meter_4_U', 'data_single_phase_meter_4_I', 'data_single_phase_meter_4_P', 'data_single_phase_meter_4_Q', 'data_single_phase_meter_1_U', 'data_single_phase_meter_1_I', 'data_single_phase_meter_1_P', 'data_single_phase_meter_1_Q', 'data_three_phase_meter_3_U_a', 'data_three_phase_meter_3_U_b', 'data_three_phase_meter_3_U_c', 'data_three_phase_meter_3_U_ab', 'data_three_phase_meter_3_U_bc', 'data_three_phase_meter_3_U_ac', 'data_three_phase_meter_3_I_a', 'data_three_phase_meter_3_I_b', 'data_three_phase_meter_3_I_c', 'data_single_phase_meter_3_U', 'data_single_phase_meter_3_I', 'data_single_phase_meter_3_P', 'data_single_phase_meter_3_Q', 'data_three_phase_meter_5_U_a', 'data_three_phase_meter_5_U_b', 'data_three_phase_meter_5_U_c', 'data_three_phase_meter_5_U_ab', 'data_three_phase_meter_5_U_bc', 'data_three_phase_meter_5_U_ac', 'data_three_phase_meter_5_I_a', 'data_three_phase_meter_5_I_b', 'data_three_phase_meter_5_I_c', 'data_three_phase_meter_5_P_a', 'data_three_phase_meter_5_P_b', 'data_three_phase_meter_5_P_c', 'data_three_phase_meter_5_Q_a', 'data_three_phase_meter_5_Q_b', 'data_three_phase_meter_5_Q_c', 'data_single_phase_meter_8_U', 'data_single_phase_meter_8_I', 'data_single_phase_meter_8_P', 'data_single_phase_meter_8_Q', 'data_single_phase_meter_2_U', 'data_single_phase_meter_2_I', 'data_single_phase_meter_2_P', 'data_single_phase_meter_2_Q', 'data_load_cabinet_meter_U_a', 'data_load_cabinet_meter_U_b', 'data_load_cabinet_meter_U_c', 'data_load_cabinet_meter_U_ab', 'data_load_cabinet_meter_U_bc', 'data_load_cabinet_meter_U_ac', 'data_load_cabinet_meter_I_a', 'data_load_cabinet_meter_I_b', 'data_load_cabinet_meter_I_c', 'data_load_cabinet_meter_P_a', 'data_load_cabinet_meter_P_b', 'data_load_cabinet_meter_P_c', 'data_load_cabinet_meter_Q_a', 'data_load_cabinet_meter_Q_b', 'data_load_cabinet_meter_Q_c', 'data_line_cabinet_meter_U_a_front', 'data_line_cabinet_meter_U_b_front', 'data_line_cabinet_meter_U_c_front', 'data_line_cabinet_meter_U_ab_front', 'data_line_cabinet_meter_U_bc_front', 'data_line_cabinet_meter_U_ac_front', 'data_line_cabinet_meter_I_a_front', 'data_line_cabinet_meter_I_b_front', 'data_line_cabinet_meter_I_c_front', 'data_line_cabinet_meter_P_a_front', 'data_line_cabinet_meter_P_b_front', 'data_line_cabinet_meter_P_c_front', 'data_line_cabinet_meter_Q_a_front', 'data_line_cabinet_meter_Q_b_front', 'data_line_cabinet_meter_Q_c_front', 'data_line_cabinet_meter_U_a_end', 'data_line_cabinet_meter_U_b_end', 'data_line_cabinet_meter_U_c_end', 'data_line_cabinet_meter_U_ab_end', 'data_line_cabinet_meter_U_bc_end', 'data_line_cabinet_meter_U_ac_end', 'data_line_cabinet_meter_I_a_end', 'data_line_cabinet_meter_I_b_end', 'data_line_cabinet_meter_I_c_end', 'data_line_cabinet_meter_P_a_end', 'data_line_cabinet_meter_P_b_end', 'data_line_cabinet_meter_P_c_end', 'data_line_cabinet_meter_Q_a_end', 'data_line_cabinet_meter_Q_b_end', 'data_line_cabinet_meter_Q_c_end', 'data_single_phase_meter_7_U', 'data_single_phase_meter_7_I', 'data_single_phase_meter_7_P', 'data_single_phase_meter_7_Q', 'data_single_phase_meter_9_U', 'data_single_phase_meter_9_I', 'data_single_phase_meter_9_P', 'data_single_phase_meter_9_Q', 'data_single_phase_meter_5_U', 'data_single_phase_meter_5_I', 'data_single_phase_meter_5_P', 'data_single_phase_meter_5_Q', 'data_three_phase_meter_2_U_a', 'data_three_phase_meter_2_U_b', 'data_three_phase_meter_2_U_c', 'data_three_phase_meter_2_U_ab', 'data_three_phase_meter_2_U_bc', 'data_three_phase_meter_2_U_ac', 'data_three_phase_meter_2_I_a', 'data_three_phase_meter_2_I_b', 'data_three_phase_meter_2_I_c', 'data_single_phase_meter_6_U', 'data_single_phase_meter_6_I', 'data_single_phase_meter_6_P', 'data_single_phase_meter_6_Q']


def reorder_measurement_columns(df, reference_feature_order, has_labels=False):
    """
    Reorder measurement feature columns to match the fixed reference order.
    Keep labels as the last column when present.
    Other logic remains unchanged.
    """
    if has_labels:
        feature_columns = [col for col in df.columns if col != 'labels']
    else:
        feature_columns = list(df.columns)

    missing_columns = [col for col in reference_feature_order if col not in feature_columns]
    extra_columns = [col for col in feature_columns if col not in reference_feature_order]

    if missing_columns:
        raise ValueError(f"Missing columns compared with fixed reference order: {missing_columns}")
    if extra_columns:
        raise ValueError(f"Extra columns not found in fixed reference order: {extra_columns}")

    reordered_columns = reference_feature_order.copy()
    if has_labels:
        reordered_columns.append('labels')

    df = df[reordered_columns]
    print("Measurement feature columns have been reordered to match the fixed reference order.")
    return df


current_dir = os.path.abspath(os.path.dirname(__file__))
dataset_dir = current_dir
output_dir = current_dir

reference_feature_order = REFERENCE_FEATURE_ORDER

merged_dfs = []          # 用于最终拼接
train_data_dict = {}     # 用于保存字典

# 循环 train_data_1 到 train_data_25
for i in range(1, 26):
    folder_name = f'train_data_{i}'
    combined_path = os.path.join(dataset_dir, folder_name)

    print(f'Processing: {folder_name}')

    df = merge_excel_files_without_first_column(combined_path)
    df = reorder_measurement_columns(df, reference_feature_order, has_labels=False)

    merged_dfs.append(df)
    train_data_dict[folder_name] = df

# 合并所有数据
merged_df = pd.concat(merged_dfs, ignore_index=True)
merged_df = reorder_measurement_columns(merged_df, reference_feature_order, has_labels=False)

# 保存总训练数据
output_path = os.path.join(output_dir, 'train_data.xlsx')
merged_df.to_excel(output_path, index=True)
print(f"Merged training data saved to: {output_path}")


test_dfs = []
# 循环 test_data_A_1 到 test_data_A_5
for i in range(1, 6):
    folder_name = f'test_data_A_{i}'
    combined_test_path = os.path.join(dataset_dir, folder_name)

    print(f'Processing test set: {folder_name}')

    if os.path.exists(combined_test_path):
        df = merge_excel_files_without_first_column_test(combined_test_path)
        df = reorder_measurement_columns(df, reference_feature_order, has_labels=True)
        test_dfs.append(df)
    else:
        print(f'Warning: folder does not exist: {combined_test_path}')

# 合并所有测试数据
merged_test_df = pd.concat(test_dfs, ignore_index=True)
merged_test_df = reorder_measurement_columns(merged_test_df, reference_feature_order, has_labels=True)

# 保存
output_path = os.path.join(output_dir, 'test_data_A.xlsx')
merged_test_df.to_excel(output_path, index=True)

print(f"Merged test data saved to: {output_path}")


test_dfs = []
# 循环 test_data_R_1 到 test_data_R_5
for i in range(1, 6):
    folder_name = f'test_data_R_{i}'
    combined_test_path = os.path.join(dataset_dir, folder_name)

    print(f'Processing test set: {folder_name}')

    if os.path.exists(combined_test_path):
        df = merge_excel_files_without_first_column_test(combined_test_path)
        df = reorder_measurement_columns(df, reference_feature_order, has_labels=True)
        test_dfs.append(df)
    else:
        print(f'Warning: folder does not exist: {combined_test_path}')

# 合并所有测试数据
merged_test_df = pd.concat(test_dfs, ignore_index=True)
merged_test_df = reorder_measurement_columns(merged_test_df, reference_feature_order, has_labels=True)

# 保存
output_path = os.path.join(output_dir, 'test_data_R.xlsx')
merged_test_df.to_excel(output_path, index=True)

print(f"Merged test data saved to: {output_path}")


test_dfs = []
# 循环 test_data_S_1 到 test_data_S_5
for i in range(1, 6):
    folder_name = f'test_data_S_{i}'
    combined_test_path = os.path.join(dataset_dir, folder_name)

    print(f'Processing test set: {folder_name}')

    if os.path.exists(combined_test_path):
        df = merge_excel_files_without_first_column_test(combined_test_path)
        df = reorder_measurement_columns(df, reference_feature_order, has_labels=True)
        test_dfs.append(df)
    else:
        print(f'Warning: folder does not exist: {combined_test_path}')

# 合并所有测试数据
merged_test_df = pd.concat(test_dfs, ignore_index=True)
merged_test_df = reorder_measurement_columns(merged_test_df, reference_feature_order, has_labels=True)

# 保存
output_path = os.path.join(output_dir, 'test_data_S.xlsx')
merged_test_df.to_excel(output_path, index=True)

print(f"Merged test data saved to: {output_path}")


merged_traffic_df = concat_traffic_data_folders(dataset_dir, 'test_data_D', 1, 5)
output_path = os.path.join(output_dir, 'test_data_D.xlsx')
merged_traffic_df.to_excel(output_path, index=False)

print(f"Merged traffic data saved to: {output_path}")
