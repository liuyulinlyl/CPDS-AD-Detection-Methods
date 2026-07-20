from __future__ import absolute_import

import argparse
import gc
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.backends import cudnn

from solvers.solver_LSTM import solver_LSTM
from solvers.solver_TCN import solver_TCN
from solvers.solver_transformer import solver_transformer

main_dir = os.path.dirname(os.path.abspath(__file__))
dataset_root_dir = os.path.join(main_dir, 'CPDS-AD_dataset')
dataset_dir = os.path.join(dataset_root_dir, 'merged_datasets')

DEFAULT_TEST_DATASETS = ["A", "S", "R", "RD", "LMC", "AL"]


def set_global_reproducibility(seed: int) -> None:
    """Configure deterministic inference before model construction."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def save_all_results(results: pd.DataFrame, config: argparse.Namespace) -> None:
    output_path = Path(config.results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    configuration = pd.DataFrame(
        [
            ("test_datasets", ", ".join(config.test_datasets)),
            ("global_seed", config.global_seed),
            ("deterministic_algorithms", True),
            ("transformer_threshold_percentile", config.transformer_threshold_percentile),
            ("LSTM_threshold_percentile", config.LSTM_threshold_percentile),
            ("TCN_threshold_percentile", config.TCN_threshold_percentile),
            ("RD_threshold_percentile", config.RD_threshold_percentile),
            ("AL_threshold_percentile", config.AL_threshold_percentile),
            ("subsequence_length", 576),
            ("transformer_seed", config.transformer_seed),
            ("transformer_window_size", config.transformer_win_size),
            ("LSTM_seed", config.LSTM_seed),
            ("LSTM_window_size", config.LSTM_win_size),
            ("TCN_seed", config.TCN_seed),
            ("TCN_window_size", config.TCN_win_size),
            ("torch_version", torch.__version__),
            ("device", "cuda" if torch.cuda.is_available() else "cpu"),
        ],
        columns=["Setting", "Value"],
    )

    with pd.ExcelWriter(  # pylint: disable=abstract-class-instantiated
        output_path, engine="openpyxl"
    ) as writer:
        results.to_excel(writer, sheet_name="Metrics", index=False)
        configuration.to_excel(writer, sheet_name="Configuration", index=False)

        metrics_sheet = writer.book["Metrics"]  # pylint: disable=no-member
        metrics_sheet.freeze_panes = "A2"
        metrics_sheet.auto_filter.ref = metrics_sheet.dimensions
        widths = {
            "A": 12,
            "B": 15,
            "C": 12,
            "D": 12,
            "E": 12,
            "F": 12,
            "G": 13,
            "H": 13,
            "I": 13,
            "J": 13,
        }
        for column, width in widths.items():
            metrics_sheet.column_dimensions[column].width = width
        for row in metrics_sheet.iter_rows(min_row=2, min_col=7, max_col=10):
            for cell in row:
                cell.number_format = "0.000000"

        config_sheet = writer.book["Configuration"]  # pylint: disable=no-member
        config_sheet.column_dimensions["A"].width = 28
        config_sheet.column_dimensions["B"].width = 45

    print(f"Results saved to {output_path}")


def test_all_datasets(config: argparse.Namespace) -> pd.DataFrame:
    model_specs = [
        (
            "Transformer",
            solver_transformer,
            "transformer_testdata_path",
            "transformer_seed",
        ),
        ("LSTM", solver_LSTM, "LSTM_testdata_path", "LSTM_seed"),
        ("TCN", solver_TCN, "TCN_testdata_path", "TCN_seed"),
    ]
    result_rows = []

    for dataset_name in config.test_datasets:
        test_path = Path(dataset_dir) / f"test_data_{dataset_name}.xlsx"
        if not test_path.exists():
            raise FileNotFoundError(f"Test dataset does not exist: {test_path}")

        print(f"\n{'=' * 72}")
        print(f"Testing dataset {dataset_name}: {test_path}")
        print(f"{'=' * 72}")

        for model_name, solver_class, path_attribute, seed_attribute in model_specs:
            setattr(config, path_attribute, str(test_path))
            set_global_reproducibility(getattr(config, seed_attribute))
            print(f"\n[{dataset_name}] Running {model_name}...")

            model_solver = solver_class(vars(config))
            metrics = model_solver.test(return_details=True)
            result_rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": model_name,
                    **metrics,
                }
            )
            del model_solver
            gc.collect()

    results = pd.DataFrame(
        result_rows,
        columns=[
            "Dataset",
            "Model",
            "TN",
            "FP",
            "FN",
            "TP",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
        ],
    )
    save_all_results(results, config)

    print("\n=== All detection metrics ===")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    return results


def main(config: argparse.Namespace):
    set_global_reproducibility(config.global_seed)
    if config.solver == 'test_all':
        return test_all_datasets(config)
    elif config.solver == 'solver_transformer':
        model_solver = solver_transformer(vars(config))
        if config.transformer_mode == 'train':
            model_solver.train()
        elif config.transformer_mode == 'test':
            model_solver.test()
    elif config.solver == 'solver_LSTM':
        model_solver = solver_LSTM(vars(config))
        if config.LSTM_mode == 'train':
            model_solver.train()
        elif config.LSTM_mode == 'test':
            model_solver.test()
    elif config.solver == 'solver_TCN':
        model_solver = solver_TCN(vars(config))
        if config.TCN_mode == 'train':
            model_solver.train()
        elif config.TCN_mode == 'test':
            model_solver.test()
    else:
        raise ValueError(f"Unsupported solver: {config.solver}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Set the solver to run and choose different argument inputs based on it
    default_solver = "test_all"
    parser.add_argument('--solver', type=str, default=default_solver)
    parser.add_argument('--global_seed', type=int, default=42)
    parser.add_argument(
        '--RD_threshold_percentile',
        type=float,
        default=80.0,
        help=(
            'Threshold percentile used by all detectors only for '
            'test_data_RD.xlsx (80 means the highest 20%% scores are anomalous)'
        ),
    )
    parser.add_argument(
        '--AL_threshold_percentile',
        type=float,
        default=91.3,
        help=(
            'Threshold percentile used by all detectors only for '
            'test_data_AL.xlsx (91.3 means approximately the highest 8.7%% '
            'scores are anomalous)'
        ),
    )
    parser.add_argument(
        '--test_datasets',
        nargs='+',
        default=DEFAULT_TEST_DATASETS,
        help='Dataset suffixes used by test_all (default: A S R RD LMC AL)',
    )
    parser.add_argument(
        '--results_path',
        type=str,
        default=os.path.join(main_dir, 'FDI_detection_performances.xlsx'),
        help='Path of the combined Excel workbook produced by test_all',
    )

    parser.add_argument('--transformer_mode', type=str, default="test",choices=["train", "test"])
    # Basic training parameters
    parser.add_argument('--transformer_batch_size', type=int, default=32)
    parser.add_argument('--transformer_win_size', type=int, default=100)
    parser.add_argument('--transformer_step', type=int, default=1) # Time-series data step size
    parser.add_argument('--transformer_lr', type=float, default=0.001)
    parser.add_argument('--transformer_num_epochs', type=int, default=30)
    parser.add_argument('--transformer_seed', type=int, default=42)
    parser.add_argument('--transformer_threshold_percentile', type=int, default=93)
    # Model architecture
    parser.add_argument('--transformer_in_features', type=int, default=132) 
    parser.add_argument('--transformer_d_model', type=int, default=256) 
    parser.add_argument('--transformer_nheads', type=int, default=8)
    parser.add_argument('--transformer_num_layers', type=int, default=1)
    # Paths
    parser.add_argument('--transformer_checkpoint_path', type=str, default=os.path.join(main_dir,'checkpoints','transformer'))
    parser.add_argument('--transformer_traindata_path', type=str, default=os.path.join(dataset_dir,'train_data.xlsx'))
    parser.add_argument('--transformer_testdata_path', type=str, default=os.path.join(dataset_dir,'test_data_R.xlsx'))
    parser.add_argument('--transformer_pretrained_model_path', type=str, default=os.path.join(main_dir,'checkpoints','transformer','checkpoint.pth'))

    parser.add_argument('--LSTM_mode', type=str, default="test",choices=["train", "test"])
    # Basic training parameters
    parser.add_argument('--LSTM_batch_size', type=int, default=32)
    parser.add_argument('--LSTM_win_size', type=int, default=30)
    parser.add_argument('--LSTM_step', type=int, default=1) # Time-series data step size
    parser.add_argument('--LSTM_lr', type=float, default=0.001)
    parser.add_argument('--LSTM_num_epochs', type=int, default=20)
    parser.add_argument('--LSTM_seed', type=int, default=42)
    parser.add_argument('--LSTM_threshold_percentile', type=int, default=93)
    # Model architecture
    parser.add_argument('--LSTM_in_features', type=int, default=132) 
    parser.add_argument('--LSTM_hidden_dim', type=int, default=256) 
    parser.add_argument('--LSTM_num_layers', type=int, default=1)
    # Paths
    parser.add_argument('--LSTM_checkpoint_path', type=str, default=os.path.join(main_dir,'checkpoints','LSTM'))
    parser.add_argument('--LSTM_traindata_path', type=str, default=os.path.join(dataset_dir,'train_data.xlsx'))
    parser.add_argument('--LSTM_testdata_path', type=str, default=os.path.join(dataset_dir,'test_data_R.xlsx'))
    parser.add_argument('--LSTM_pretrained_model_path', type=str, default=os.path.join(main_dir,'checkpoints','LSTM','checkpoint.pth'))

    parser.add_argument('--TCN_mode', type=str, default="test",choices=["train", "test"])
    # Basic training parameters
    parser.add_argument('--TCN_batch_size', type=int, default=32)
    parser.add_argument('--TCN_win_size', type=int, default=100)
    parser.add_argument('--TCN_step', type=int, default=1) # Time-series data step size
    parser.add_argument('--TCN_lr', type=float, default=0.001)
    parser.add_argument('--TCN_num_epochs', type=int, default=100)
    parser.add_argument('--TCN_seed', type=int, default=42)
    parser.add_argument('--TCN_threshold_percentile', type=int, default=92)
    # Model architecture
    parser.add_argument('--TCN_in_features', type=int, default=132) 
    # Paths
    parser.add_argument('--TCN_checkpoint_path', type=str, default=os.path.join(main_dir,'checkpoints','TCN'))
    parser.add_argument('--TCN_traindata_path', type=str, default=os.path.join(dataset_dir,'train_data.xlsx'))
    parser.add_argument('--TCN_testdata_path', type=str, default=os.path.join(dataset_dir,'test_data_S.xlsx'))
    parser.add_argument('--TCN_pretrained_model_path', type=str, default=os.path.join(main_dir,'checkpoints','TCN','checkpoint.pth'))
        
    parsed_config = parser.parse_args()
    args = vars(parsed_config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    # Print all command-line arguments and their values
    main(parsed_config)
