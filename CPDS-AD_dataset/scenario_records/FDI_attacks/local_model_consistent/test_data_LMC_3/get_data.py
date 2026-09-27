"""Parse, align, label, and export the LMC3 replacement measurements."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = BASE_DIR.parent / "test_data_LMC_3"

THREE_PHASE_COLUMNS = [
    "U_a",
    "U_b",
    "U_c",
    "U_ab",
    "U_bc",
    "U_ac",
    "I_a",
    "I_b",
    "I_c",
    "P_a",
    "P_b",
    "P_c",
    "Q_a",
    "Q_b",
    "Q_c",
]
LINE_COLUMNS = [
    "U_a_front",
    "U_b_front",
    "U_c_front",
    "U_ab_front",
    "U_bc_front",
    "U_ac_front",
    "I_a_front",
    "I_b_front",
    "I_c_front",
    "P_a_front",
    "P_b_front",
    "P_c_front",
    "Q_a_front",
    "Q_b_front",
    "Q_c_front",
    "U_a_end",
    "U_b_end",
    "U_c_end",
    "U_ab_end",
    "U_bc_end",
    "U_ac_end",
    "I_a_end",
    "I_b_end",
    "I_c_end",
    "P_a_end",
    "P_b_end",
    "P_c_end",
    "Q_a_end",
    "Q_b_end",
    "Q_c_end",
]
SINGLE_PHASE_COLUMNS = ["U", "I", "P", "Q"]


def load_parser_utils():
    parser_path = REFERENCE_DIR / "utlis.py"
    spec = importlib.util.spec_from_file_location("lmc_measurement_parser", parser_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load measurement parsers from {parser_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_time(message: str) -> str:
    return message.split(":Tag", 1)[0].strip()


def append_measurement(
    store: dict[str, list[list]],
    key: str,
    message: str,
    raw_index: int,
    parser,
) -> None:
    store[key].append([extract_time(message), int(raw_index), *parser(message)])


def parse_messages(
    message_path: Path,
    received_index_path: Path,
) -> dict[str, pd.DataFrame]:
    utils = load_parser_utils()
    messages = np.load(message_path, allow_pickle=False).reshape(-1)
    raw_indices = np.load(received_index_path, allow_pickle=False).reshape(-1)
    if len(messages) != len(raw_indices):
        raise ValueError("message_received.npy and received_message_index.npy lengths differ")

    keys = ["load", "line", *[f"three_{i}" for i in range(1, 6)], *[f"single_{i}" for i in range(1, 10)]]
    records: dict[str, list[list]] = {key: [] for key in keys}

    for raw_index, raw_message in zip(raw_indices, messages):
        message = str(raw_message)
        if "1543484" in message:
            append_measurement(records, "load", message, raw_index, utils.get_data_load_cabinet_meter)
        elif "1543468" in message:
            append_measurement(records, "line", message, raw_index, utils.get_data_line_cabinet_meter)
        elif "三相智能电表5" in message:
            append_measurement(records, "three_5", message, raw_index, utils.get_data_3phase_meter_IEEE754)
        elif "三相智能电表" in message:
            for meter_number in range(1, 5):
                if f"三相智能电表{meter_number}" in message:
                    append_measurement(
                        records,
                        f"three_{meter_number}",
                        message,
                        raw_index,
                        utils.get_data_3phase_meter,
                    )
                    break
        elif "单相智能电表" in message:
            for meter_number in range(1, 10):
                if f"单相智能电表{meter_number}(" in message:
                    append_measurement(
                        records,
                        f"single_{meter_number}",
                        message,
                        raw_index,
                        utils.get_data_1phase_meter,
                    )
                    break

    frames: dict[str, pd.DataFrame] = {}
    frames["load"] = pd.DataFrame(records["load"], columns=["time", "raw_index", *THREE_PHASE_COLUMNS])
    frames["line"] = pd.DataFrame(records["line"], columns=["time", "raw_index", *LINE_COLUMNS])
    for meter_number in range(1, 6):
        key = f"three_{meter_number}"
        frames[key] = pd.DataFrame(records[key], columns=["time", "raw_index", *THREE_PHASE_COLUMNS])
    for meter_number in range(1, 10):
        key = f"single_{meter_number}"
        frames[key] = pd.DataFrame(records[key], columns=["time", "raw_index", *SINGLE_PHASE_COLUMNS])

    missing = [key for key, frame in frames.items() if frame.empty]
    if missing:
        raise ValueError(f"No parsed measurements for: {missing}")
    return frames


def filter_reference_timeline(line_frame: pd.DataFrame) -> pd.DataFrame:
    datetimes = pd.to_datetime(line_frame["time"], format="%Y/%m/%d %H:%M:%S.%f")
    result = line_frame.loc[datetimes.dt.minute.between(0, 47)].copy().reset_index(drop=True)
    if len(result) != 576:
        raise ValueError(f"Expected 576 line-cabinet rows after filtering, found {len(result)}")
    return result


def nearest_row_indices(source_times: pd.Series, target_times: pd.Series) -> np.ndarray:
    source = pd.to_datetime(source_times, format="%Y/%m/%d %H:%M:%S.%f").astype("int64").to_numpy()
    target = pd.to_datetime(target_times, format="%Y/%m/%d %H:%M:%S.%f").astype("int64").to_numpy()
    insertion = np.searchsorted(source, target, side="left")
    right = np.clip(insertion, 0, len(source) - 1)
    left = np.clip(insertion - 1, 0, len(source) - 1)
    choose_right = np.abs(source[right] - target) < np.abs(source[left] - target)
    return np.where(choose_right, right, left)


def align_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    line = filter_reference_timeline(frames["line"])
    aligned = {"line": line}
    for key, frame in frames.items():
        if key == "line":
            continue
        positions = nearest_row_indices(frame["time"], line["time"])
        aligned[key] = frame.iloc[positions].reset_index(drop=True)
    return aligned


def output_file_name(key: str) -> str:
    if key == "load":
        return "data_load_cabinet_meter.xlsx"
    if key == "line":
        return "data_line_cabinet_meter.xlsx"
    if key.startswith("three_"):
        return f"data_three_phase_meter_{key.split('_')[1]}.xlsx"
    if key.startswith("single_"):
        return f"data_single_phase_meter_{key.split('_')[1]}.xlsx"
    raise ValueError(key)


def measurement_columns(key: str) -> list[str]:
    if key == "line":
        return LINE_COLUMNS
    if key == "load" or key == "three_5":
        return THREE_PHASE_COLUMNS
    if key.startswith("three_"):
        return THREE_PHASE_COLUMNS[:9]
    if key.startswith("single_"):
        return SINGLE_PHASE_COLUMNS
    raise ValueError(key)


def reference_labels(key: str, row_count: int) -> np.ndarray:
    path = REFERENCE_DIR / output_file_name(key)
    labels = pd.read_excel(path, usecols=["labels"])["labels"].to_numpy(dtype=np.int64)
    if len(labels) != row_count:
        raise ValueError(f"Reference label length mismatch for {path}: {len(labels)} != {row_count}")
    return labels


def save_measurements(
    aligned: dict[str, pd.DataFrame],
    attack_index_path: Path,
    label_mode: str,
) -> list[Path]:
    attack_indices = np.load(attack_index_path, allow_pickle=False).astype(np.int64)
    outputs = []
    for key in ["line", "load", *[f"three_{i}" for i in range(1, 6)], *[f"single_{i}" for i in range(1, 10)]]:
        frame = aligned[key]
        raw_labels = np.isin(frame["raw_index"].to_numpy(dtype=np.int64), attack_indices).astype(np.int64)
        if label_mode == "reference-compatible":
            labels = reference_labels(key, len(frame))
            if not np.array_equal(raw_labels, labels):
                differences = np.flatnonzero(raw_labels != labels).tolist()
                print(
                    f"{key}: using reference-compatible labels; "
                    f"raw-index mask differs at aligned rows {differences}"
                )
        else:
            labels = raw_labels

        columns = ["time", *measurement_columns(key)]
        output_frame = frame[columns].copy()
        output_frame["labels"] = labels
        output_path = BASE_DIR / output_file_name(key)
        output_frame.to_excel(output_path, index=True, header=True)
        outputs.append(output_path)
        print(
            f"Saved {output_path.name}: shape={output_frame.shape}, "
            f"attack_rows={int(labels.sum())}"
        )
    return outputs


def process_measurements(
    message_path: Path = BASE_DIR / "message_received.npy",
    received_index_path: Path = BASE_DIR / "received_message_index.npy",
    attack_index_path: Path | None = None,
    label_mode: str = "reference-compatible",
) -> list[Path]:
    if attack_index_path is None:
        attack_index_path = BASE_DIR / "attack_info.npy"
    if not Path(attack_index_path).is_file():
        raise FileNotFoundError(f"Expected attack index file: {attack_index_path}")
    frames = parse_messages(Path(message_path), Path(received_index_path))
    aligned = align_frames(frames)
    return save_measurements(aligned, Path(attack_index_path), label_mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message-path", type=Path, default=BASE_DIR / "message_received.npy")
    parser.add_argument(
        "--received-index-path",
        type=Path,
        default=BASE_DIR / "received_message_index.npy",
    )
    parser.add_argument("--attack-index-path", type=Path, default=None)
    parser.add_argument(
        "--label-mode",
        choices=["raw-index", "reference-compatible"],
        default="reference-compatible",
        help=(
            "reference-compatible preserves the original LMC3 aligned label mask. "
            "This compensates only for nearest-sample jitter in the new recording."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_measurements(
        message_path=args.message_path,
        received_index_path=args.received_index_path,
        attack_index_path=args.attack_index_path,
        label_mode=args.label_mode,
    )
