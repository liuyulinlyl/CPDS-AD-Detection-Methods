"""Extract received Modbus messages from the generated attacked log."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent


def find_default_log() -> Path:
    log_path = BASE_DIR / "log_20260712_21.log"
    if not log_path.is_file():
        raise FileNotFoundError(
            f"Expected log file {log_path.name} in {BASE_DIR}"
        )
    return log_path


def extract_received_messages(log_path: Path) -> tuple[Path, Path]:
    log_path = Path(log_path)
    messages = []
    line_numbers = []
    with log_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if "接收报文" in line:
                messages.append(line.strip())
                line_numbers.append(line_number)

    if not messages:
        raise ValueError(f"No received messages found in {log_path}")

    message_array = np.asarray(messages, dtype=str).reshape(-1, 1)
    index_array = np.asarray(line_numbers, dtype=np.int64)
    message_path = BASE_DIR / "message_received.npy"
    index_path = BASE_DIR / "received_message_index.npy"
    np.save(message_path, message_array)
    np.save(index_path, index_array)
    np.savetxt(
        BASE_DIR / "message_received.txt",
        message_array,
        fmt="%s",
        encoding="utf-8",
    )

    print(f"Extracted received messages: {len(messages)}")
    print(f"Saved: {message_path}")
    print(f"Saved: {index_path}")
    return message_path, index_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_received_messages(args.log_path or find_default_log())
