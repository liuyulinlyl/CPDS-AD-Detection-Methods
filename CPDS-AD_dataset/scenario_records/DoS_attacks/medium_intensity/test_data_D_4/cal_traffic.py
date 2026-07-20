import re
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd

LOG_FILE = "log_20260630_3.log"
OUTPUT_FILE = "traffic_data.xlsx"
TIME_RANGES_FILE = "attack_time_ranges.csv"

WINDOW_SECONDS = 5   # Customize the time window length here, in seconds

# Timestamp format
TIME_FMT = "%Y/%m/%d %H:%M:%S.%f"

# Match the timestamp at the start of each line
time_pattern = re.compile(
    r"^(\d{4}/\d{1,2}/\d{1,2} \d{1,2}:\d{2}:\d{2}\.\d+)"
)

# Match hexadecimal bytes
hex_pattern = re.compile(r"\b[0-9A-Fa-f]{2}\b")

traffic = defaultdict(int)

def parse_attack_time(value):
    value = str(value).strip()
    for time_format in (
        "%Y/%m/%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%H:%M:%S.%f",
        "%H:%M:%S",
    ):
        try:
            return datetime.strptime(value, time_format).time()
        except ValueError:
            continue
    raise ValueError(f"Unsupported attack time format: {value}")


# Read attack time ranges from the CSV file
def read_attack_times(file_path):
    attack_times = []
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        # Parse attack time ranges and compare only hour, minute, and second
        start_time = parse_attack_time(row['Start Time'])
        end_time = parse_attack_time(row['End Time'])
        attack_times.append((start_time, end_time))
    return attack_times

# Check whether the time window overlaps with an attack time range
def is_in_attack_range(timestamp, window_seconds, attack_times):
    # Calculate the end time of this time window
    window_end_time = timestamp + timedelta(seconds=window_seconds)
    
    # Extract the time portion and ignore the date
    timestamp_time = timestamp.time()
    window_end_time_time = window_end_time.time()
    
    for start_time, end_time in attack_times:
        # Check whether the time window overlaps with the attack time range
        if (timestamp_time < end_time and window_end_time_time > start_time):  # Overlap
            return True
    return False

# Floor the time to the specified time window
def floor_to_window(dt: datetime, window: int) -> datetime:
    total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    floored_seconds = total_seconds - (total_seconds % window)

    hour = floored_seconds // 3600
    minute = (floored_seconds % 3600) // 60
    second = floored_seconds % 60

    return dt.replace(hour=hour, minute=minute, second=second, microsecond=0)

# Read attack time ranges
attack_times = read_attack_times(TIME_RANGES_FILE)

with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        # Extract timestamp
        time_match = time_pattern.search(line)
        if not time_match:
            continue

        try:
            timestamp = datetime.strptime(time_match.group(1), TIME_FMT)
        except ValueError:
            continue

        # Only count records whose minute is between 0 and 47
        if not (0 <= timestamp.minute <= 47):
            continue

        # Extract hexadecimal message bytes
        hex_bytes = hex_pattern.findall(line)
        if not hex_bytes:
            continue

        byte_count = len(hex_bytes)

        # Assign to a custom time bucket
        bucket_time = floor_to_window(timestamp, WINDOW_SECONDS)
        traffic[bucket_time] += byte_count

# Convert to DataFrame
df = pd.DataFrame(
    sorted(traffic.items()),
    columns=["Time", "Traffic_volume"]
)

# Check whether each time point overlaps with an attack time range and add the Labels column
df["Labels"] = df["Time"].apply(lambda x: 1 if is_in_attack_range(x, WINDOW_SECONDS, attack_times) else 0)

# Save to Excel
df.to_excel(OUTPUT_FILE, index=False)

print(f"Processing completed: {len(df)} {WINDOW_SECONDS}s time windows (minutes 0-47). Results saved to {OUTPUT_FILE}")
