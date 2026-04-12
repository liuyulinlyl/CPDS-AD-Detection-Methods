import struct
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def hex_to_float(hex_str):
    # Convert a big-endian 32-bit float hex string to a decimal value
    # Remove spaces and convert to uppercase for consistency
    cleaned_hex = hex_str.replace(" ", "").upper()

    # Validate the length (32 bits = 8 hexadecimal characters)
    if len(cleaned_hex) != 8:
        raise ValueError("输入必须是8字符的十六进制字符串（32位），当前长度: {}".format(len(cleaned_hex)))

    try:
        # Convert the hexadecimal string to a byte sequence
        bytes_data = bytes.fromhex(cleaned_hex)

        # Parse the big-endian floating-point value with struct
        return struct.unpack('>f', bytes_data)[0]
    except ValueError as e:
        raise ValueError("无效的十六进制输入: {}".format(hex_str)) from e


# Example usage
# hex_string = "43 3c e8 14"
# result = hex_to_float(hex_string)
# print(result)


def hex_to_float_IEEE754(hex_str):
    """
    Convert a spaced hexadecimal string with swapped word order
    to an IEEE 754 floating-point value.
    """
    # Split the hexadecimal string into byte tokens
    hex_bytes = hex_str.split()

    # Check the number of bytes
    if len(hex_bytes) not in (4, 8):
        raise ValueError("输入必须是4字节（单精度）或8字节（双精度）的十六进制字符串")

    # Reorder the bytes by swapping the two groups
    if len(hex_bytes) == 4:
        # For 4-byte input, swap the first two and last two bytes
        reordered_bytes = hex_bytes[2:4] + hex_bytes[0:2]
    else:
        # For 8-byte input, swap the first four and last four bytes
        reordered_bytes = hex_bytes[4:8] + hex_bytes[0:4]

    # Combine the reordered bytes into one continuous string
    combined_hex = ''.join(reordered_bytes)

    # Convert the hexadecimal string to bytes
    byte_data = bytes.fromhex(combined_hex)

    # Determine the floating-point type from the byte length
    if len(byte_data) == 4:
        # Single-precision float (32-bit)
        return struct.unpack('>f', byte_data)[0]
    elif len(byte_data) == 8:
        # Double-precision float (64-bit)
        return struct.unpack('>d', byte_data)[0]


# print(hex_to_float_IEEE754("99 9a 43 5d"))


def split_at_info(input_str):
    """
    Split the input string at the information marker and return the
    prefix and suffix.

    Args:
        input_str: Input message string.

    Returns:
        A tuple of (before_info, after_info). If the marker is not found,
        the function returns (input_str, "").
    """
    # Find the position of the information marker
    index = input_str.find("信息:")

    if index == -1:
        # If the marker is not found, return the original string and an empty suffix
        return input_str, ""
    else:
        # Compute the split position, including the marker itself
        end_index = index + len("信息:")

        # Extract the prefix up to and including the marker
        before_info = input_str[:end_index]

        # Extract the remaining suffix
        after_info = input_str[end_index:]

        return before_info, after_info


# Example usage
# test_str = "sample message with an information section"
# before, after = split_at_info(test_str)
# print(after)


def get_data_3phase_meter(message_str):
    # Parse one three-phase meter message and return the measurements
    message_str_before, message_str_numbers = split_at_info(message_str)
    n = 3  # Each message block spans three positions
    U_a = hex_to_float(message_str_numbers[3 * n:7 * n - 1])
    U_b = hex_to_float(message_str_numbers[7 * n:11 * n - 1])
    U_c = hex_to_float(message_str_numbers[11 * n:15 * n - 1])
    U_ab = hex_to_float(message_str_numbers[15 * n:19 * n - 1])
    U_bc = hex_to_float(message_str_numbers[19 * n:23 * n - 1])
    U_ac = hex_to_float(message_str_numbers[23 * n:27 * n - 1])
    I_a = hex_to_float(message_str_numbers[27 * n:31 * n - 1])
    I_b = hex_to_float(message_str_numbers[31 * n:35 * n - 1])
    I_c = hex_to_float(message_str_numbers[35 * n:39 * n - 1])
    P_a = hex_to_float(message_str_numbers[39 * n:43 * n - 1])
    P_b = hex_to_float(message_str_numbers[43 * n:47 * n - 1])
    P_c = hex_to_float(message_str_numbers[47 * n:51 * n - 1])
    Q_a = hex_to_float(message_str_numbers[55 * n:59 * n - 1])
    Q_b = hex_to_float(message_str_numbers[59 * n:63 * n - 1])
    Q_c = hex_to_float(message_str_numbers[63 * n:67 * n - 1])
    measurement_data = [U_a, U_b, U_c, U_ab, U_bc, U_ac, I_a, I_b, I_c, P_a, P_b, P_c, Q_a, Q_b, Q_c]
    return measurement_data


# Example usage for get_data_3phase_meter
# message_str = "sample three-phase meter message"
# measurement_data = get_data_3phase_meter(message_str)
# print(measurement_data)


def get_data_3phase_meter_IEEE754(message_str):
    # Parse one IEEE754 three-phase meter message and return the measurements
    message_str_before, message_str_numbers = split_at_info(message_str)
    n = 3  # Each message block spans three positions
    U_a = hex_to_float_IEEE754(message_str_numbers[3 * n:7 * n - 1])
    U_b = hex_to_float_IEEE754(message_str_numbers[7 * n:11 * n - 1])
    U_c = hex_to_float_IEEE754(message_str_numbers[11 * n:15 * n - 1])
    U_ab = hex_to_float_IEEE754(message_str_numbers[15 * n:19 * n - 1])
    U_bc = hex_to_float_IEEE754(message_str_numbers[19 * n:23 * n - 1])
    U_ac = hex_to_float_IEEE754(message_str_numbers[23 * n:27 * n - 1])
    I_a = hex_to_float_IEEE754(message_str_numbers[27 * n:31 * n - 1])
    I_b = hex_to_float_IEEE754(message_str_numbers[31 * n:35 * n - 1])
    I_c = hex_to_float_IEEE754(message_str_numbers[35 * n:39 * n - 1])
    P_a = hex_to_float_IEEE754(message_str_numbers[39 * n:43 * n - 1]) * 1000
    P_b = hex_to_float_IEEE754(message_str_numbers[43 * n:47 * n - 1]) * 1000
    P_c = hex_to_float_IEEE754(message_str_numbers[47 * n:51 * n - 1]) * 1000
    Q_a = hex_to_float_IEEE754(message_str_numbers[55 * n:59 * n - 1]) * 1000
    Q_b = hex_to_float_IEEE754(message_str_numbers[59 * n:63 * n - 1]) * 1000
    Q_c = hex_to_float_IEEE754(message_str_numbers[63 * n:67 * n - 1]) * 1000
    measurement_data = [U_a, U_b, U_c, U_ab, U_bc, U_ac, I_a, I_b, I_c, P_a, P_b, P_c, Q_a, Q_b, Q_c]
    return measurement_data


# Example usage for get_data_3phase_meter_IEEE754
# message_str = "sample IEEE754 three-phase meter message"
# measurement_data = get_data_3phase_meter_IEEE754(message_str)
# print(measurement_data)


def get_data_load_cabinet_meter(message_str):
    # Parse one controllable three-phase load cabinet message
    message_str_before, message_str_numbers = split_at_info(message_str)
    n = 3  # Each message block spans three positions
    U_a = hex_to_float(message_str_numbers[9 * n:13 * n - 1])
    U_b = hex_to_float(message_str_numbers[13 * n:17 * n - 1])
    U_c = hex_to_float(message_str_numbers[17 * n:21 * n - 1])
    U_ab = hex_to_float(message_str_numbers[21 * n:25 * n - 1])
    U_bc = hex_to_float(message_str_numbers[25 * n:29 * n - 1])
    U_ac = hex_to_float(message_str_numbers[29 * n:33 * n - 1])
    I_a = hex_to_float(message_str_numbers[33 * n:37 * n - 1])
    I_b = hex_to_float(message_str_numbers[37 * n:41 * n - 1])
    I_c = hex_to_float(message_str_numbers[41 * n:45 * n - 1])
    P_a = hex_to_float(message_str_numbers[45 * n:49 * n - 1]) * 1000
    P_b = hex_to_float(message_str_numbers[49 * n:53 * n - 1]) * 1000
    P_c = hex_to_float(message_str_numbers[53 * n:57 * n - 1]) * 1000
    Q_a = hex_to_float(message_str_numbers[61 * n:65 * n - 1]) * 1000
    Q_b = hex_to_float(message_str_numbers[65 * n:69 * n - 1]) * 1000
    Q_c = hex_to_float(message_str_numbers[69 * n:73 * n - 1]) * 1000

    measurement_data = [U_a, U_b, U_c, U_ab, U_bc, U_ac, I_a, I_b, I_c, P_a, P_b, P_c, Q_a, Q_b, Q_c]
    return measurement_data


# Example usage for get_data_load_cabinet_meter
# message_str = "sample controllable load cabinet message"
# measurement_data = get_data_load_cabinet_meter(message_str)
# print(measurement_data)


def get_data_line_cabinet_meter(message_str):
    # Parse one line impedance simulation cabinet message
    message_str_before, message_str_numbers = split_at_info(message_str)
    n = 3  # Each message block spans three positions
    U_a_front = hex_to_float(message_str_numbers[9 * n:13 * n - 1])
    U_b_front = hex_to_float(message_str_numbers[13 * n:17 * n - 1])
    U_c_front = hex_to_float(message_str_numbers[17 * n:21 * n - 1])
    U_ab_front = hex_to_float(message_str_numbers[21 * n:25 * n - 1])
    U_bc_front = hex_to_float(message_str_numbers[25 * n:29 * n - 1])
    U_ac_front = hex_to_float(message_str_numbers[29 * n:33 * n - 1])
    I_a_front = hex_to_float(message_str_numbers[33 * n:37 * n - 1])
    I_b_front = hex_to_float(message_str_numbers[37 * n:41 * n - 1])
    I_c_front = hex_to_float(message_str_numbers[41 * n:45 * n - 1])
    P_a_front = hex_to_float(message_str_numbers[45 * n:49 * n - 1]) * 1000
    P_b_front = hex_to_float(message_str_numbers[49 * n:53 * n - 1]) * 1000
    P_c_front = hex_to_float(message_str_numbers[53 * n:57 * n - 1]) * 1000
    Q_a_front = hex_to_float(message_str_numbers[61 * n:65 * n - 1]) * 1000
    Q_b_front = hex_to_float(message_str_numbers[65 * n:69 * n - 1]) * 1000
    Q_c_front = hex_to_float(message_str_numbers[69 * n:73 * n - 1]) * 1000

    U_a_end = hex_to_float(message_str_numbers[113 * n:117 * n - 1])
    U_b_end = hex_to_float(message_str_numbers[117 * n:121 * n - 1])
    U_c_end = hex_to_float(message_str_numbers[121 * n:125 * n - 1])
    U_ab_end = hex_to_float(message_str_numbers[125 * n:129 * n - 1])
    U_bc_end = hex_to_float(message_str_numbers[129 * n:133 * n - 1])
    U_ac_end = hex_to_float(message_str_numbers[133 * n:137 * n - 1])
    I_a_end = hex_to_float(message_str_numbers[137 * n:141 * n - 1])
    I_b_end = hex_to_float(message_str_numbers[141 * n:145 * n - 1])
    I_c_end = hex_to_float(message_str_numbers[145 * n:149 * n - 1])
    P_a_end = hex_to_float(message_str_numbers[149 * n:153 * n - 1]) * 1000
    P_b_end = hex_to_float(message_str_numbers[153 * n:157 * n - 1]) * 1000
    P_c_end = hex_to_float(message_str_numbers[157 * n:161 * n - 1]) * 1000
    Q_a_end = hex_to_float(message_str_numbers[165 * n:169 * n - 1]) * 1000
    Q_b_end = hex_to_float(message_str_numbers[169 * n:173 * n - 1]) * 1000
    Q_c_end = hex_to_float(message_str_numbers[173 * n:177 * n - 1]) * 1000

    measurement_data = [
        U_a_front, U_b_front, U_c_front, U_ab_front, U_bc_front, U_ac_front,
        I_a_front, I_b_front, I_c_front, P_a_front, P_b_front, P_c_front, Q_a_front, Q_b_front, Q_c_front,
        U_a_end, U_b_end, U_c_end, U_ab_end, U_bc_end, U_ac_end,
        I_a_end, I_b_end, I_c_end, P_a_end, P_b_end, P_c_end, Q_a_end, Q_b_end, Q_c_end
    ]
    return measurement_data


# Example usage for get_data_line_cabinet_meter
# message_str = "sample line impedance simulation cabinet message"
# measurement_data = get_data_line_cabinet_meter(message_str)
# print(measurement_data)


def hex_str_to_decimal(hex_str):
    # Convert a hexadecimal string to a decimal integer
    # Remove all spaces from the string
    cleaned_hex = ''.join(hex_str.split())
    # Convert the hexadecimal string to a decimal integer
    return int(cleaned_hex, 16)


# print(hex_str_to_decimal("f4 ad"))  # Expected output: 62637
# print(hex_str_to_decimal("1A 3F"))   # Expected output: 6719
# print(hex_str_to_decimal("00 FF"))   # Expected output: 255


def get_data_1phase_meter(message_str):
    # Parse one single-phase smart meter message
    message_str_before, message_str_numbers = split_at_info(message_str)
    n = 3  # Each message block spans three positions
    U = hex_str_to_decimal(message_str_numbers[3 * n:5 * n - 1]) * 0.01
    I = hex_str_to_decimal(message_str_numbers[5 * n:7 * n - 1]) * 0.001
    P = hex_str_to_decimal(message_str_numbers[9 * n:11 * n - 1]) * 0.1
    Q = hex_str_to_decimal(message_str_numbers[17 * n:19 * n - 1]) * 0.1
    measurement_data = [U, I, P, Q]
    return measurement_data


# Example usage for get_data_1phase_meter
# message_str = "sample single-phase smart meter message"
# measurement_data = get_data_1phase_meter(message_str)
# print(measurement_data)


def extract_time(log_string):
    """
    Extract the timestamp portion from a log string before ":Tag".

    Args:
        log_string (str): Log string in the form "timestamp:Tag: ...".

    Returns:
        str: Extracted timestamp string with trailing spaces removed.
    """
    # Find the position of ":Tag"
    tag_index = log_string.find(":Tag")

    # If ":Tag" is found, extract the preceding substring
    if tag_index != -1:
        # Return the timestamp portion without trailing spaces
        return log_string[:tag_index].rstrip()
    else:
        # Return an empty string when the marker is not found
        return ""


# log = "2025/9/30 14:30:50.216:Tag: sample single-phase meter message"
# time_part = extract_time(log)
# print(time_part)


import pandas as pd
from datetime import datetime


def find_closest_time_index(df, time_column, target_time_str):
    """
    Find the index of the row whose timestamp is closest to the target time.

    Args:
        df: DataFrame containing time strings.
        time_column: Name of the column that stores the time strings.
        target_time_str: Target timestamp string, for example
            "2025/9/8 16:07:21.540".

    Returns:
        The index of the row with the closest timestamp.
    """
    # Convert the target timestamp string to a datetime object
    target_time = datetime.strptime(target_time_str, "%Y/%m/%d %H:%M:%S.%f")

    # Convert the DataFrame time column to datetime values
    time_series = pd.to_datetime(df[time_column], format="%Y/%m/%d %H:%M:%S.%f")

    # Compute the absolute time differences
    time_diffs = (time_series - target_time).abs()

    # Return the index with the smallest time difference
    return time_diffs.idxmin()


# Example DataFrame
# data = {
#     "timestamp": [
#         "2025/9/8 16:01:01.430",
#         "2025/9/8 16:05:22.100",
#         "2025/9/8 16:07:20.990",
#         "2025/9/8 16:10:45.300"
#     ],
#     "value": [10, 20, 30, 40]
# }
# df = pd.DataFrame(data)
#
# Target timestamp string
# target = "2025/9/8 16:05:21.540"
#
# Find the closest row
# closest_row = find_closest_time_index(df, "timestamp", target)
# print(closest_row)


import pandas as pd


def filter_minutes(df):
    """
    Filter rows whose minute value in the time column is between 0 and 47.

    Args:
        df: DataFrame whose first column is named 'time' and uses the format
            "2025/9/8 16:01:16.464".

    Returns:
        A DataFrame containing only rows with minute values from 0 to 47.
    """
    # Ensure the first column is named 'time'
    df.columns = ['time'] + list(df.columns[1:])

    # Convert to datetime and extract the minute component
    df['datetime'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H:%M:%S.%f')
    df['minute'] = df['datetime'].dt.minute

    # Keep only rows whose minute value is between 0 and 47
    filtered_df = df[(df['minute'] >= 0) & (df['minute'] <= 47)].copy()

    # Drop temporary columns and keep the original time format
    filtered_df.drop(columns=['datetime', 'minute'], inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)

    return filtered_df


# Example input data
# data = {
#     'time': [
#         "2025/9/8 16:01:16.464",  # Minute 1 -> keep
#         "2025/9/8 16:48:59.999",  # Minute 48 -> drop
#         "2025/9/8 16:00:00.000",  # Minute 0 -> keep
#         "2025/9/8 16:49:00.000"   # Minute 49 -> drop
#     ],
#     'value': [1, 2, 3, 4]
# }
# df = pd.DataFrame(data)
#
# Apply the function
# filtered = filter_minutes(df)
# print(filtered)
