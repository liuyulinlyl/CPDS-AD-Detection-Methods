import struct
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def hex_to_float(hex_str):
    # 将字符串格式的“32位浮点数，高位在前”转为十进制数字
    # 移除所有空格并转换为大写（确保一致性）
    cleaned_hex = hex_str.replace(" ", "").upper()
    
    # 验证长度（32位 = 8个十六进制字符）
    if len(cleaned_hex) != 8:
        raise ValueError("输入必须是8字符的十六进制字符串（32位），当前长度: {}".format(len(cleaned_hex)))
    
    try:
        # 将十六进制字符串转换为字节序列
        bytes_data = bytes.fromhex(cleaned_hex)
        
        # 使用struct模块解析大端序浮点数
        return struct.unpack('>f', bytes_data)[0]
    except ValueError as e:
        raise ValueError("无效的十六进制输入: {}".format(hex_str)) from e

# # 示例用法
# hex_string = "43 3c e8 14"  
# result = hex_to_float(hex_string)
# print(result) 


def hex_to_float_IEEE754(hex_str):
    """
    将包含空格的十六进制字符串（高位在后）转换为IEEE 754浮点数，
    """
    # 移除空格并分割十六进制字节
    hex_bytes = hex_str.split()
    
    # 检查字节数量
    if len(hex_bytes) not in (4, 8):
        raise ValueError("输入必须是4字节（单精度）或8字节（双精度）的十六进制字符串")
    
    # 重新排列字节顺序：将输入分成两组并交换位置
    if len(hex_bytes) == 4:
        # 对于4字节输入：交换前两个和后两个字节
        reordered_bytes = hex_bytes[2:4] + hex_bytes[0:2]
    else:
        # 对于8字节输入：交换前四个和后四个字节
        reordered_bytes = hex_bytes[4:8] + hex_bytes[0:4]
    
    # 将重新排列后的十六进制字符串组合成连续字符串
    combined_hex = ''.join(reordered_bytes)
    
    # 将十六进制字符串转换为字节
    byte_data = bytes.fromhex(combined_hex)
    
    # 根据字节长度确定浮点数类型
    if len(byte_data) == 4:
        # 单精度浮点数 (32位)
        return struct.unpack('>f', byte_data)[0]
    elif len(byte_data) == 8:
        # 双精度浮点数 (64位)
        return struct.unpack('>d', byte_data)[0]

# print(hex_to_float_IEEE754("99 9a 43 5d"))


def split_at_info(input_str):
    """
    截取字符串中"信息："之前的部分（包含"信息："），返回剩下的部分
    
    参数:
        input_str: 输入字符串
        
    返回:
        一个元组 (before_info, after_info)
        before_info: "信息："之前的部分（包含"信息："）
        after_info: 剩下的部分（"信息："之后的内容）
        
        如果字符串中没有"信息："，则返回 (整个字符串, "")
    """
    # 查找"信息："在字符串中的位置
    index = input_str.find("信息:")
    
    if index == -1:
        # 如果没有找到"信息："，返回整个字符串和空字符串
        return input_str, ""
    else:
        # 计算截取位置（包含"信息："）
        end_index = index + len("信息:")
        
        # 截取"信息："之前的部分（包含"信息："）
        before_info = input_str[:end_index]
        
        # 获取剩下的部分
        after_info = input_str[end_index:]
        
        return before_info, after_info

# # 示例用法
# test_str = "2025/9/30 14:00:14.435:Tag: 可控三项负载柜(1543484) 等级:接收报文 信息:08 da 00 00 00 47 01 03 44 43 48 76 39 43 4d 10 ec 43 52 75 cf 43 af 9a 3e 43 b3 ee e7 43 b1 f3 b6 3f ad be 3f 3f b2 4b 3c 3f b9 8a 0a 3e 62 d6 88 3e 6d 72 c3 3e 7c 3a a3 3f 33 20 fb 3e 1f 3e 28 3e 28 a0 80 3e 35 89 7d 3e fe b4 13"
# before, after = split_at_info(test_str)
# print(after)   # 输出: "这是主要的内容部分"
# 2025/9/30 14:00:24.827:Tag: 三相智能电表5(55) 等级:接收报文 信息:37 03 44 99 9a 43 40 66 66 43 48 33 33 43 4f 33 33 43 aa 4c cd 43 af 40 00 43 ae d2 f2 40 2d 16 87 40 59 e1 48 40 3a bc 02 3f 05 e0 0d 3f 2d b9 f5 3f 1a 38 1d 3f e7 09 6c 3c f9 a0 27 3d 09 bb 99 3d 16 70 3b 3d ce 60 56


def get_data_3phase_meter(message_str):
    # 输入一行三相电表的报文（字符串格式）输出对应的量测变量值
    message_str_before, message_str_numbers = split_at_info(message_str)
    n=3  # 一个报文块占三个位置
    U_a = hex_to_float(message_str_numbers[3*n:7*n-1])
    U_b = hex_to_float(message_str_numbers[7*n:11*n-1])
    U_c = hex_to_float(message_str_numbers[11*n:15*n-1])
    U_ab = hex_to_float(message_str_numbers[15*n:19*n-1])
    U_bc = hex_to_float(message_str_numbers[19*n:23*n-1])
    U_ac = hex_to_float(message_str_numbers[23*n:27*n-1])
    I_a = hex_to_float(message_str_numbers[27*n:31*n-1])
    I_b = hex_to_float(message_str_numbers[31*n:35*n-1])
    I_c = hex_to_float(message_str_numbers[35*n:39*n-1])
    P_a = hex_to_float(message_str_numbers[39*n:43*n-1])
    P_b = hex_to_float(message_str_numbers[43*n:47*n-1])
    P_c = hex_to_float(message_str_numbers[47*n:51*n-1])
    Q_a = hex_to_float(message_str_numbers[55*n:59*n-1])
    Q_b = hex_to_float(message_str_numbers[59*n:63*n-1])
    Q_c = hex_to_float(message_str_numbers[63*n:67*n-1])
    measurement_data = [U_a,U_b,U_c,U_ab,U_bc,U_ac,I_a,I_b,I_c,P_a,P_b,P_c,Q_a,Q_b,Q_c]
    return measurement_data

# #　get_data_3phase_meter函数使用示例
# message_str = "2025/9/30 14:00:34.463:Tag: 三相智能电表3（线路阻抗柜端口5）(53) 等级:接收报文 信息:35 03 44 43 3c 86 a3 43 43 f9 3e 43 4e bb 12 43 a6 80 eb 43 ae 65 96 43 ab 36 30 40 48 bf 2f 3f dd 66 5c 3f 8a b0 33 43 f4 11 ba 43 a7 c2 94 43 5f ce 0a 44 82 ee d4 43 84 26 a2 00 00 00 00 00 00 00 00 43 84 26 a2 0c c6"
# measurement_data = get_data_3phase_meter(message_str)
# print(measurement_data)

def get_data_3phase_meter_IEEE754(message_str):
    # 输入一行三相电表的报文（字符串格式）输出对应的量测变量值
    message_str_before, message_str_numbers = split_at_info(message_str)
    n=3  # 一个报文块占三个位置
    U_a = hex_to_float_IEEE754(message_str_numbers[3*n:7*n-1])
    U_b = hex_to_float_IEEE754(message_str_numbers[7*n:11*n-1])
    U_c = hex_to_float_IEEE754(message_str_numbers[11*n:15*n-1])
    U_ab = hex_to_float_IEEE754(message_str_numbers[15*n:19*n-1])
    U_bc = hex_to_float_IEEE754(message_str_numbers[19*n:23*n-1])
    U_ac = hex_to_float_IEEE754(message_str_numbers[23*n:27*n-1])
    I_a = hex_to_float_IEEE754(message_str_numbers[27*n:31*n-1])
    I_b = hex_to_float_IEEE754(message_str_numbers[31*n:35*n-1])
    I_c = hex_to_float_IEEE754(message_str_numbers[35*n:39*n-1])
    P_a = hex_to_float_IEEE754(message_str_numbers[39*n:43*n-1])*1000
    P_b = hex_to_float_IEEE754(message_str_numbers[43*n:47*n-1])*1000
    P_c = hex_to_float_IEEE754(message_str_numbers[47*n:51*n-1])*1000
    Q_a = hex_to_float_IEEE754(message_str_numbers[55*n:59*n-1])*1000
    Q_b = hex_to_float_IEEE754(message_str_numbers[59*n:63*n-1])*1000
    Q_c = hex_to_float_IEEE754(message_str_numbers[63*n:67*n-1])*1000
    measurement_data = [U_a,U_b,U_c,U_ab,U_bc,U_ac,I_a,I_b,I_c,P_a,P_b,P_c,Q_a,Q_b,Q_c]
    return measurement_data

# message_str = "2025/9/30 14:31:06.450:Tag: 三相智能电表5(55) 等级:接收报文 信息:37 03 44 e6 66 43 3c b3 33 43 44 80 00 43 4c 00 00 43 a7 4c cd 43 ac d9 9a 43 ab b4 39 40 28 91 68 40 4d 8d 50 40 37 91 00 3e fe 96 53 3f 21 04 19 3f 16 7e 91 3f db d2 89 3c de 09 6c 3c f9 fc b9 3d 07 98 c8 3d bb 0a fc"
# measurement_data = get_data_3phase_meter_IEEE754(message_str)
# print(measurement_data)

def get_data_load_cabinet_meter(message_str):
    # 输入三相可控负载柜的报文（字符串格式）  输出对应的量测变量值
    message_str_before, message_str_numbers = split_at_info(message_str)
    n=3  # 一个报文块占三个位置
    U_a = hex_to_float(message_str_numbers[9*n:13*n-1])
    U_b = hex_to_float(message_str_numbers[13*n:17*n-1])
    U_c = hex_to_float(message_str_numbers[17*n:21*n-1])
    U_ab = hex_to_float(message_str_numbers[21*n:25*n-1])
    U_bc = hex_to_float(message_str_numbers[25*n:29*n-1])
    U_ac = hex_to_float(message_str_numbers[29*n:33*n-1])
    I_a = hex_to_float(message_str_numbers[33*n:37*n-1])
    I_b = hex_to_float(message_str_numbers[37*n:41*n-1])
    I_c = hex_to_float(message_str_numbers[41*n:45*n-1])
    P_a = hex_to_float(message_str_numbers[45*n:49*n-1])*1000
    P_b = hex_to_float(message_str_numbers[49*n:53*n-1])*1000
    P_c = hex_to_float(message_str_numbers[53*n:57*n-1])*1000
    Q_a = hex_to_float(message_str_numbers[61*n:65*n-1])*1000
    Q_b = hex_to_float(message_str_numbers[65*n:69*n-1])*1000
    Q_c = hex_to_float(message_str_numbers[69*n:73*n-1])*1000

    measurement_data = [U_a,U_b,U_c,U_ab,U_bc,U_ac,I_a,I_b,I_c,P_a,P_b,P_c,Q_a,Q_b,Q_c]
    return measurement_data

# message_str = "2025/9/30 14:00:35.653:Tag: 可控三项负载柜(1543484) 等级:接收报文 信息:0a 7b 00 00 00 47 01 03 44 43 49 5d 31 43 4c c4 b8 43 51 dd 89 43 af dc d2 43 b3 8b d5 43 b2 14 8a 3f ae 5f ac 3f b1 ce 8f 3f b9 73 69 3e 65 81 29 3e 6c ac c3 3e 7d 10 d3 3f 33 cf b0 3e 20 05 9c 3e 28 54 76 3e 33 aa f5 3e fe 02 84"
# measurement_data = get_data_load_cabinet_meter(message_str)
# print(measurement_data)

def get_data_line_cabinet_meter(message_str):
# 输入线路阻抗模拟柜的报文（字符串格式）  输出对应的量测变量值
    message_str_before, message_str_numbers = split_at_info(message_str)
    n=3  # 一个报文块占三个位置
    U_a_front = hex_to_float(message_str_numbers[9*n:13*n-1])
    U_b_front = hex_to_float(message_str_numbers[13*n:17*n-1])
    U_c_front = hex_to_float(message_str_numbers[17*n:21*n-1])
    U_ab_front = hex_to_float(message_str_numbers[21*n:25*n-1])
    U_bc_front = hex_to_float(message_str_numbers[25*n:29*n-1])
    U_ac_front = hex_to_float(message_str_numbers[29*n:33*n-1])
    I_a_front = hex_to_float(message_str_numbers[33*n:37*n-1])
    I_b_front = hex_to_float(message_str_numbers[37*n:41*n-1])
    I_c_front = hex_to_float(message_str_numbers[41*n:45*n-1])
    P_a_front = hex_to_float(message_str_numbers[45*n:49*n-1])*1000
    P_b_front = hex_to_float(message_str_numbers[49*n:53*n-1])*1000
    P_c_front = hex_to_float(message_str_numbers[53*n:57*n-1])*1000
    Q_a_front = hex_to_float(message_str_numbers[61*n:65*n-1])*1000
    Q_b_front = hex_to_float(message_str_numbers[65*n:69*n-1])*1000
    Q_c_front = hex_to_float(message_str_numbers[69*n:73*n-1])*1000

    U_a_end = hex_to_float(message_str_numbers[113*n:117*n-1])
    U_b_end = hex_to_float(message_str_numbers[117*n:121*n-1])
    U_c_end = hex_to_float(message_str_numbers[121*n:125*n-1])
    U_ab_end = hex_to_float(message_str_numbers[125*n:129*n-1])
    U_bc_end = hex_to_float(message_str_numbers[129*n:133*n-1])
    U_ac_end = hex_to_float(message_str_numbers[133*n:137*n-1])
    I_a_end = hex_to_float(message_str_numbers[137*n:141*n-1])
    I_b_end = hex_to_float(message_str_numbers[141*n:145*n-1])
    I_c_end = hex_to_float(message_str_numbers[145*n:149*n-1])
    P_a_end = hex_to_float(message_str_numbers[149*n:153*n-1])*1000
    P_b_end = hex_to_float(message_str_numbers[153*n:157*n-1])*1000
    P_c_end = hex_to_float(message_str_numbers[157*n:161*n-1])*1000
    Q_a_end = hex_to_float(message_str_numbers[165*n:169*n-1])*1000
    Q_b_end = hex_to_float(message_str_numbers[169*n:173*n-1])*1000
    Q_c_end = hex_to_float(message_str_numbers[173*n:177*n-1])*1000

    measurement_data = [U_a_front,U_b_front,U_c_front,U_ab_front,U_bc_front,U_ac_front,
        I_a_front,I_b_front,I_c_front,P_a_front,P_b_front,P_c_front,Q_a_front,Q_b_front,Q_c_front,
        U_a_end,U_b_end,U_c_end,U_ab_end,U_bc_end,U_ac_end,
        I_a_end,I_b_end,I_c_end,P_a_end,P_b_end,P_c_end,Q_a_end,Q_b_end,Q_c_end]
    return measurement_data    

# message_str = "2025/9/6 15:00:21.224:Tag: 线路阻抗模拟柜(1543468) 等级:接收报文 信息:00 2a 00 00 00 af 01 03 ac 43 5c 89 52 43 5c dc 56 43 5d 7a fe 43 bf 21 6d 43 bf 8a 15 43 bf 66 28 41 4d 7d 74 41 35 3b 08 41 04 93 0a 40 11 1e 29 40 06 23 c3 3f c0 d2 17 40 bb d5 7b 3f d9 42 68 3f ae d2 a1 3f 86 28 c3 40 83 8f 73 00 00 00 00 00 00 00 00 00 00 00 00 40 e5 6a 2e 3f 51 99 c4 42 48 19 9d 43 bf 8a 15 41 4d 7d 74 40 11 1e 29 43 3c e8 14 43 3d 06 00 43 49 c3 31 43 a3 a6 04 43 a9 3e c8 43 a9 31 f7 3f 36 39 52 3f 37 d5 3e 3f 30 9d 87 3d d2 9d 46 3d be 84 e8 3d c3 8c 80 3e 95 2b ab 3d b1 6c 85 3d ca 6f f2 3d cf 6e bf 3e 92 d2 cd"
# measurement_data = get_data_line_cabinet_meter(message_str)
# print(measurement_data)


def hex_str_to_decimal(hex_str):
    # 十六进制转十进制
    # 移除字符串中的空格
    cleaned_hex = ''.join(hex_str.split())
    # 将十六进制字符串转换为十进制整数
    return int(cleaned_hex, 16)

# print(hex_str_to_decimal("f4 ad"))  # 输出: 62637
# print(hex_str_to_decimal("1A 3F"))   # 输出: 6719
# print(hex_str_to_decimal("00 FF"))   # 输出: 255

def get_data_1phase_meter(message_str):
    # 输入单相智能电表的报文（字符串格式）  输出对应的量测变量值
    message_str_before, message_str_numbers = split_at_info(message_str)
    n=3  # 一个报文块占三个位置
    U = hex_str_to_decimal(message_str_numbers[3*n:5*n-1])*0.01
    I = hex_str_to_decimal(message_str_numbers[5*n:7*n-1])*0.001
    P = hex_str_to_decimal(message_str_numbers[9*n:11*n-1])*0.1
    Q = hex_str_to_decimal(message_str_numbers[17*n:19*n-1])*0.1
    measurement_data = [U,I,P,Q]
    return measurement_data    
   
# message_str = "2025/9/30 14:30:50.216:Tag: 单相智能电表8(8) 等级:接收报文 信息:08 03 12 4a fa 08 23 00 00 0a 9f 00 00 0f 98 00 00 07 97 00 00 67 44"
# measurement_data = get_data_1phase_meter(message_str)
# print(measurement_data)


def extract_time(log_string):
    """
    从日志字符串中提取时间部分（":Tag"之前的内容）
    
    参数:
        log_string (str): 日志字符串，格式为"时间:Tag: 其他内容"
        
    返回:
        str: 提取的时间字符串（去除尾部空格）
    """
    # 查找":Tag"的位置
    tag_index = log_string.find(":Tag")
    
    # 如果找到":Tag"，提取其前面的部分
    if tag_index != -1:
        # 提取时间部分并去除尾部空格
        return log_string[:tag_index].rstrip()
    else:
        # 如果未找到":Tag"，返回空字符串或根据需求处理
        return ""
    
# log = "2025/9/30 14:30:50.216:Tag: 单相智能电表8(8) 等级:接收报文 信息:"
# time_part = extract_time(log)
# print(time_part)
    
import pandas as pd
from datetime import datetime

def find_closest_time_index(df, time_column, target_time_str):
    """
    在DataFrame中查找与目标时间最接近的行的索引
    
    参数:
    df -- 包含时间字符串的DataFrame
    time_column -- 包含时间字符串的列名
    target_time_str -- 要比较的目标时间字符串（格式："2025/9/8 16:07:21.540"）
    
    返回:
    最接近时间行的索引（整数）
    """
    # 将目标时间字符串转换为datetime对象
    target_time = datetime.strptime(target_time_str, "%Y/%m/%d %H:%M:%S.%f")
    
    # 将DataFrame中的时间列转换为datetime对象
    time_series = pd.to_datetime(df[time_column], format="%Y/%m/%d %H:%M:%S.%f")
    
    # 计算时间差并取绝对值
    time_diffs = (time_series - target_time).abs()
    
    # 找到最小时间差的索引并返回
    return time_diffs.idxmin()

# # 创建示例DataFrame
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

# # 目标时间字符串
# target = "2025/9/8 16:05:21.540"

# # 查找最接近的行
# closest_row = find_closest_time_index(df, "timestamp", target)
# print(closest_row)


import pandas as pd

def filter_minutes(df):
    """
    筛选DataFrame中时间列的分钟在1-48之间的行
    
    参数：
    df : DataFrame - 必须包含名为'time'的第一列，格式为"2025/9/8 16:01:16.464"
    
    返回：
    DataFrame - 仅包含分钟在1-48之间的行
    """
    # 确保第一列名为'time'
    df.columns = ['time'] + list(df.columns[1:])
    
    # 转换为datetime类型并提取分钟
    df['datetime'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H:%M:%S.%f')
    df['minute'] = df['datetime'].dt.minute
    
    # 筛选分钟在1-48之间的行
    filtered_df = df[(df['minute'] >= 0) & (df['minute'] <= 47)].copy()
    
    # 清理临时列并保持原始时间格式
    filtered_df.drop(columns=['datetime', 'minute'], inplace=True)
    filtered_df.reset_index(drop=True, inplace=True)
    
    return filtered_df

# # 创建示例数据
# data = {
#     'time': [
#         "2025/9/8 16:01:16.464",  # 分钟1 → 保留
#         "2025/9/8 16:48:59.999",  # 分钟48 → 保留
#         "2025/9/8 16:00:00.000",  # 分钟0 → 排除
#         "2025/9/8 16:49:00.000"   # 分钟49 → 排除
#     ],
#     'value': [1, 2, 3, 4]
# }
# df = pd.DataFrame(data)

# # 应用函数
# filtered = filter_minutes(df)
# print(filtered)