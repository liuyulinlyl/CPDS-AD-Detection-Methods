import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utlis import get_data_3phase_meter
from utlis import get_data_3phase_meter_IEEE754
from utlis import get_data_load_cabinet_meter
from utlis import get_data_line_cabinet_meter
from utlis import get_data_1phase_meter
from utlis import extract_time
from utlis import find_closest_time_index
from utlis import filter_minutes
message_received = np.load('message_received.npy')
received_message_index = np.load('received_message_index.npy') #序号
FDI_index = np.load('attack_info.npy')
FDI_index_set = set(FDI_index.tolist())
rows=message_received.shape[0]
# 可控三项负载柜
num_load_cabinet_meter = 0
columns = ['time','raw_index','U_a','U_b','U_c','U_ab','U_bc','U_ac','I_a','I_b','I_c',
                'P_a','P_b','P_c','Q_a','Q_b','Q_c'] 
data_load_cabinet_meter = pd.DataFrame(columns=columns)
data_load_cabinet_meter_new = pd.DataFrame(columns=columns)
# 线路阻抗模拟柜
num_line_cabinet_meter = 0
columns = ['time','raw_index','U_a_front','U_b_front','U_c_front','U_ab_front','U_bc_front','U_ac_front',
        'I_a_front','I_b_front','I_c_front','P_a_front','P_b_front','P_c_front','Q_a_front','Q_b_front','Q_c_front',
        'U_a_end','U_b_end','U_c_end','U_ab_end','U_bc_end','U_ac_end',
        'I_a_end','I_b_end','I_c_end','P_a_end','P_b_end','P_c_end','Q_a_end','Q_b_end','Q_c_end']
data_line_cabinet_meter = pd.DataFrame(columns=columns)
data_line_cabinet_meter_new = pd.DataFrame(columns=columns)
# 三相智能电表
num_three_phase_meter_1 = 0
num_three_phase_meter_2 = 0
num_three_phase_meter_3 = 0
num_three_phase_meter_4 = 0
num_three_phase_meter_5 = 0
num_three_phase_meter_6 = 0
columns = ['time','raw_index','U_a','U_b','U_c','U_ab','U_bc','U_ac','I_a','I_b','I_c',
                'P_a','P_b','P_c','Q_a','Q_b','Q_c'] 
data_three_phase_meter_1 = pd.DataFrame(columns=columns)
data_three_phase_meter_2 = pd.DataFrame(columns=columns)
data_three_phase_meter_3 = pd.DataFrame(columns=columns)
data_three_phase_meter_4 = pd.DataFrame(columns=columns)
data_three_phase_meter_5 = pd.DataFrame(columns=columns)
data_three_phase_meter_6 = pd.DataFrame(columns=columns)
data_three_phase_meter_1_new = pd.DataFrame(columns=columns)
data_three_phase_meter_2_new = pd.DataFrame(columns=columns)
data_three_phase_meter_3_new = pd.DataFrame(columns=columns)
data_three_phase_meter_4_new = pd.DataFrame(columns=columns)
data_three_phase_meter_5_new = pd.DataFrame(columns=columns)
data_three_phase_meter_6_new = pd.DataFrame(columns=columns)
# 单相智能电表
num_single_phase_meter_1 = 0
num_single_phase_meter_2 = 0
num_single_phase_meter_3 = 0
num_single_phase_meter_4 = 0
num_single_phase_meter_5 = 0
num_single_phase_meter_6 = 0
num_single_phase_meter_7 = 0
num_single_phase_meter_8 = 0
num_single_phase_meter_9 = 0
columns = ['time','raw_index','U','I','P','Q']
data_single_phase_meter_1 = pd.DataFrame(columns=columns)
data_single_phase_meter_2 = pd.DataFrame(columns=columns)
data_single_phase_meter_3 = pd.DataFrame(columns=columns)
data_single_phase_meter_4 = pd.DataFrame(columns=columns)
data_single_phase_meter_5 = pd.DataFrame(columns=columns)
data_single_phase_meter_6 = pd.DataFrame(columns=columns)
data_single_phase_meter_7 = pd.DataFrame(columns=columns)
data_single_phase_meter_8 = pd.DataFrame(columns=columns)
data_single_phase_meter_9 = pd.DataFrame(columns=columns)
data_single_phase_meter_1_new = pd.DataFrame(columns=columns)
data_single_phase_meter_2_new = pd.DataFrame(columns=columns)
data_single_phase_meter_3_new = pd.DataFrame(columns=columns)
data_single_phase_meter_4_new = pd.DataFrame(columns=columns)
data_single_phase_meter_5_new = pd.DataFrame(columns=columns)
data_single_phase_meter_6_new = pd.DataFrame(columns=columns)
data_single_phase_meter_7_new = pd.DataFrame(columns=columns)
data_single_phase_meter_8_new = pd.DataFrame(columns=columns)
data_single_phase_meter_9_new = pd.DataFrame(columns=columns)
for i in range(rows):
    print(i)
    k = received_message_index[i]  #对应原始报文的行序号
    message = message_received[i,0] 
    t = extract_time(message)
    if "可控三项负载柜" in message:
        num_load_cabinet_meter = num_load_cabinet_meter + 1
        measurement_data = get_data_load_cabinet_meter(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_load_cabinet_meter.loc[len(data_load_cabinet_meter)] = measurement_data
    if "线路阻抗模拟柜" in message:
        num_line_cabinet_meter = num_line_cabinet_meter + 1
        measurement_data = get_data_line_cabinet_meter(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_line_cabinet_meter.loc[len(data_line_cabinet_meter)] = measurement_data
    if "三相智能电表1" in message:
        num_three_phase_meter_1 = num_three_phase_meter_1 + 1
        measurement_data = get_data_3phase_meter(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_three_phase_meter_1.loc[len(data_three_phase_meter_1)] = measurement_data
    if "三相智能电表2" in message:
        num_three_phase_meter_2 = num_three_phase_meter_2 + 1
        measurement_data = get_data_3phase_meter(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_three_phase_meter_2.loc[len(data_three_phase_meter_2)] = measurement_data
    if "三相智能电表3" in message:
        num_three_phase_meter_3 = num_three_phase_meter_3 + 1
        measurement_data = get_data_3phase_meter(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_three_phase_meter_3.loc[len(data_three_phase_meter_3)] = measurement_data
    if "三相智能电表4" in message:
        num_three_phase_meter_4 = num_three_phase_meter_4 + 1
        measurement_data = get_data_3phase_meter(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_three_phase_meter_4.loc[len(data_three_phase_meter_4)] = measurement_data
    if "三相智能电表5" in message:
        num_three_phase_meter_5 = num_three_phase_meter_5 + 1
        measurement_data = get_data_3phase_meter_IEEE754(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_three_phase_meter_5.loc[len(data_three_phase_meter_5)] = measurement_data
    if "单相智能电表1" in message:
        num_single_phase_meter_1 = num_single_phase_meter_1 + 1
        measurement_data = get_data_1phase_meter(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_single_phase_meter_1.loc[len(data_single_phase_meter_1)] = measurement_data
    if "单相智能电表2" in message:
        num_single_phase_meter_2 = num_single_phase_meter_2 + 1
        measurement_data = get_data_1phase_meter(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_single_phase_meter_2.loc[len(data_single_phase_meter_2)] = measurement_data
    if "单相智能电表3" in message:
        num_single_phase_meter_3 = num_single_phase_meter_3 + 1
        measurement_data = get_data_1phase_meter(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_single_phase_meter_3.loc[len(data_single_phase_meter_3)] = measurement_data
    if "单相智能电表4" in message:
        num_single_phase_meter_4 = num_single_phase_meter_4 + 1
        measurement_data = get_data_1phase_meter(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_single_phase_meter_4.loc[len(data_single_phase_meter_4)] = measurement_data
    if "单相智能电表5" in message:
        num_single_phase_meter_5 = num_single_phase_meter_5 + 1
        measurement_data = get_data_1phase_meter(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_single_phase_meter_5.loc[len(data_single_phase_meter_5)] = measurement_data
    if "单相智能电表6" in message:
        num_single_phase_meter_6 = num_single_phase_meter_6 + 1
        measurement_data = get_data_1phase_meter(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_single_phase_meter_6.loc[len(data_single_phase_meter_6)] = measurement_data
    if "单相智能电表7" in message:
        num_single_phase_meter_7 = num_single_phase_meter_7 + 1
        measurement_data = get_data_1phase_meter(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_single_phase_meter_7.loc[len(data_single_phase_meter_7)] = measurement_data
    if "单相智能电表8" in message:
        num_single_phase_meter_8 = num_single_phase_meter_8 + 1
        measurement_data = get_data_1phase_meter(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_single_phase_meter_8.loc[len(data_single_phase_meter_8)] = measurement_data
    if "单相智能电表9" in message:
        num_single_phase_meter_9 = num_single_phase_meter_9 + 1
        measurement_data = get_data_1phase_meter(message)
        measurement_data.insert(0,k)
        measurement_data.insert(0,t)
        data_single_phase_meter_9.loc[len(data_single_phase_meter_9)] = measurement_data

data_line_cabinet_meter = filter_minutes(data_line_cabinet_meter)
for i in range(len(data_line_cabinet_meter)):
    # 获取第一列第i行的元素
    time_line_cabinet_meter = data_line_cabinet_meter.iloc[i, 0]
    # 可控三项负载柜
    row_index = find_closest_time_index(data_load_cabinet_meter, "time", time_line_cabinet_meter)
    row_to_add = data_load_cabinet_meter.iloc[[row_index]]
    data_load_cabinet_meter_new = pd.concat([data_load_cabinet_meter_new, row_to_add], ignore_index=True)
    # 三相智能电表1
    row_index = find_closest_time_index(data_three_phase_meter_1, "time", time_line_cabinet_meter)
    row_to_add = data_three_phase_meter_1.iloc[[row_index]]
    data_three_phase_meter_1_new = pd.concat([data_three_phase_meter_1_new, row_to_add], ignore_index=True)
    # 三相智能电表2
    row_index = find_closest_time_index(data_three_phase_meter_2, "time", time_line_cabinet_meter)
    row_to_add = data_three_phase_meter_2.iloc[[row_index]]
    data_three_phase_meter_2_new = pd.concat([data_three_phase_meter_2_new, row_to_add], ignore_index=True)
    # 三相智能电表3
    row_index = find_closest_time_index(data_three_phase_meter_3, "time", time_line_cabinet_meter)
    row_to_add = data_three_phase_meter_3.iloc[[row_index]]
    data_three_phase_meter_3_new = pd.concat([data_three_phase_meter_3_new, row_to_add], ignore_index=True)
    # 三相智能电表4
    row_index = find_closest_time_index(data_three_phase_meter_4, "time", time_line_cabinet_meter)
    row_to_add = data_three_phase_meter_4.iloc[[row_index]]
    data_three_phase_meter_4_new = pd.concat([data_three_phase_meter_4_new, row_to_add], ignore_index=True)
    # 三相智能电表5
    row_index = find_closest_time_index(data_three_phase_meter_5, "time", time_line_cabinet_meter)
    row_to_add = data_three_phase_meter_5.iloc[[row_index]]
    data_three_phase_meter_5_new = pd.concat([data_three_phase_meter_5_new, row_to_add], ignore_index=True)
    # 单相智能电表1
    row_index = find_closest_time_index(data_single_phase_meter_1, "time", time_line_cabinet_meter)
    row_to_add = data_single_phase_meter_1.iloc[[row_index]]
    data_single_phase_meter_1_new = pd.concat([data_single_phase_meter_1_new, row_to_add], ignore_index=True)
    # 单相智能电表2
    row_index = find_closest_time_index(data_single_phase_meter_2, "time", time_line_cabinet_meter)
    row_to_add = data_single_phase_meter_2.iloc[[row_index]]
    data_single_phase_meter_2_new = pd.concat([data_single_phase_meter_2_new, row_to_add], ignore_index=True)
    # 单相智能电表3
    row_index = find_closest_time_index(data_single_phase_meter_3, "time", time_line_cabinet_meter)
    row_to_add = data_single_phase_meter_3.iloc[[row_index]]
    data_single_phase_meter_3_new = pd.concat([data_single_phase_meter_3_new, row_to_add], ignore_index=True)
    # 单相智能电表4
    row_index = find_closest_time_index(data_single_phase_meter_4, "time", time_line_cabinet_meter)
    row_to_add = data_single_phase_meter_4.iloc[[row_index]]
    data_single_phase_meter_4_new = pd.concat([data_single_phase_meter_4_new, row_to_add], ignore_index=True)
    # 单相智能电表5
    row_index = find_closest_time_index(data_single_phase_meter_5, "time", time_line_cabinet_meter)
    row_to_add = data_single_phase_meter_5.iloc[[row_index]]
    data_single_phase_meter_5_new = pd.concat([data_single_phase_meter_5_new, row_to_add], ignore_index=True)
    # 单相智能电表6
    row_index = find_closest_time_index(data_single_phase_meter_6, "time", time_line_cabinet_meter)
    row_to_add = data_single_phase_meter_6.iloc[[row_index]]
    data_single_phase_meter_6_new = pd.concat([data_single_phase_meter_6_new, row_to_add], ignore_index=True)
    # 单相智能电表7
    row_index = find_closest_time_index(data_single_phase_meter_7, "time", time_line_cabinet_meter)
    row_to_add = data_single_phase_meter_7.iloc[[row_index]]
    data_single_phase_meter_7_new = pd.concat([data_single_phase_meter_7_new, row_to_add], ignore_index=True)
    # 单相智能电表8
    row_index = find_closest_time_index(data_single_phase_meter_8, "time", time_line_cabinet_meter)
    row_to_add = data_single_phase_meter_8.iloc[[row_index]]
    data_single_phase_meter_8_new = pd.concat([data_single_phase_meter_8_new, row_to_add], ignore_index=True)
    # 单相智能电表9
    row_index = find_closest_time_index(data_single_phase_meter_9, "time", time_line_cabinet_meter)
    row_to_add = data_single_phase_meter_9.iloc[[row_index]]
    data_single_phase_meter_9_new = pd.concat([data_single_phase_meter_9_new, row_to_add], ignore_index=True)

def add_label_column(df):
    df['labels'] = df['raw_index'].apply(
        lambda x: 1 if x in FDI_index_set else 0
    )
    return df

data_load_cabinet_meter_new = add_label_column(data_load_cabinet_meter_new)
data_line_cabinet_meter = add_label_column(data_line_cabinet_meter)
data_three_phase_meter_1_new = add_label_column(data_three_phase_meter_1_new)
data_three_phase_meter_2_new = add_label_column(data_three_phase_meter_2_new)
data_three_phase_meter_3_new = add_label_column(data_three_phase_meter_3_new)
data_three_phase_meter_4_new = add_label_column(data_three_phase_meter_4_new)
data_three_phase_meter_5_new = add_label_column(data_three_phase_meter_5_new)

data_single_phase_meter_1_new = add_label_column(data_single_phase_meter_1_new)
data_single_phase_meter_2_new = add_label_column(data_single_phase_meter_2_new)
data_single_phase_meter_3_new = add_label_column(data_single_phase_meter_3_new)
data_single_phase_meter_4_new = add_label_column(data_single_phase_meter_4_new)
data_single_phase_meter_5_new = add_label_column(data_single_phase_meter_5_new)
data_single_phase_meter_6_new = add_label_column(data_single_phase_meter_6_new)
data_single_phase_meter_7_new = add_label_column(data_single_phase_meter_7_new)
data_single_phase_meter_8_new = add_label_column(data_single_phase_meter_8_new)
data_single_phase_meter_9_new = add_label_column(data_single_phase_meter_9_new)

# 删除raw_index列
data_load_cabinet_meter_new.drop(columns=['raw_index'], inplace=True)
data_line_cabinet_meter.drop(columns=['raw_index'], inplace=True)
data_three_phase_meter_1_new.drop(columns=['raw_index'], inplace=True)
data_three_phase_meter_2_new.drop(columns=['raw_index'], inplace=True)
data_three_phase_meter_3_new.drop(columns=['raw_index'], inplace=True)
data_three_phase_meter_4_new.drop(columns=['raw_index'], inplace=True)
data_three_phase_meter_5_new.drop(columns=['raw_index'], inplace=True)
data_single_phase_meter_1_new.drop(columns=['raw_index'], inplace=True)
data_single_phase_meter_2_new.drop(columns=['raw_index'], inplace=True)
data_single_phase_meter_3_new.drop(columns=['raw_index'], inplace=True)
data_single_phase_meter_4_new.drop(columns=['raw_index'], inplace=True)
data_single_phase_meter_5_new.drop(columns=['raw_index'], inplace=True)
data_single_phase_meter_6_new.drop(columns=['raw_index'], inplace=True)
data_single_phase_meter_7_new.drop(columns=['raw_index'], inplace=True)
data_single_phase_meter_8_new.drop(columns=['raw_index'], inplace=True)
data_single_phase_meter_9_new.drop(columns=['raw_index'], inplace=True)


# 将时间对齐后的各个量测仪表的数据保存到excel文件
data_load_cabinet_meter_new.to_excel('data_load_cabinet_meter.xlsx', index=True, header=True)
data_line_cabinet_meter.to_excel('data_line_cabinet_meter.xlsx', index=True, header=True)

meters = {
    '1': data_three_phase_meter_1_new,
    '2': data_three_phase_meter_2_new,
    '3': data_three_phase_meter_3_new,
    '4': data_three_phase_meter_4_new,
}
for k, df in meters.items():
    cols = list(df.columns[:10]) + ['labels']
    df[cols].to_excel(f'data_three_phase_meter_{k}.xlsx', index=True, header=True)
# data_three_phase_meter_1_new.iloc[:, :10].to_excel('data_three_phase_meter_1.xlsx', index=True, header=True)
# data_three_phase_meter_2_new.iloc[:, :10].to_excel('data_three_phase_meter_2.xlsx', index=True, header=True)
# data_three_phase_meter_3_new.iloc[:, :10].to_excel('data_three_phase_meter_3.xlsx', index=True, header=True)
# data_three_phase_meter_4_new.iloc[:, :10].to_excel('data_three_phase_meter_4.xlsx', index=True, header=True)
data_three_phase_meter_5_new.to_excel('data_three_phase_meter_5.xlsx', index=True, header=True)
data_single_phase_meter_1_new.to_excel('data_single_phase_meter_1.xlsx', index=True, header=True)
data_single_phase_meter_2_new.to_excel('data_single_phase_meter_2.xlsx', index=True, header=True)
data_single_phase_meter_3_new.to_excel('data_single_phase_meter_3.xlsx', index=True, header=True)
data_single_phase_meter_4_new.to_excel('data_single_phase_meter_4.xlsx', index=True, header=True)
data_single_phase_meter_5_new.to_excel('data_single_phase_meter_5.xlsx', index=True, header=True)
data_single_phase_meter_6_new.to_excel('data_single_phase_meter_6.xlsx', index=True, header=True)
data_single_phase_meter_7_new.to_excel('data_single_phase_meter_7.xlsx', index=True, header=True)
data_single_phase_meter_8_new.to_excel('data_single_phase_meter_8.xlsx', index=True, header=True)
data_single_phase_meter_9_new.to_excel('data_single_phase_meter_9.xlsx', index=True, header=True)
