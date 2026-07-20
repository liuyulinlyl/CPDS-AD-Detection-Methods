import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

file_path = 'log_20260227_3.log'
message_received = np.empty((0, 1)) 
i=0
line_numbers = []  # 新增：用于存储行序号的列表
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.readlines()  # 读取所有行
    for line in content:
        # if i>1000:
        #     break
        if "接收报文" in line:
            message_received = np.vstack((message_received, line.strip()))
            line_numbers.append(i) 
        i=i+1
        

np.save('message_received.npy', message_received)
np.savetxt('message_received.txt', 
           message_received, 
           fmt='%s',          # 指定字符串格式
           encoding='utf-8')  # 支持中文和特殊字符

received_index_path =  r'received_message_index'

FDI_index_array = np.array(line_numbers)
np.save(received_index_path, FDI_index_array)
print(f"已从 {file_path} 提取接收报文: {len(line_numbers)} 条")
print(f"已保存 received_message 到: message_received.npy / message_received.txt")
print(f"已保存 received_message_index 到: {received_index_path}.npy")
