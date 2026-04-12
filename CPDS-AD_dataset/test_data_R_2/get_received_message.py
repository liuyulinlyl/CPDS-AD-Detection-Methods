import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
file_path = 'log_20260224_20.log'
message_received = np.empty((0, 1)) 
i=0
line_numbers = []  # 新增：用于存储行序号的列表
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.readlines()  # 读取所有行
    for line in content:
        print(i)
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
print(f"已保存FDI_index到: {received_index_path}")
