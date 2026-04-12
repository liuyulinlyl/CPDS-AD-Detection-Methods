import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
file_path = 'log_20260225_16.log'
message_received = np.empty((0, 1)) 
i=0
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.readlines()  # 读取所有行
    for line in content:
        i=i+1
        print(i)
        # if i>1000:
        #     break
        if "接收报文" in line:
            message_received = np.vstack((message_received, line.strip()))

np.save('message_received.npy', message_received)
np.savetxt('message_received.txt', 
           message_received, 
           fmt='%s',          # 指定字符串格式
           encoding='utf-8')  # 支持中文和特殊字符