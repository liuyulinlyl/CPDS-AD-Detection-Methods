import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

RECEIVE_MARK = "接收报文"
GAN_LOG = None
ATTACK_INDEX_PATH = None


def latest_file(pattern):
    candidates = list(BASE_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def matching_attack_index(log_path):
    return BASE_DIR / "attack_info.npy"


file_path = latest_file("log_*.log") if GAN_LOG is None else GAN_LOG
attack_index_path = matching_attack_index(file_path) if ATTACK_INDEX_PATH is None else ATTACK_INDEX_PATH
if not file_path.exists():
    raise FileNotFoundError(file_path)
if not attack_index_path.exists():
    raise FileNotFoundError(attack_index_path)

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
print(f"使用攻击索引: {attack_index_path.name}")
