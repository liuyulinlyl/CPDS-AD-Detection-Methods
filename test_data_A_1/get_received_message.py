import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
file_path = 'log_20251027_11.log'
message_received = np.empty((0, 1)) 
i=0
line_numbers = []  # Store the line indices of received messages
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.readlines()  # Read all lines from the log file
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
           fmt='%s',          # Save values as strings
           encoding='utf-8')  # Keep UTF-8 compatibility for text output

received_index_path =  r'received_message_index'

FDI_index_array = np.array(line_numbers)
np.save(received_index_path, FDI_index_array)
print(f"已保存FDI_index到: {received_index_path}")
