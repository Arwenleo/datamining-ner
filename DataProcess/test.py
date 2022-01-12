# import numpy as np
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '../data/mydata/test.txt')

data, label = [], []
with open(file_path, 'r') as f:
    # line_data,  line_label = [], []
    # w = []
    # t = []
    count = 0
    for line in f.readlines():
        count += 1
        # print(line)
        if line != '\n':
            # w, t = line.split()
            # print(w)
            # print(t)
            temp = line.split()
            # w = temp[0]
            t = temp[1]
            # w.append(temp[0])
            # t.append(temp[1])
            
            print(count)
# print(w)
print(t)
