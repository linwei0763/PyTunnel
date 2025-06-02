import numpy as np
import os
import pandas as pd


'''------config------'''
path_i = '../Seg2Tunnel/seg2tunnel'    
tunnel_no = ['1', '2', '3', '4', '5']
'''------config------'''

rings = {}
for no in tunnel_no:
    rings[no] = []

files = os.listdir(path_i)
for file in files:
    no = file.split('-')[0]
    if no in tunnel_no:
        pc = pd.read_csv(os.path.join(path_i, file), sep=' ', header=None)
        pc = np.asarray(pc)
        mean = np.mean(pc[:, 0:3], axis=0)
        count = []
        for i in range(8):
            count.append(pc[pc[:, 4] == i].size)
        d = np.linalg.norm(mean)
        rings[no].append([d,] + count)

for key in rings.keys():
    rings[key] = np.asarray(rings[key])