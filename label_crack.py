import numpy as np
import os
import pandas as pd


'''config'''
path_input = 'input'
path_output = 'output'
'''config'''

files = os.listdir(path_input)
for file in files:
    pc = pd.read_csv(os.path.join(path_input, file), sep=' ', header=None)
    pc = np.asarray(pc)
    theta = np.arctan2(pc[:, 6], pc[:, 5])
    label_in = np.zeros((theta.shape[0]))
    label_out = np.zeros((theta.shape[0]))
    
    # 9
    if int(file.split('-')[1]) >= 9:
        # 内
        index = np.where((theta >= (- 146.5 / 180 * np.pi)) & (theta <= (- 141.5 / 180 * np.pi)))[0]
        label_in[index] = 1
        # 内
        index = np.where((theta >= (- 154.5 / 180 * np.pi)) & (theta <= (- 149.5 / 180 * np.pi)))[0]
        label_in[index] = 1
        # 内
        index = np.where((theta >= (11.25 / 180 * np.pi)) & (theta <= (16.25 / 180 * np.pi)))[0]
        label_in[index] = 1
        # 内
        index = np.where((theta >= (3.25 / 180 * np.pi)) & (theta <= (8.25 / 180 * np.pi)))[0]
        label_in[index] = 1
    
    # 11
    if int(file.split('-')[1]) >= 11:
        # 内
        index = np.where((theta >= (- 150.5 / 180 * np.pi)) & (theta <= (- 145.5 / 180 * np.pi)))[0]
        label_in[index] = 1
    
    # 32
    if int(file.split('-')[1]) >= 32:
        # 内
        index = np.where((theta >= (7.25 / 180 * np.pi)) & (theta <= (12.25 / 180 * np.pi)))[0]
        label_in[index] = 1
    
    # 34
    if int(file.split('-')[1]) >= 34:
        # 内
        index = np.where((theta >= (- 138.5 / 180 * np.pi)) & (theta <= (- 133.5 / 180 * np.pi)))[0]
        label_in[index] = 1
        # 内
        index = np.where((theta >= (- 162.5 / 180 * np.pi)) & (theta <= (- 157.5 / 180 * np.pi)))[0]
        label_in[index] = 1
        # 内
        index = np.where((theta >= (15.25 / 180 * np.pi)) & (theta <= (20.25 / 180 * np.pi)))[0]
        label_in[index] = 1
        # 内
        index = np.where((theta >= (- 6.75 / 180 * np.pi)) & (theta <= (- 1.75 / 180 * np.pi)))[0]
        label_in[index] = 1
        # 外
        index = np.where((theta >= (- 70 / 180 * np.pi)) & (theta <= (- 65 / 180 * np.pi)))[0]
        label_out[index] = 2
        # 外
        index = np.where((theta >= (- 58.75 / 180 * np.pi)) & (theta <= (- 53.75 / 180 * np.pi)))[0]
        label_out[index] = 2
    
    # 77
    if int(file.split('-')[1]) >= 77:
        # 内
        index = np.where((theta >= (23.25 / 180 * np.pi)) & (theta <= (28.25 / 180 * np.pi)))[0]
        label_in[index] = 1
    
    # 78
    if int(file.split('-')[1]) >= 78:
        # 内
        index = np.where((theta >= (- 130.5 / 180 * np.pi)) & (theta <= (- 125.5 / 180 * np.pi)))[0]
        label_in[index] = 1
        # 内
        index = np.where((theta >= (- 134.5 / 180 * np.pi)) & (theta <= (- 129.5 / 180 * np.pi)))[0]
        label_in[index] = 1
        # 内
        index = np.where((theta >= (- 0.75 / 180 * np.pi)) & (theta <= (4.25 / 180 * np.pi)))[0]
        label_in[index] = 1
    
    # 98
    if int(file.split('-')[1]) >= 98:
        # 内
        index = np.where((theta >= (- 158.5 / 180 * np.pi)) & (theta <= (- 153.5 / 180 * np.pi)))[0]
        label_in[index] = 1
        # 内
        index = np.where((theta >= (27.25 / 180 * np.pi)) & (theta <= (32.25 / 180 * np.pi)))[0]
        label_in[index] = 1
        # 内
        index = np.where((theta >= (19.25 / 180 * np.pi)) & (theta <= (24.25 / 180 * np.pi)))[0]
        label_in[index] = 1
    
    # 99
    if int(file.split('-')[1]) >= 99:
        # 内
        index = np.where((theta >= (- 166.5 / 180 * np.pi)) & (theta <= (- 161.5 / 180 * np.pi)))[0]
        label_in[index] = 1
        
    # 100
    if int(file.split('-')[1]) >= 100:
        # 内
        index = np.where((theta >= (- 142.5 / 180 * np.pi)) & (theta <= (- 137.5 / 180 * np.pi)))[0]
        label_in[index] = 1
    
    new_pc = np.zeros((theta.shape[0], 7))
    new_pc[:, 0:3] = pc[:, 5:8]
    new_pc[:, 3:5] = pc[:, 3:5]
    new_pc[:, 5] = label_in[:]
    new_pc[:, 6] = label_out[:]
    
    np.savetxt(os.path.join(path_output, file), new_pc, fmt='%.8f %.8f %.8f %d %d %d %d')