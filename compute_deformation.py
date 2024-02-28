import numpy as np
import os
import pandas as pd

from module.ring import Ring


if __name__ == '__main__':
    
    '''------config------'''
    max_num = 40960
    
    path_i = '../Seg2Tunnel/seg2tunnel'
    # path_i = 'data'
    
    flag_all = True
    flag_all = False
    if not flag_all:
        stations = ['0-103', '1-1', '3-1', '4-1']
    
    r_all = [2.75, 2.75, 2.75, 2.95, 3.75, 3.75]
    num_seg_all = [6, 6, 6, 6, 7, 7]
    v0_dir_0 = [[0, 0, 1], None, None, None, None, None]
    
    path_o = 'result'
    
    index_label = 4
    '''------config------'''
    
    if not os.path.exists(path_o):
        os.makedirs(path_o)

    files = os.listdir(path_i)
    
    
    
    
    
    
    
    
    # ovalization_all = []
    # pd.DataFrame(columns=['name', 'long', 'short', 'theta'])
    
    for file in files:
        
        r = r_all[int(file.split('-')[0])]
        num_seg = num_seg_all[int(file.split('-')[0])]
        
        pc = pd.read_csv(os.path.join(path_i, file), sep=' ', header=None)
        pc = np.asarray(pc)
        pc = pc[pc[:, index_label] != 0, :]
        if pc.shape[0] > max_num:
            np.random.shuffle(pc)
            pc = pc[0:max_num, :]
        xyz = pc[:, 0:3]
        intensity = pc[:, 3]
        label = pc[:, index_label]
    
        '''------call------'''
        # ring = Ring(xyz, intensity, label, r, num_seg)
        # ring.compute_deformation()
        # ovalization = ring.ovalization
        # d = ring.d
        '''------call------'''

    #     new_pc = np.zeros((pc.shape[0], pc.shape[1] + d.shape[1]))
    #     new_pc[:, 0:pc.shape[1]] = pc[:, :]
    #     new_pc[:, pc.shape[1]:] = d[:, :]
    #     new_pc = pd.DataFrame(new_pc)
    #     new_pc.to_csv(os.path.join(path_o, file), sep=' ', header=None, index=None)
        
    #     ovalization = {'name': file.split('.')[0], 'long': ovalization[0], 'short': ovalization[1], 'ang': ovalization[2]}
    #     ovalization_all.append(ovalization)
    
    # ovalization_all = pd.DataFrame(ovalization_all)
    # ovalization_all.to_csv(os.path.join(path_o, 'ovalization.csv'))
    
    










