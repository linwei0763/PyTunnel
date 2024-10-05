import numpy as np
import os
import pandas as pd


if __name__ == '__main__':

    '''---config---'''
    path_input = 'data'
    path_output = 'result'
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    
    index_scale = [3, 4, 13, 15]
    name_scale = ['intensity', 'segmentation', 'deformation', 'dislocation']
    label = [
        [-1400, -800],
        [1, 2, 3, 4, 5, 6, 7],
        [-0.02, 0, 0.02],
        [0, 0.015],
        ]
    label_r = [
        [0, 255],
        [253, 144, 53, 33, 49, 68, 68],
        [0, 255, 255],
        [255, 255],
        ]
    label_g = [
        [0, 255],
        [231, 215, 183, 145, 104, 67, 1],
        [0, 255, 0],
        [255, 0],
        ]
    label_b = [
        [0, 255],
        [37, 67, 121, 140, 142, 131, 84],
        [255, 255, 0],
        [255, 0],
        ]
    '''---config---'''

    files = os.listdir(path_input)
    pc_all = []
    for file in files:
        pc = pd.read_csv(os.path.join(path_input, file), sep=' ', header=None)
        pc = np.asarray(pc)
        pc_all.append(pc)
    pc_all = np.vstack(pc_all[:])
    pc = pc_all
    
    xyz = pc[:, 0:3]
    xyz = pd.DataFrame(xyz)
    xyz.to_csv(os.path.join(path_output, 'xyz.txt'), sep=',', header=['x', 'y', 'z'], index=False)
    
    pc_all = xyz
    
    for i in range(len(index_scale)):
        label_i = np.asarray(label[i])
        label_r_i = np.asarray(label_r[i]) / 255
        label_g_i = np.asarray(label_g[i]) / 255
        label_b_i = np.asarray(label_b[i]) / 255
        
        scale = pc[:, index_scale[i]]
        scale_r = np.interp(scale, label_i, label_r_i).reshape(-1, 1)
        scale_g = np.interp(scale, label_i, label_g_i).reshape(-1, 1)
        scale_b = np.interp(scale, label_i, label_b_i).reshape(-1, 1)
        rgb = np.hstack((scale_r, scale_g, scale_b))
        rgb = pd.DataFrame(rgb)
        rgb.to_csv(os.path.join(path_output, name_scale[i] + '.txt'), sep=',', header=['r', 'g', 'b'], index=False)
        pc_all = pd.concat([pc_all, rgb], axis=1, join='outer')
        
    pc_all.to_csv(os.path.join(path_output, 'pc_all.txt'), sep=' ', header=False, index=False)
