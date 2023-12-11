import numpy as np
import os
import pandas as pd

from module.tunnel import Tunnel


if __name__ == '__main__':
    
    '''------config------'''
    path_i = '../Seg2Tunnel/seg2tunnel_0.04'
    file = '1-4'
    
    path_o = 'result'
    
    k_n = 16
    num_layer = 5
    sampling_ratio = [4, 4, 4, 4, 2]
    color_list = np.asarray([[253, 231, 37], [144, 215, 67], [53, 183, 121], [33, 145, 140], [49, 104, 142], [68, 57, 131], [68, 1, 84]], dtype=int)
    '''------config------'''
    
    if not os.path.exists(path_o):
        os.makedirs(path_o)
    
    pc = np.load(os.path.join(path_i, file + '.npy'))
    np.random.shuffle(pc)
    
    for i in range(pc.shape[0]):
        if pc[i, 4] != 1:
            continue
        else:
            pc[0, :], pc[i, :] = pc[i, :], pc[0, :]
            break
    
    centre_xyz = pc[0, 0:3]
    
    xyz = pc[:, 0:3]
    intensity = pc[:, 3]
    label = np.asarray(pc[:, 4], dtype=int)
    
    all_num_point = pc.shape[0]
    
    tunnel = Tunnel(xyz, intensity, label)
    
    local_neigh_index = tunnel.find_local_neighbour(centre_xyz, k_n)
    ring_neigh_index = tunnel.find_ring_neighbour(centre_xyz, k_n)
    
    color_local_neigh = np.zeros((k_n, 3), dtype=int)
    color_local_neigh[:, :] = color_list[label[local_neigh_index], :]
    color_local_neigh[0, :] = [255, 0, 0]
    
    color_ring_neigh = np.zeros((k_n, 3), dtype=int)
    color_ring_neigh[:, :] = color_list[label[ring_neigh_index], :]
    color_ring_neigh[0, :] = [255, 0, 0]
    
    pc_local_neigh = np.hstack([xyz, intensity.reshape(-1, 1), label.reshape(-1, 1), color_local_neigh])
    pc_ring_neigh = np.hstack([xyz, intensity.reshape(-1, 1), label.reshape(-1, 1), color_ring_neigh])
    
    pc_local_neigh = pd.DataFrame(pc_local_neigh)
    pc_local_neigh.to_csv(os.path.join(path_o, 'pc_local_neigh.txt'), sep=' ', header=False, index=False)
    pc_ring_neigh = pd.DataFrame(pc_ring_neigh)
    pc_ring_neigh.to_csv(os.path.join(path_o, 'pc_ring_neigh.txt'), sep=' ', header=False, index=False)
    
    for i in range(len(sampling_ratio))
    
    
    
    pc = pd.DataFrame(pc)
    pc.to_csv(os.path.join(path_o, 'pc.txt'), sep=' ', header=False, index=False)
    
    
    
    
    