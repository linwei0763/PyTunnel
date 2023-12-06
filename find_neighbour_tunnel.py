import numpy as np
import os
import pandas as pd

from module.tunnel import Tunnel


if __name__ == '__main__':
    
    '''------config------'''
    path_i = '../Seg2Tunnel/seg2tunnel_0.04'
    file = '1-12'
    
    path_o = 'result'
    
    k_n = 16
    num_layer = 5
    sampling_ratio = [4, 4, 4, 4, 2]
    color_list = [[253, 231, 37], [144, 215, 67], [53, 183, 121], [33, 145, 140], [49, 104, 142], [68, 57, 131], [68, 1, 84]]
    '''------config------'''
    
    if not os.path.exists(path_o):
        os.makedirs(path_o)
    
    pc = np.load(os.path.join(path_i, file + '.npy'))
    np.random.shuffle(pc)
    
    xyz = pc[:, 0:3]
    intensity = pc[:, 3]
    label = pc[:, 4]
    
    all_num_point = pc.shape[0]
    
    tunnel = Tunnel(xyz, intensity, label)
    centre_xyz = pc[0, 0:3]
    local_neigh_index = tunnel.find_local_neighbour(centre_xyz, k_n)
    local_neigh_index = np.asarray(local_neigh_index, dtype=int)
    ring_neigh_index = tunnel.find_ring_neighbour(centre_xyz, k_n)
    ring_neigh_index = np.asarray(ring_neigh_index, dtype=int)
    
    color_local_neigh = np.zeros((all_num_point, 3), dtype=int)
    color_local_neigh[:, :] = 128
    color_local_neigh[local_neigh_index, :] = color_list[label[local_neigh_index], :]
    color_local_neigh[0, :] = [255, 0, 0]
    
    color_ring_neigh = np.zeros((all_num_point, 3), dtype=int)
    color_ring_neigh[:, :] = 128
    color_ring_neigh[ring_neigh_index, :] = color_list[label[ring_neigh_index], :]
    color_ring_neigh[0, :] = [255, 0, 0]
    
    pc_local_neigh = np.hstack(xyz, intensity.reshpe(-1, 1), color_local_neigh)
    pc_ring_neigh = np.hstack(xyz, intensity.reshpe(-1, 1), color_ring_neigh)
    
    
    
    
    
    