import numpy as np
import open3d as o3d
import os
import pandas as pd
from sklearn.metrics import confusion_matrix

from module.utils import norm_intensity, grid_sample


def prepare_sparse_data(voxel_size, path_data, sparse_stations, sparse_rings):
    
    input_path = path_data
    files = os.listdir(input_path)
    output_path = path_data + '_' + str(voxel_size) + '_sparse_data'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    stations = {}
    for file in files:
        station = file.rsplit('-', 1)[0]
        if station in sparse_stations:
            if station not in stations.keys():
                stations[station] = []
            stations[station].append(file)
    
    for station in stations.keys():
        pc = []
        for i in range(len(stations[station])):
            ring = pd.read_csv(os.path.join(input_path, stations[station][i]), sep=' ', header=None)
            ring = np.asarray(ring)
            
            new_ring = np.zeros((ring.shape[0], ring.shape[1] + 1))
            new_ring[:, 0:-1] = ring[:, :]
            new_ring[:, -1] = 1 if stations[station][i].split('.')[0] in sparse_rings else 0
            ring = new_ring
            
            pc.append(ring)
        pc = np.vstack(pc)
        
        if voxel_size != 0:
            pc = grid_sample(pc, voxel_size)
        
        pc[:, 3] = norm_intensity(pc[:, 3])
        
        pc[:, 0:3] -= np.mean(pc[:, 0:3], axis=0)
        np.random.shuffle(pc)
        
        index = pc[:, -1] == 1
        pc = pc[index, 0:-1]
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.save(output_path + '/' + station + '.npy', pc)


if __name__ == '__main__':
    
    '''------config------'''
    path_data = '../Seg2Tunnel/seg2tunnel'
    voxel_size = 0.04
    
    flag_prepare = False
    sparse_stations = ['1-4', '1-12', '2-10']
    sparse_rings = ['1-4-199', '1-4-204', '1-4-210', '1-12-634', '2-10-773', '2-10-786']
    
    path_i = 'data'
    '''------config------'''
    
    '''------run------'''
    if flag_prepare:
        prepare_sparse_data(voxel_size, path_data, sparse_stations, sparse_rings)
    
    spar_pcs = []
    for sparse_station in sparse_stations:
        pc = pd.read_csv(os.path.join(path_i, sparse_station + '.txt'), sep=' ', header=None)
        pc = np.asarray(pc)
        spar_pc = np.load(os.path.join(path_data + '_' + str(voxel_size) + '_sparse_data', sparse_station + '.npy'))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[:, 0:3])
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        preds = np.zeros((spar_pc.shape[0], 1))
        for i in range(spar_pc.shape[0]):
            [_, neigh_idx, _] = pcd_tree.search_knn_vector_3d(spar_pc[i, 0:3], 1)
            neigh_idx = np.asarray(neigh_idx, dtype=int)
            preds[i, 0] = pc[neigh_idx[0], -2]
        spar_pc = np.hstack((spar_pc, preds))
        spar_pcs.append(spar_pc)
    spar_pcs = np.vstack(spar_pcs[:])

    conf_matrix = confusion_matrix(spar_pcs[:, -2], spar_pcs[:, -1], labels=np.arange(0, 7, 1))
    gt_classes = np.sum(conf_matrix, axis=1)
    positive_classes = np.sum(conf_matrix, axis=0)
    true_positive_classes = np.diagonal(conf_matrix)

    iou_list = []
    for n in range(0, 7, 1):
        if float(gt_classes[n] + positive_classes[n] - true_positive_classes[n]) != 0:
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        else:
            iou_list.append(0.0)
    mean_iou = sum(iou_list) / float(7)

    correct = np.sum(spar_pcs[:, -2] == spar_pcs[:, -1])
    val_total_correct = correct
    val_total_seen = len(spar_pcs[:, -2])
    oa = val_total_correct / val_total_seen
    
    print(oa)
    print(mean_iou)
    '''------run------'''