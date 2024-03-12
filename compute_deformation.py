import numpy as np
import os
import pandas as pd

from module.ring import Ring
from module.utils import grid_sample


if __name__ == '__main__':
    
    '''------config------'''
    
    path_i = '../Seg2Tunnel/seg2tunnel'
    # path_i = 'data'
    
    # flag_all = True
    flag_all = False
    if not flag_all:
        # part_stations = ['0-0', '0-103', '1-1', '3-1', '4-1']
        # part_stations = ['0-103', '1-1', '3-1', '4-1']
        # part_stations = ['0-0', '0-103']
        part_stations = ['0-0', '0-12', '0-16', '0-19', '0-20', '0-25', '0-76', '0-81', '0-89', '0-96', '0-98', '0-101', '0-103']
    
    voxel_size = 0.04
    max_num = 40960
    
    r_all = [2.75, 2.75, 2.75, 2.95, 3.75, 3.75]
    length_all = [1.2, 1.2, 1.2, 1.2, 1.8, 1.8]
    width_all = [0.35, 0.35, 0.35, 0.35, 0.4, 0.4]
    num_seg_all = [6, 6, 6, 6, 7, 7]
    
    angles_b_all = [[[9, -9], [35.375, -34.375], [33.75, -33.75], [33.75, -33.75], [33.75, -33.75], [34.375, -35.375]],
                    [[9.6, -9.6], [35.4, -33.75], [33.75, -33.75], [33.75, -33.75], [33.75, -33.75], [33.75, -35.4]],
                    [[9.6, -9.6], [35.4, -33.75], [33.75, -33.75], [33.75, -33.75], [33.75, -33.75], [33.75, -35.4]],
                    [[6.73, -6.73], [33.77, -32.5], [32.5, -32.5], [42, -42], [32.5, -32.5], [32.5, -33.77]],
                    [[7, -7], [30.89475, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -30.89475]],
                    [[7, -7], [30.89475, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -30.89475]]]
    for i in range(len(angles_b_all)):
        for j in range(len(angles_b_all[i])):
            for k in range(len(angles_b_all[i][j])):
                angles_b_all[i][j][k] = angles_b_all[i][j][k] / 180 * np.pi
    
    angles_m_all = [[[10, -10], [34.375, -34.375], [33.75, -33.75], [33.75, -33.75], [33.75, -33.75], [34.375, -34.375]],
                    [[11.25, -11.25], [33.75, -33.75], [33.75, -33.75], [33.75, -33.75], [33.75, -33.75], [33.75, -33.75]],
                    [[11.25, -11.25], [33.75, -33.75], [33.75, -33.75], [33.75, -33.75], [33.75, -33.75], [33.75, -33.75]],
                    [[8, -8], [32.5, -32.5], [32.5, -32.5], [42, -42], [32.5, -32.5], [32.5, -32.5]],
                    [[9.4737, -9.4737], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105]],
                    [[9.4737, -9.4737], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105]]]
    for i in range(len(angles_m_all)):
        for j in range(len(angles_m_all[i])):
            for k in range(len(angles_m_all[i][j])):
                angles_m_all[i][j][k] = angles_m_all[i][j][k] / 180 * np.pi
    
    angles_f_all = [[[11.5, -11.5], [32.875, -34.375], [33.75, -33.75], [33.75, -33.75], [33.75, -33.75], [34.375, -32.875]],
                    [[12.9, -12.9], [32.1, -33.75], [33.75, -33.75], [33.75, -33.75], [33.75, -33.75], [33.75, -32.1]],
                    [[12.9, -12.9], [32.1, -33.75], [33.75, -33.75], [33.75, -33.75], [33.75, -33.75], [33.75, -32.1]],
                    [[9.23, -9.23], [31.27, -32.5], [32.5, -32.5], [42, -42], [32.5, -32.5], [32.5, -31.27]],
                    [[11.7467, -11.7467], [26.14805, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -26.14805]],
                    [[11.7467, -11.7467], [26.14805, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -28.42105], [28.42105, -26.14805]]]
    for i in range(len(angles_f_all)):
        for j in range(len(angles_f_all[i])):
            for k in range(len(angles_f_all[i][j])):
                angles_f_all[i][j][k] = angles_f_all[i][j][k] / 180 * np.pi
    
    # angles_b_all = [[[-9.00, 9.00], [9.00, 78.75], [78.75, 146.25], [146.25, 213.75], [213.75, 281.25], [281.25, 351.00]],
    #                 [[-9.60, 9.60], [9.60, 78.75], [78.75, 146.25], [146.25, 213.75], [213.75, 281.25], [281.25, 350.40]],
    #                 [[-9.60, 9.60], [9.60, 78.75], [78.75, 146.25], [146.25, 213.75], [213.75, 281.25], [281.25, 350.40]],
    #                 [[-6.73, 6.73], [6.73, 73.00], [73.00, 138.00], [138.00, 222.00], [222.00, 287.00], [287.00, 353.27]],
    #                 [[-7.0000, 7.0000], [7.0000, 66.3158], [66.3158, 123.1579], [123.1579, 180.0000], [180.0000, 236.8421], [236.8421, 293.6842], [293.6842, 353.0000]],
    #                 [[-7.0000, 7.0000], [7.0000, 66.3158], [66.3158, 123.1579], [123.1579, 180.0000], [180.0000, 236.8421], [236.8421, 293.6842], [293.6842, 353.0000]]]
    # angles_m_all = [[[-10.00, 10.00], [10.00, 78.75], [78.75, 146.25], [146.25, 213.75], [213.75, 281.25], [281.25, 350.00]],
    #                 [[-11.25, 11.25], [11.25, 78.75], [78.75, 146.25], [146.25, 213.75], [213.75, 281.25], [281.25, 348.75]],
    #                 [[-11.25, 11.25], [11.25, 78.75], [78.75, 146.25], [146.25, 213.75], [213.75, 281.25], [281.25, 348.75]],
    #                 [[-8.00, 8.00], [8.00, 73.00], [73.00, 138.00], [138.00, 222.00], [222.00, 287.00], [287.00, 352.00]],
    #                 [[-9.4737, 9.4737], [9.4737, 66.3158], [66.3158, 123.1579], [123.1579, 180.0000], [180.0000, 236.8421], [236.8421, 293.6842], [293.6842, 350.5263]],
    #                 [[-9.4737, 9.4737], [9.4737, 66.3158], [66.3158, 123.1579], [123.1579, 180.0000], [180.0000, 236.8421], [236.8421, 293.6842], [293.6842, 350.5263]]]
    # angles_f_all = [[[-11.50, 11.50], [11.50, 78.75], [78.75, 146.25], [146.25, 213.75], [213.75, 281.25], [281.25, 348.50]],
    #                 [[-12.90, 12.90], [12.90, 78.75], [78.75, 146.25], [146.25, 213.75], [213.75, 281.25], [281.25, 347.10]],
    #                 [[-12.90, 12.90], [12.90, 78.75], [78.75, 146.25], [146.25, 213.75], [213.75, 281.25], [281.25, 347.10]],
    #                 [[-9.23, 9.23], [9.23, 73.00], [73.00, 138.00], [138.00, 222.00], [222.00, 287.00], [287.00, 350.77]],
    #                 [[-11.7467, 11.7467], [11.7467, 66.3158], [66.3158, 123.1579], [123.1579, 180.0000], [180.0000, 236.8421], [236.8421, 293.6842], [293.6842, 348.2533]],
    #                 [[-11.7467, 11.7467], [11.7467, 66.3158], [66.3158, 123.1579], [123.1579, 180.0000], [180.0000, 236.8421], [236.8421, 293.6842], [293.6842, 348.2533]]]
    
    flag_trans_yz = [True, False, False, False, False, False]
    flag_v0_dir = [[0, 1, 0], None, None, None, None, None]
    
    fmt = '%.8f %.8f %.8f %.8f %d'
    fmt +=  ' %.8f %.8f %.8f %.8f %.8f'
    fmt +=  ' %.8f %.8f %.8f %.8f %.8f'
    fmt +=  ' %.8f %.8f %.8f %.8f %.8f'
    fmt +=  ' %.8f %.8f %.8f %.8f %.8f'
    path_o = 'result'
    
    index_label = 4
    '''------config------'''
    
    if not os.path.exists(path_o):
        os.makedirs(path_o)

    files = os.listdir(path_i)
    
    stations = {}
    for file in files:
        station = file.rsplit('-', 1)[0]
        if not flag_all:
            if station not in part_stations:
                continue
        if station not in stations.keys():
            stations[station] = []
        stations[station].append(file)
    
    v0_dir_all = {}
    
    ovalisation_all = []
    dislocation_all_all = []
    rotation_all_all = []
    
    for station in stations.keys():
        
        tunnel_no = int(station.split('-')[0])
        
        if flag_v0_dir[tunnel_no] is not None:
            v0_dir_all[station] = np.asarray(flag_v0_dir[tunnel_no])
        else:
            ring0 = pd.read_csv(os.path.join(path_i, stations[station][0]), sep=' ', header=None)
            ring0 = np.asarray(ring0)
            ring0_no = int(stations[station][0].split('.')[0].split('-')[-1])
            ring1 = pd.read_csv(os.path.join(path_i, stations[station][1]), sep=' ', header=None)
            ring1 = np.asarray(ring1)
            ring1_no = int(stations[station][1].split('.')[0].split('-')[-1])
            mean_xyz_0 = np.mean(ring0[:, 0:3], axis=0)
            mean_xyz_1 = np.mean(ring1[:, 0:3], axis=0)
            v0_dir = mean_xyz_1 - mean_xyz_0
            if ring0_no > ring1_no:
                v0_dir = - v0_dir
            v0_dir_all[station] = v0_dir
    
        for file in stations[station]:
            
            pc = pd.read_csv(os.path.join(path_i, file), sep=' ', header=None)
            pc = np.asarray(pc)
            pc = pc[pc[:, index_label] != 0, :]
            if voxel_size != 0:
                pc = grid_sample(pc, voxel_size)
            if pc.shape[0] > max_num:
                np.random.shuffle(pc)
                pc = pc[0:max_num, :]
                
            if flag_trans_yz[tunnel_no]:
                pc[:, 0]= - pc[:, 0]
                pc[:, [1, 2]]= pc[:, [2, 1]]
            
            ring = Ring(pc[:, 0:3], pc[:, 3], pc[:, 4], r_all[tunnel_no], length_all[tunnel_no], width_all[tunnel_no],  num_seg_all[tunnel_no], angles_b_all[tunnel_no], angles_m_all[tunnel_no], angles_f_all[tunnel_no], v0_dir_all[station])
            
            xyz_p, d, error = ring.compute_d_circle()
            pc = np.hstack((pc, xyz_p, d, error))
            xyz_p, d, error, ovalisation = ring.compute_d_ellipse()
            pc = np.hstack((pc, xyz_p, d, error))
            ovalisation_all.append([file.split('.')[0], ovalisation[0], ovalisation[1], ovalisation[2]])
            xyz_p, d, error = ring.compute_d_seg_circle()
            pc = np.hstack((pc, xyz_p, d, error))
            xyz_p, d, error, dislocation_all, rotation_all = ring.compute_d_seg_lin()
            pc = np.hstack((pc, xyz_p, d, error))
            dislocation_all_all.append([int(file.split('.')[0].split('-')[0]), int(file.split('.')[0].split('-')[1]), int(file.split('.')[0].split('-')[2])])
            rotation_all_all.append([int(file.split('.')[0].split('-')[0]), int(file.split('.')[0].split('-')[1]), int(file.split('.')[0].split('-')[2])])
            for i in range(num_seg_all[tunnel_no]):
                dislocation_all_all[-1].append(dislocation_all[i])
                rotation_all_all[-1].append(rotation_all[i])
            
            if flag_trans_yz[tunnel_no]:
                pc[:, 0]= - pc[:, 0]
                pc[:, [1, 2]]= pc[:, [2, 1]]
            
            np.savetxt(os.path.join(path_o, file), pc, fmt=fmt)
            
    ovalisation_all = pd.DataFrame(ovalisation_all)
    ovalisation_all.to_excel(os.path.join(path_o, 'ovalisation.xlsx'), header=False, index=False)
    dislocation_all_all = pd.DataFrame(dislocation_all_all)
    dislocation_all_all.to_excel(os.path.join(path_o, 'dislocation.xlsx'), header=False, index=False)
    rotation_all_all = pd.DataFrame(rotation_all_all)
    rotation_all_all.to_excel(os.path.join(path_o, 'rotation.xlsx'), header=False, index=False)
