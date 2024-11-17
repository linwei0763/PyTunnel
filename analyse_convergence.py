import fnmatch
import numpy as np
import os
import pandas as pd


if __name__ == '__main__':

    '''config'''
    file_gt = 'gt_convergence.xlsx'
    path_o = 'result'
    file_o = 'convergence_comparison.xlsx'
    stations = [0]
    ref_ring_no = ['0-0-2']
    r_all = [2.75]
    angles = [22.5, 112.5, 180, 157.5, 135, 90, 45]
    '''config'''
    
    gt = {}
    data = pd.read_excel(file_gt, header=None, names=None)
    data = np.asarray(data)
    for i in range(data.shape[0]):
        ring_no = str(int(data[i, 0])) + '-' + str(int(data[i, 1])) + '-' + str(int(data[i, 2]))
        gt[ring_no] = data[i, 3:]
    
    convergence = {}
    for i in range(len(stations)):
        station = stations[i]
        files = fnmatch.filter(os.listdir(path_o), str(station) + '-*-ellipse-polynomial.xlsx')
        r = r_all[i]
        for file in files:
            ring_no = file.split('-')[0] + '-' + file.split('-')[1] + '-' + file.split('-')[2]
            data = pd.read_excel(os.path.join(path_o, file), header=None, names=None)
            data = np.asarray(data)
            data = data[:, 0:2]
            polar = np.arctan2(data[:, 1], data[:, 0]) / np.pi * 180
            temp = []
            for angle in angles:
                index = (np.abs(polar - angle)).argmin()
                c = np.linalg.norm(data[index, :]) - r
                index = (np.abs(polar - (angle - 180))).argmin()
                c += np.linalg.norm(data[index, :]) - r
                temp.append(c*1000)
            convergence[ring_no] = np.asarray(temp)
        for key in convergence.keys():
            if key == ref_ring_no[i]:
                continue
            convergence[key] = convergence[key] - convergence[ref_ring_no[i]]
    
    for i in range(len(stations)):
        gt.pop(ref_ring_no[i], None)
        convergence.pop(ref_ring_no[i], None)
    
    data_all = []
    error = []
    for key in gt.keys():
        if key not in convergence.keys():
            continue
        data_all.append([int(key.split('-')[0]), int(key.split('-')[1]), int(key.split('-')[2])])
        for i in range(gt[key].shape[0]):
            data_all[-1].append(gt[key][i])
        for i in range(convergence[key].shape[0]):
            data_all[-1].append(convergence[key][i])
        error.append(gt[key] - convergence[key])
    
    error = np.asarray(error)
    mae = np.mean(np.abs(error))
    print('The MAE of %d convergence data is %.1f mm.' % (error.size, mae))
    
    data_all = np.asarray(data_all)
    data_all = pd.DataFrame(data_all)
    data_all.to_excel(os.path.join(path_o, file_o), header=False, index=False)
    
    