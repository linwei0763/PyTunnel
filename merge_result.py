import numpy as np
import os
import torch


if __name__ == '__main__':
    
    '''------config------'''
    path_pc = 'input/seg2tunnel_pointcept_0.04/validation_set'
    path_pred = 'input/test_epoch89'
    path_o = 'result'
    str_fmt = '%.8f ' * 6 + '%d ' * 2
    '''------config------'''
    
    if not os.path.exists(path_o):
        os.makedirs(path_o)
    
    stations = os.listdir(path_pc)
    for station in stations:
        station = station.split('.')[0]
        pc = torch.load(os.path.join(path_pc, station + '.pth'), weights_only=False)
        coord = np.asarray(pc['coord'])
        color = np.asarray(pc['color'])
        semantic_gt = pc['semantic_gt'].reshape((-1, 1))
        pred = np.load(os.path.join(path_pred, station + '_pred.npy')).reshape(-1, 1)
        pc = np.hstack((coord, color, semantic_gt, pred,))
        np.savetxt(os.path.join(path_o, station + '.txt'), pc, fmt=str_fmt)