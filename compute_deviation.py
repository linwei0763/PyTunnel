import copy
import numpy as np
import os
import pandas as pd
from scipy import optimize

from module.utils import fit_ellipse_v, fit_fourier, grid_sample, rotate_xy, project2plane


if __name__ == '__main__':
    '''-----config-----'''
    path_i = 'data'
    path_o = 'result'
    file = 'pc.txt'
    
    voxel_size = 0.01
    v_estimated = [1, 0, 0]
    r = 1.59
    
    flag = 'circle'
    # flag = 'fourier'
    
    if flag == 'circle':
        pass
    elif flag == 'fourier':
        k_fourier = 16
    '''-----config-----'''
    
    pc = pd.read_csv(os.path.join(path_i, file), sep=' ', header=None)
    pc = np.asarray(pc)
    
    if voxel_size != 0:
        pc = grid_sample(pc, voxel_size)
    xyz = pc[:, 0:3]
    mean_xyz = np.mean(xyz, axis=0)
    xyz = xyz - mean_xyz
    
    param = np.zeros(8)
    param[2:5] = np.asarray(v_estimated)
    param[7] = r
    param_ls = optimize.least_squares(fit_ellipse_v, param, args=(xyz,)).x
    xy_o = param_ls[0:2]
    v = param_ls[2:5]
    f_delta = param_ls[5:7]
    r_ellipse = param_ls[7]
    if np.dot(v, v_estimated) < 0:
        v = - v
        xy_o[0] = - xy_o[0]
        f_delta[0] = - f_delta[0]
    if f_delta[0] < 0:
        f_delta = - f_delta
    a = r_ellipse
    c = np.linalg.norm(f_delta)
    b = np.sqrt(a ** 2 - c ** 2)
    theta_ellipse = np.arctan2(f_delta[1], f_delta[0])
    ovalisation = np.asarray([a, b, theta_ellipse])
    
    xy_p_ellipse = []
    for theta_per in np.arange(0, 2 * np.pi, step=np.pi/1800):
        theta_per = theta_per - theta_ellipse
        xy_per = np.asarray([a * np.cos(theta_per), b * np.sin(theta_per)]).reshape(1, 2)
        xy_per = rotate_xy(xy_per, theta_ellipse).reshape(2)
        xy_p_ellipse.append([xy_per[0], xy_per[1]])
    xy_p_ellipse = np.asarray(xy_p_ellipse)
    xy_p_ellipse_10 = copy.deepcopy(xy_p_ellipse)
    d_ellipse_10 = np.linalg.norm(xy_p_ellipse_10[:, 0:2], axis=1)
    xy_p_ellipse_10[:, 0] = ((d_ellipse_10 - r) * 10 + r) * xy_p_ellipse[:, 0] / d_ellipse_10
    xy_p_ellipse_10[:, 1] = ((d_ellipse_10 - r) * 10 + r) * xy_p_ellipse[:, 1] / d_ellipse_10
    
    xyz_p = project2plane(xyz, v)
    xyz_p[:, 0:2] -= xy_o
    mid_z_p = (np.max(xyz_p[:, 2]) + np.min(xyz_p[:, 2])) / 2
    xyz_p[:, 2] -= mid_z_p
    theta_p = np.arctan2(xyz_p[:, 1], xyz_p[:, 0])
    d = np.zeros(xyz_p.shape[0])
    if flag == 'circle':
        d = np.linalg.norm(xyz_p[:, 0:2], axis=1) - r
    elif flag == 'fourier':
        param = np.zeros(2 * k_fourier + 1)
        param[-1] = r_ellipse
        param_ls = optimize.least_squares(fit_fourier, param, loss='soft_l1', f_scale=0.001, args=(k_fourier, xyz_p[:, 0:2], r_ellipse, 0)).x
        param_ls[-1] = r_ellipse
        d[:] = param_ls[-1]
        for i in range(k_fourier):
            d[:] += param_ls[i] * np.cos((i + 1) * theta_p) + param_ls[k_fourier + i] * np.sin((i + 1) * theta_p)
        d = np.linalg.norm(xyz_p[:, 0:2], axis=1) - d
    xy_p_10 = copy.deepcopy(xyz_p[:, 0:2])
    d_10 = np.linalg.norm(xy_p_10[:, 0:2], axis=1)
    xy_p_10[:, 0] = ((d_10 - r) * 10 + r) * xy_p_10[:, 0] / d_10
    xy_p_10[:, 1] = ((d_10 - r) * 10 + r) * xy_p_10[:, 1] / d_10
    
    ovalisation = pd.DataFrame(ovalisation)
    ovalisation.to_excel(os.path.join(path_o, 'ovalisation.xlsx'), header=False, index=False)
    xy_p_ellipse_10 = pd.DataFrame(xy_p_ellipse_10)
    xy_p_ellipse_10.to_excel(os.path.join(path_o, 'xy_p_ellipse_10.xlsx'), header=False, index=False)
    xy_p_10 = pd.DataFrame(xy_p_10)
    xy_p_10.to_excel(os.path.join(path_o, 'xy_p_10.xlsx'), header=False, index=False)
    
    d = d.reshape(-1, 1)
    pc = np.hstack((pc, d))
    if not os.path.exists(path_o):
        os.makedirs(path_o)
    fmt = '%.8f'
    for _ in range(pc.shape[1] - 1):
        fmt +=  ' %.8f'
    np.savetxt(os.path.join(path_o, file), pc, fmt=fmt)
    
