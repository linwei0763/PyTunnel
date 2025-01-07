import numpy as np
import os
import pandas as pd
from scipy import optimize

from module.utils import fit_ellipse_v, fit_fourier, grid_sample, project2plane

path_i = 'data'
path_o = 'result'
file = 'pc.txt'
voxel_size = 0.04
r_estimated = 4
k_fourier = 16

pc = pd.read_csv(os.path.join(path_i, file), sep=' ', header=None)
pc = np.asarray(pc)

if voxel_size != 0:
    pc = grid_sample(pc, voxel_size)
xyz = pc[:, 0:3]
mean_xyz = np.mean(xyz, axis=0)
xyz = xyz - mean_xyz

param = np.zeros(8)
param[2:5] = np.asarray([0, 1, 0])
param[7] = r_estimated
param_ls = optimize.least_squares(fit_ellipse_v, param, args=(xyz,))
param_ls = param_ls.x
xy_o = param_ls[0:2]
v = param_ls[2:5]
r_estimated = param_ls[-1]

xyz_p = project2plane(xyz, v)
xyz_p[:, 0:2] -= xy_o
mid_z_p = (np.max(xyz_p[:, 2]) + np.min(xyz_p[:, 2])) / 2
xyz_p[:, 2] -= mid_z_p

param = np.zeros(2 * k_fourier + 1)
param[0] = r_estimated
param_ls = optimize.least_squares(fit_fourier, param, loss='soft_l1', f_scale=0.001, args=(k_fourier, xyz_p[:, 0:2], r_estimated, 0))
param_ls = param_ls.x
param_ls[-1] = r_estimated
param_fourier = param_ls

theta_p = np.arctan2(xyz_p[:, 1], xyz_p[:, 0])

d = np.zeros(xyz_p.shape[0])
d[:] = param_fourier[-1]
for i in range(k_fourier):
    d[:] += param_fourier[i] * np.cos((i + 1) * theta_p) + param_ls[k_fourier + i] * np.sin((i + 1) * theta_p)
d = np.linalg.norm(xyz_p[:, 0:2], axis=1) - d
d = d.reshape(-1, 1)

pc = np.hstack((pc, d))
if not os.path.exists(path_o):
    os.makedirs(path_o)
fmt = '%.8f'
for _ in range(pc.shape[1] - 1):
    fmt +=  ' %.8f'
np.savetxt(os.path.join(path_o, 'pc.txt'), pc, fmt=fmt)
