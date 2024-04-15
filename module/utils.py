import copy
import numpy as np
import open3d as o3d
import random as rd


def compute_dis2axis_2d(param, xy):
    
    a, b, c = param[0], param[1], param[2]
    d = np.sqrt((a * xy[:, 0] + b * xy[:, 1] + c) ** 2 / (a ** 2 + b ** 2))
    
    return d


def compute_normal(xyz, num_search):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(num_search))
    normal = np.asarray(pcd.normals)
    
    return normal


def fit_circle(param, xy_p, r):
    
    xy_o = param
    
    d = np.linalg.norm(xy_p - xy_o, axis=1) - r
    
    return d


def fit_circle_v(param, xyz, r):
    
    xy_o = param[0:2]
    v = param[2:]
    
    xyz_p = project2plane(xyz, v)
    xy_p = xyz_p[:, 0:2]
    
    d = np.linalg.norm(xy_p - xy_o, axis=1) - r
    
    return d


def fit_ellipse(param, xy_p):
    
    f_delta = param[0:2]
    r_ellipse = param[2]
    
    f1 = - f_delta
    f2 = f_delta
    
    d = np.linalg.norm(xy_p - f1, axis=1) + np.linalg.norm(xy_p - f2, axis=1) - 2 * r_ellipse
    # d = np.sqrt(np.sum(np.square(d))/d.shape[0])
    
    return d


def fit_ellipse_v(param, xyz):
    
    xy_o = param[0:2]
    v = param[2:5]
    f_delta = param[5:7]
    r_ellipse = param[7]
    
    f1 = xy_o - f_delta
    f2 = xy_o + f_delta
    
    xyz_p = project2plane(xyz, v)
    xy_p = xyz_p[:, 0:2]

    d = np.linalg.norm(xy_p - f1, axis=1) + np.linalg.norm(xy_p - f2, axis=1) - 2 * r_ellipse
    
    return d


def fit_fourier(param, xy_p, k):
    
    param_0 = param[2 * k]
    param_1 = param[0:k]
    param_2 = param[k:2 * k]
    
    d = np.zeros(xy_p.shape[0])
    d[:] = param_0
    
    theta_xy_p = np.arctan2(xy_p[:, 1], xy_p[:, 0])
    for i in range(k):
        d += param_1[i] * np.cos((i + 1) * theta_xy_p) + param_2[i] * np.sin((i + 1) * theta_xy_p)
    
    d = np.linalg.norm(xy_p, axis=1) - d
    
    return d


def grid_sample(points, voxel_size):
    
    features = points[:, 3:]
    points = points[:, 0:3]

    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    idx_pts_vox_sorted = np.argsort(inverse)
    voxel_grid={}
    voxel_grid_f={}
    sub_points, sub_features = [], []
    last_seen=0

    for idx, vox in enumerate(non_empty_voxel_keys):
        voxel_grid[tuple(vox)] = points[idx_pts_vox_sorted[last_seen: last_seen+nb_pts_per_voxel[idx]]]
        voxel_grid_f[tuple(vox)] = features[idx_pts_vox_sorted[last_seen: last_seen+nb_pts_per_voxel[idx]]]
        sub_points.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)], axis=0), axis=1).argmin()])
        sub_features.append(voxel_grid_f[tuple(vox)][np.linalg.norm(voxel_grid_f[tuple(vox)] - np.mean(voxel_grid_f[tuple(vox)], axis=0), axis=1).argmin()])
        last_seen += nb_pts_per_voxel[idx]
        
    sub_points = np.hstack((np.asarray(sub_points), np.asarray(sub_features)))

    return sub_points


def norm_intensity(intensity):
    
    bottom, up = np.percentile(intensity, 1), np.percentile(intensity, 99)
    intensity[intensity < bottom] = bottom
    intensity[intensity > up] = up
    intensity -= bottom
    intensity = intensity / (up - bottom)
    
    return intensity


def project2axis_2d(xy, param):
    
    a, b, c = param[0], param[1], param[2]
    x0, y0 = xy[:, 0], xy[:, 1]
    x = x0 - a * (a * x0 + b * y0 + c) / (a ** 2 + b ** 2)
    y = y0 - b * (a * x0 + b * y0 + c) / (a ** 2 + b ** 2)
    d = x / abs(x) * np.sqrt(x ** 2 + (y + c / b) ** 2)
    
    return d


def project2plane(xyz, v):
    
    v = v / np.linalg.norm(v)
    
    v_left = np.cross(v, np.array([0, 0, -1]))
    v_left = v_left / np.linalg.norm(v_left)
    v_up = np.cross(v, v_left)
    v_up = v_up / np.linalg.norm(v_up)
    
    xyz_p = np.zeros((xyz.shape[0], 3))
    
    v_p = xyz - np.dot(xyz, v).reshape(-1, 1) * v.reshape(1, 3)
    xyz_p[:, 0] = np.dot(v_p, v_left)
    xyz_p[:, 1] = np.dot(v_p, v_up)
    xyz_p[:, 2] = np.dot(xyz, v)
        
    return xyz_p


def rotate_xz(pc, ang):
    
    xz = copy.deepcopy(pc[:, [0, 2]])
    mat = [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]
    mat = np.asarray(mat)
    xz = np.dot(mat, xz.T).T
    
    new_pc = copy.deepcopy(pc)
    new_pc[:, 0] = xz[:, 0]
    new_pc[:, 2] = xz[:, 1]
    
    return new_pc


def rotate_xy(pc, ang):
    
    xy = copy.deepcopy(pc[:, 0:2])
    mat = [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]
    mat = np.asarray(mat)
    xy = np.dot(mat, xy.T).T
    
    new_pc = copy.deepcopy(pc)
    new_pc[:, 0:2] = xy[:, :]
    
    return new_pc


def rotate_xyz_random(pc):
    
    ang = rd.uniform(0, 360)/180*np.pi
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    pc[:, 0:3] = np.dot(R, pc[:, 0:3].T).T
    ang = rd.uniform(0, 360)/180*np.pi
    R = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])
    pc[:, 0:3] = np.dot(R, pc[:, 0:3].T).T
    ang = rd.uniform(0, 360)/180*np.pi
    R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    pc[:, 0:3] = np.dot(R, pc[:, 0:3].T).T
    
    return pc


def solve_contradiction(matrix):
    
    e_vals, e_vecs = np.linalg.eig(np.dot(matrix.T, matrix))
    vec = e_vecs[:, np.argmin(e_vals)]
    vec = vec / np.linalg.norm(vec)
    
    return vec