import numpy as np
import os
import random as rd


def crop_random(pc, sca_crop, ratio_crop):
    
    sca_crop = rd.uniform(0, 1) * sca_crop
    
    centre = pc[rd.randint(0, pc.shape[0] - 1), 0:3]
    d = np.linalg.norm(pc[:, 0:3] - centre.reshape(1, 3), axis=1)
    pc = np.hstack((pc, d.reshape(-1, 1)))
    pc = pc[pc[:, -1] > sca_crop, 0:-1]
    
    return pc


def generate_jitter(pc, sca_jit):
        
    jit = np.random.random((pc.shape[0], 3)) * sca_jit
    pc[:, 0:3] = pc[:, 0:3] + jit
    
    return pc


def generate_noise(res_noise, sca_noise, xyz):
    
    pc_noise = []
    
    sca_noise = rd.uniform(0, 1) * sca_noise
    
    for x in np.arange(- sca_noise, sca_noise, res_noise):
        for y in np.arange(- sca_noise, sca_noise, res_noise):
            pc_noise.append([x, y, 0, 0, 0])
    pc_noise = np.asarray(pc_noise)
    
    pc_noise = rotate_random_3d(pc_noise)
    pc_noise[:, 0:3] = pc_noise[:, 0:3] + xyz.reshape(1, 3)
    
    return pc_noise


def generate_tunnel(res, a, b, length, pav_z, cel_z, res_noise, sca_noise, ratio_noise_lin, ratio_noise_pav, sca_crop, ratio_crop_lin, ratio_crop_pav, sca_jit):
    
    pc_lin = []
    for ang in np.arange(0, 2 * np.pi, 2 * res / (a + b)):
        x = a * np.cos(ang)
        z = b * np.sin(ang)
        if z >= pav_z and z <= cel_z:
            for y in np.arange(- length / 2, length / 2, res):
                pc_lin.append([x, y, z, 0, 1])
    pc_lin = np.asarray(pc_lin)
    
    pc_pav = []
    pav_wid_hal = np.sqrt(a**2 * (1 - pav_z**2 / b**2))
    z = pav_z
    for x in np.arange(- pav_wid_hal, pav_wid_hal, res):
        for y in np.arange(- length / 2, length / 2, res):
            pc_pav.append([x, y, z, 0, 2])
    pc_pav = np.asarray(pc_pav)    
    
    num = pc_lin.shape[0]
    while pc_lin.shape[0] < num * (1 + ratio_noise_lin):
        pc_lin = np.vstack((pc_lin, generate_noise(res_noise, sca_noise, pc_lin[rd.randint(0, pc_lin.shape[0] - 1), 0:3])))
        
    num = pc_pav.shape[0]
    while pc_pav.shape[0] < num * (1 + ratio_noise_pav):
        pc_pav = np.vstack((pc_pav, generate_noise(res_noise, sca_noise, pc_pav[rd.randint(0, pc_pav.shape[0] - 1), 0:3])))
    
    num = pc_lin.shape[0]
    while pc_lin.shape[0] > num * (1 - ratio_crop_lin):
        pc_lin = crop_random(pc_lin, sca_crop, ratio_crop_lin)
        
    num = pc_pav.shape[0]
    while pc_pav.shape[0] > num * (1 - ratio_crop_pav):
        pc_pav = crop_random(pc_pav, sca_crop, ratio_crop_pav)
    
    pc = np.vstack((pc_lin ,pc_pav))
    
    pc = generate_jitter(pc, sca_jit)
    
    return pc


def rotate_random_3d(pc):
    
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


if __name__ == '__main__':
    
    '''config'''
    path_o = 'result'
    fns = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '1-11', '1-12', '1-13', '1-14', '1-15', '1-16', '1-17', '1-18', '1-19', '1-20']
    
    res = 0.01
    
    a = 7.421
    b = 5.18
    length = 5
    pav_z = - 2.8
    cel_z = 2.8
    
    res_noise = 5 * res
    sca_noise = 0.5
    ratio_noise_lin = 0.05
    ratio_noise_pav = 0.01
    
    sca_crop = 1
    ratio_crop_lin = 0.5
    ratio_crop_pav = 0.1
    
    sca_jit = res
    
    sca_ran = 0.1
    '''config'''
    
    '''call'''
    for fn in fns:
        aa = rd.uniform(b, a)
        bb = rd.uniform(b, aa)
        
        pc = generate_tunnel(res, aa, bb, length, pav_z, cel_z, res_noise, sca_noise, ratio_noise_lin, ratio_noise_pav, sca_crop, ratio_crop_lin, ratio_crop_pav, sca_jit)
        
        sca_ran_ran = rd.uniform(1 - sca_ran, 1 + sca_ran)
        pc[:, 0:3] = pc[:, 0:3] * sca_ran_ran
        
        np.savetxt(os.path.join(path_o, fn + '.txt'), pc, fmt='%.8f %.8f %.8f %.8f %d')
    '''call'''
    
    