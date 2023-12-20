import copy
import numpy as np
import random as rd


def generate_ring(len_seg, r, wid_joi, dep_joi, res, res_joi):
    
    ang_joi = wid_joi / r
    
    ring = []
    
    for y in np.arange(-len_seg / 2, -len_seg / 2 + wid_joi / 2, res_joi):
        rr = r + dep_joi * ((-len_seg / 2 + wid_joi / 2) - y) / (wid_joi / 2)
        for ang in np.arange(0, 2 * np.pi, res_joi / r):
            p = [rr * np.sin(ang), y, rr * np.cos(ang), 0, 0]
            ring.append(p)
            
    for y in np.arange(len_seg / 2 - wid_joi / 2, len_seg / 2, res_joi):
        rr = r + dep_joi * (y - (len_seg / 2 - wid_joi / 2)) / (wid_joi / 2)
        for ang in np.arange(0, 2 * np.pi, res_joi / r):
            p = [rr * np.sin(ang), y, rr * np.cos(ang), 0, 0]
            ring.append(p)
    
    for y in np.arange(-len_seg / 2 + wid_joi / 2, len_seg / 2 - wid_joi / 2, res_joi):
        for i in range(6):
            if i == 0:
                ang_c = (-22.5 / 2 - 2.5 * y / (len_seg / 2 - wid_joi / 2)) / 180 * np.pi
            elif i == 5:
                ang_c = (22.5 / 2 + 2.5 * y / (len_seg / 2 - wid_joi / 2)) / 180 * np.pi
            else:
                ang_c = (22.5 / 2 + 67.5 * i) / 180 * np.pi
            ang_s = ang_c - ang_joi / 2
            ang_e = ang_c + ang_joi / 2
            for ang in np.arange(ang_s, ang_c, res_joi / r):
                rr = r + dep_joi * (ang - ang_s) / (ang_joi / 2)
                p = [rr * np.sin(ang), y, rr * np.cos(ang), 0, 0]
                ring.append(p)
            for ang in np.arange(ang_c, ang_e, res_joi / r):
                rr = r + dep_joi * (ang_e - ang) / (ang_joi / 2)
                p = [rr * np.sin(ang), y, rr * np.cos(ang), 0, 0]
                ring.append(p)
        
    for y in np.arange(-len_seg / 2 + wid_joi / 2, len_seg / 2 - wid_joi / 2, res):
        for i in range(7):
            if i == 0:
                ang_s = 0
                ang_e = (22.5 / 2 + 2.5 * y / (len_seg / 2 - wid_joi / 2)) / 180 * np.pi - ang_joi / 2
                for ang in np.arange(ang_s, ang_e, res / r):
                    p = [r * np.sin(ang), y, r * np.cos(ang), 0, 1]
                    ring.append(p)
            elif i == 1:
                ang_s = (22.5 / 2 + 2.5 * y / (len_seg / 2 - wid_joi / 2)) / 180 * np.pi + ang_joi / 2
                ang_e = (22.5 / 2 + 67.5) / 180 * np.pi - ang_joi / 2
                for ang in np.arange(ang_s, ang_e, res / r):
                    p = [r * np.sin(ang), y, r * np.cos(ang), 0, 6]
                    ring.append(p)
            elif i == 5:
                ang_s = (-22.5 / 2 - 67.5) / 180 * np.pi + ang_joi / 2
                ang_e = (-22.5 / 2 - 2.5 * y / (len_seg / 2 - wid_joi / 2)) / 180 * np.pi - ang_joi / 2
                for ang in np.arange(ang_s, ang_e, res / r):
                    p = [r * np.sin(ang), y, r * np.cos(ang), 0, 2]
                    ring.append(p)
            elif i == 6:
                ang_s = (-22.5 / 2 - 2.5 * y / (len_seg / 2 - wid_joi / 2)) / 180 * np.pi + ang_joi / 2
                ang_e = 0
                for ang in np.arange(ang_s, ang_e, res / r):
                    p = [r * np.sin(ang), y, r * np.cos(ang), 0, 1]
                    ring.append(p)
            else:
                ang_s = (22.5 / 2 + (i - 1) * 67.5) / 180 * np.pi + ang_joi / 2
                ang_e = (22.5 / 2 + i * 67.5) / 180 * np.pi - ang_joi / 2
                for ang in np.arange(ang_s, ang_e, res / r):
                    p = [r * np.sin(ang), y, r * np.cos(ang), 0, 7 - i]
                    ring.append(p)
    
    ring = np.asarray(ring)
    return ring


def generate_tunnel(len_seg, r, wid_joi, dep_joi, res, res_joi):
    
    pc = []
    ring = generate_ring(len_seg, r, wid_joi, dep_joi, res, res_joi)
    
    for i in np.arange(-10, 10):
        ang = rd.randint(0, 16) * 22.5 / 180 * np.pi
        new_ring = rotate_xz(ring, ang)
        new_ring[:, 1] += i * len_seg
        pc.append(new_ring)
    
    pc = np.vstack(pc[:])
    ang = rd.randint(0, 360) / 180 * np.pi
    pc = rotate_xy(pc, ang)
    np.random.shuffle(pc)
    
    return pc


def rotate_xz(ring, ang):
    
    xz = copy.deepcopy(ring[:, [0, 2]])
    mat = [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]
    mat = np.asarray(mat)
    xz = np.dot(mat, xz.T).T
    
    new_ring = copy.deepcopy(ring)
    new_ring[:, 0] = xz[:, 0]
    new_ring[:, 2] = xz[:, 1]
    
    return new_ring


def rotate_xy(pc, ang):
    
    xy = copy.deepcopy(pc[:, 0:2])
    mat = [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]
    mat = np.asarray(mat)
    xy = np.dot(mat, xy.T).T
    
    new_pc = copy.deepcopy(pc)
    new_pc[:, 0:2] = xy[:, :]
    
    return new_pc


if __name__ == '__main__':
    
    '''config'''
    fns = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9', '1-10', '1-11', '1-12', '1-13', '1-14', '1-15', '1-16', '1-17', '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12', '2-13', '2-14']
    
    len_seg = 1.5
    r = 2.75
    wid_joi = 0.008
    dep_joi = 0.022
    res = 0.0075
    res_joi = 0.002
    '''config'''
    
    '''call'''
    for fn in fns:
        pc = generate_tunnel(len_seg, r, wid_joi, dep_joi, res, res_joi)
        np.savetxt('result/' + fn + '.txt', pc, fmt='%.8f %.8f %.8f %.8f %d')
    '''call'''
    
    