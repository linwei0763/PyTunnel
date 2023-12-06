import numpy as np
from scipy import optimize
from sklearn.neighbors import KDTree


class Tunnel():
    
    def __init__(self, xyz, intensity, label):
        
        self.offset = np.mean(xyz, axis=0)
        
        self.xyz = xyz - self.offset
        self.intensity = intensity
        self.label = label
        
        self.local_tree = KDTree(self.xyz)
        
        pini = np.asarray([1, 1, 0], dtype=float)
        plsq = optimize.leastsq(self.compute_dis2axis, pini, args=(self.xyz))
        self.axis_param = plsq[0]
        self.axis_d = np.zeros((self.xyz.shape[0], 3))
        self.axis_d[:, 0] = self.project2axis(self.xyz, self.axis_param)[:]
        self.ring_tree = KDTree(self.axis_d)
    
    def find_local_neighbour(self, centre_xyz, k_n):
        
        centre_xyz = centre_xyz - self.offset
        _, neigh_idx = self.local_tree.query([centre_xyz], k=k_n)
        neigh_idx = neigh_idx[0]
        
        return neigh_idx
    
    def find_ring_neighbour(self, centre_xyz, k_n):
        
        centre_xyz = centre_xyz - self.offset        
        centre_axis_d = np.zeros(3)
        centre_axis_d[0] = self.project2axis(centre_xyz.reshape(-1, 3), self.axis_param)
        _, neigh_idx = self.ring_tree.query([centre_axis_d], k=k_n)
        neigh_idx = neigh_idx[0]
        
        return neigh_idx
        
    @staticmethod
    def compute_dis2axis(p, xy):
        
        a, b, c = p[0], p[1], p[2]
        d2 = (a * xy[:, 0] + b * xy[:, 1] + c) ** 2 / (a ** 2 + b ** 2)
        
        return d2
    
    @staticmethod
    def project2axis(xy, axis_param):
        
        a, b, c = axis_param[0], axis_param[1], axis_param[2]
        x0, y0 = xy[:, 0], xy[:, 1]
        x = x0 - a * (a * x0 + b * y0 + c) / (a ** 2 + b ** 2)
        y = y0 - b * (a * x0 + b * y0 + c) / (a ** 2 + b ** 2)
        d = x / abs(x) * np.sqrt(x ** 2 + (y + c / b) ** 2)
        
        return d