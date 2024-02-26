import numpy as np
from scipy import optimize
from sklearn.neighbors import KDTree

from utils import compute_dis2axis_2d, project2axis_2d


class Tunnel():
    
    def __init__(self, xyz, intensity, label):
        
        self.offset = np.mean(xyz, axis=0)
        
        self.xyz = xyz - self.offset
        self.intensity = intensity
        self.label = label
        
        self.local_tree = KDTree(self.xyz)
        
        param_ini = np.asarray([1, 1, 0], dtype=float)
        plsq = optimize.leastsq(compute_dis2axis_2d, param_ini, args=(self.xyz))
        self.axis_param = plsq[0]
        self.axis_d = np.zeros((self.xyz.shape[0], 3))
        self.axis_d[:, 0] = project2axis_2d(self.xyz, self.axis_param)[:]
        self.ring_tree = KDTree(self.axis_d)
    
    def find_local_neighbour(self, centre_xyz, k_n):
        
        centre_xyz = centre_xyz - self.offset
        _, neigh_idx = self.local_tree.query([centre_xyz], k=k_n)
        neigh_idx = neigh_idx[0]
        
        return neigh_idx
    
    def find_ring_neighbour(self, centre_xyz, k_n):
        
        centre_xyz = centre_xyz - self.offset        
        centre_axis_d = np.zeros(3)
        centre_axis_d[0] = project2axis_2d(centre_xyz.reshape(-1, 3), self.axis_param)
        _, neigh_idx = self.ring_tree.query([centre_axis_d], k=k_n)
        neigh_idx = neigh_idx[0]
        
        return neigh_idx
    
