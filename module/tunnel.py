import numpy as np
import open3d as o3d
from scipy import optimize


class Tunnel():
    
    def __init__(self, xyz, intensity, label):
        
        self.offset = np.mean(xyz, axis=0)
        
        self.xyz = xyz - self.offset
        self.intensity = intensity
        self.label = label
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        self.local_tree = pcd_tree
        
        pini = np.asarray([1, 1, 0], dtype=float)
        plsq = optimize.leastsq(self.compute_dis2axis, pini, args=(self.xyz))
        self.axis_param = plsq[0]
        self.axis_d = np.zeros((self.xyz.shape[0], 3))
        self.axis_d[:, 0] = self.project2axis(self.xyz, self.axis_param)[:]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.axis_d)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        self.ring_tree = pcd_tree
    
    def find_local_neighbour(self, centre_xyz, k_n):
        
        [_, neigh_idx, _] = self.tree.search_knn_vector_3d(centre_xyz, k_n)
        
        return neigh_idx
    
    def find_ring_neighbour(self, centre_xyz, k_n):
        
        centre_axis_d = np.zeros(3)
        centre_axis_d = self.project2axis(self.centre_xyz, self.axis_param)
        [_, neigh_idx, _] = self.tree.search_knn_vector_3d(centre_axis_d, k_n)
        
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