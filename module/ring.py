import numpy as np
import open3d as o3d
from scipy import optimize


class Ring():
    
    def __init__(self, xyz, intensity, label, r, num_seg):
        
        self.offset = np.mean(xyz, axis=0)
        
        self.xyz = xyz - self.offset
        self.intensity = intensity
        self.label = label
        
        self.r = r
        self.num_seg = num_seg
        
        self.normal = Ring.compute_normal(xyz)
        self.v0 = Ring.compute_v0(self.normal)
        
        self.v = None
        self.xy0_ring = None
        self.ovalization = None
        self.d = None
    
    def compute_deformation(self):
        
        xyz = self.xyz
        label = self.label
        r = self.r
        num_seg = self.num_seg
        v0 = self.v0
        d = []

        param_ring = np.zeros(5)
        param_ring[2:] = v0[:]
        plsq_ring = optimize.leastsq(self.compute_dis2axi, param_ring, args=(xyz, r))
        plsq_ring = plsq_ring[0]
        xy0_ring = plsq_ring[0:2]
        v = plsq_ring[2:]
        
        d1 = self.compute_dis2axi(plsq_ring, xyz, r)
        d1 = d1.reshape(-1, 1)
        d.append(d1)
        
        param_ring_e = np.zeros(3)
        param_ring_e[0], param_ring_e[1], param_ring_e[2] = 0.01, 0, r
        plsq_ring_e = optimize.leastsq(self.fit_ellipse, param_ring_e, args=(xyz, r, xy0_ring, v))
        plsq_ring_e = plsq_ring_e[0]
        f_delta = plsq_ring_e[0:2]
        r_ellipse = plsq_ring_e[2]

        ovalization, d2 = self.compute_dis_fitted_ellipse(xyz, r, xy0_ring, v, f_delta, r_ellipse)
        d2 = d2.reshape(-1, 1)
        d.append(d2)
        
        if num_seg > 1:
            d3 = np.zeros(xyz.shape[0])
            for i in range(1, num_seg + 1):
                index = label[:] == i
                if index.any() == True:
                    xyz_seg = xyz[index, :]
                    plsq_seg = optimize.leastsq(self.fit_circle, xy0_ring, args=(xyz_seg, r, v))
                    plsq_seg = plsq_seg[0]
                    xy0_seg = plsq_seg
                    d3[index] = self.compute_dis_fitted_circle(xyz_seg, r, xy0_ring, v, xy0_seg)[:]
            d3 = d3.reshape(-1, 1)
            d.append(d3)
        
        d = np.hstack(d)
        
        self.v = v
        self.xy0_ring = xy0_ring
        self.ovalization = ovalization
        self.d = d

        return self.d
    
    @staticmethod
    def compute_dis_fitted_circle(xyz, r, xy0_ring, v, xy0_seg):
        
        xy_p = Ring.project2plane(xyz, v)
        xy_p_fitted = xy_p - xy0_seg.reshape(1, -1)
        xy_p_fitted = xy_p_fitted / np.linalg.norm(xy_p_fitted, axis=1).reshape(-1, 1) * r
        xy_p_fitted = xy_p_fitted + xy0_seg.reshape(1, -1)
        
        d = xy_p_fitted - xy0_ring
        d = np.linalg.norm(d, axis=1) - r
        
        return d
    
    @staticmethod
    def compute_dis_fitted_ellipse(xyz, r, xy0_ring, v, f_delta, r_ellipse):
        
        xy_p = Ring.project2plane(xyz, v)
        xy_p = xy_p - xy0_ring.reshape(1, -1)
        xy_p = xy_p / np.linalg.norm(xy_p, axis=1).reshape(-1, 1)
        
        ang = np.arcsin(xy_p[:, 1])
        index = xy_p[:, 0] <= 0
        ang[index] = np.pi - ang[index]
        
        a = r_ellipse
        c = np.linalg.norm(f_delta)
        b = np.sqrt(a ** 2 - c ** 2)
        ang_delta = np.arcsin(f_delta[1] / c)
        
        ovalization = np.asarray([a, b, ang_delta])
        
        ang = ang - ang_delta
        
        x_p_fitted = a * np.cos(ang) * np.cos(ang_delta) - b * np.sin(ang) * np.sin(ang_delta)
        y_p_fitted = b * np.sin(ang) * np.cos(ang_delta) + a * np.cos(ang) * np.sin(ang_delta)
        
        xy_p_fitted = np.hstack((x_p_fitted.reshape(-1, 1), y_p_fitted.reshape(-1, 1)))
        
        d = np.linalg.norm(xy_p_fitted, axis=1) - r
        
        return ovalization, d
    
    @staticmethod
    def compute_dis2axi(param, xyz, r):
        
        xy0 = param[0:2]
        v0 = param[2:]
        
        xy_p = Ring.project2plane(xyz, v0)
        d = np.linalg.norm(xy_p - xy0, axis=1) - r
        
        return d
    
    @staticmethod
    def compute_normal(xyz):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(1000, 5))
        normal = np.asarray(pcd.normals)
        return normal
    
    @staticmethod
    def compute_v0(normal):
        e_vals, e_vecs = np.linalg.eig(np.dot(normal.T, normal))
        v0 = e_vecs[:, np.argmin(e_vals)]
        v0 = v0 / np.linalg.norm(v0)
        return v0    
    
    @staticmethod
    def fit_circle(param, xyz, r, v):
        
        xy0 = param[0:2]
        
        xy_p = Ring.project2plane(xyz, v)
        d = np.linalg.norm(xy_p - xy0, axis=1) - r
        
        return d
    
    @staticmethod
    def fit_ellipse(param, xyz, r, xy0_ring, v):
        
        f_delta = param[0:2]
        r_ellipse = param[2]
        
        f1 = xy0_ring - f_delta
        f2 = xy0_ring + f_delta
        
        xy_p = Ring.project2plane(xyz, v)

        d = np.linalg.norm(xy_p - f1, axis=1) + np.linalg.norm(xy_p - f2, axis=1) - 2 * r_ellipse
        
        return d
    
    @staticmethod
    def project2plane(xyz, v):
        
        v_left = np.cross(v, np.array([0, 0, -1]))
        v_left = v_left / np.linalg.norm(v_left)
        v_up = np.cross(v, v_left)
        v_up = v_up / np.linalg.norm(v_up)
        
        xy_p = np.zeros((xyz.shape[0], 2))
        
        v_p = xyz - np.dot(xyz, v).reshape(-1, 1) * v.reshape(1, 3)
        xy_p[:, 0] = np.dot(v_p, v_left)
        xy_p[:, 1] = np.dot(v_p, v_up)
            
        return xy_p