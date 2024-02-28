import numpy as np
from scipy import optimize

from module.utils import compute_normal, fit_circle_v, fit_ellipse_v, project2plane, rotate_xy


class Ring():
    
    def __init__(self, xyz, intensity, label, r, length, num_seg, angles, angles_key, v0_dir):
        
        self.offset = np.mean(xyz, axis=0)
        
        self.xyz = xyz - self.offset
        self.intensity = intensity
        self.label = label
        
        self.r = r
        self.length = length
        self.num_seg = num_seg
        self.angles = angles
        self.angles_key = angles_key
        
        normal = compute_normal(xyz, 16)
        self.v0 = Ring.compute_v0(normal)
        if np.dot(self.v0, v0_dir) < 0:
            self.v0 = - self.v0
    
    def compute_d_circle(self):
        
        param = np.zeros(5)
        param[2:] = self.v0[:]
        param_lsq = optimize.least_squares(fit_circle_v, param, args=(self.xyz, self.r))
        param_lsq = param_lsq.x
        xy_o = param_lsq[0:2]
        v = param_lsq[2:]
        
        xyz_p = project2plane(self.xyz, v)
        xyz_p[:, 0:2] = xyz_p[:, 0:2] - xy_o
        xy_p = xyz_p[:, 0:2]
        
        d = np.linalg.norm(xy_p, axis=1) - self.r
        d = d.reshape(-1, 1)
        
        error = d
        
        return xyz_p, d, error
    
    def compute_d_ellipse(self):
        
        param = np.zeros(8)
        param[2:5] = self.v0[:]
        param[7] = self.r
        param_lsq = optimize.least_squares(fit_ellipse_v, param, args=(self.xyz))
        param_lsq = param_lsq.x
        xy_o = param_lsq[0:2]
        v = param_lsq[2:5]
        f_delta = param_lsq[5:7]
        r_ellipse = param_lsq[7]
        
        xyz_p = project2plane(self.xyz, v)
        xyz_p[:, 0:2] = xyz_p[:, 0:2] - xy_o
        xy_p = xyz_p[:, 0:2]
        theta = np.arctan2(xy_p[1], xy_p[0])
        
        a = r_ellipse
        c = np.linalg.norm(f_delta)
        b = np.sqrt(a ** 2 - c ** 2)
        theta_ellipse = np.arcsin(f_delta[1] / c)
        ovalisation = np.asarray([a, b, theta_ellipse])
        
        theta = theta - theta_ellipse
        
        x_ellipse = a * np.cos(theta)
        y_ellipse = b * np.sin(theta)
        xy_ellipse = np.hstack(x_ellipse.reshape(-1, 1), y_ellipse.reshape(-1, 1))
        xy_ellipse = rotate_xy(xy_ellipse, theta_ellipse)
        
        d = np.linalg.norm(xy_ellipse, axis=1) - self.r
        d = d.reshape(-1, 1)
        
        error = np.linalg.norm(xy_p, axis=1) - np.linalg.norm(xy_ellipse, axis=1)
        
        return xyz_p, d, error, ovalisation
    
    
    # def compute_deformation(self):
        
    #     xyz = self.xyz
    #     label = self.label
    #     r = self.r
    #     num_seg = self.num_seg
    #     v0 = self.v0
    #     d = []

    #     param_ring = np.zeros(5)
    #     param_ring[2:] = v0[:]
    #     plsq_ring = optimize.leastsq(self.compute_dis2axi, param_ring, args=(xyz, r))
    #     plsq_ring = plsq_ring[0]
    #     xy0_ring = plsq_ring[0:2]
    #     v = plsq_ring[2:]
        
    #     d1 = self.compute_dis2axi(plsq_ring, xyz, r)
    #     d1 = d1.reshape(-1, 1)
    #     d.append(d1)
        
    #     param_ring_e = np.zeros(3)
    #     param_ring_e[0], param_ring_e[1], param_ring_e[2] = 0.01, 0, r
    #     plsq_ring_e = optimize.leastsq(self.fit_ellipse, param_ring_e, args=(xyz, r, xy0_ring, v))
    #     plsq_ring_e = plsq_ring_e[0]
    #     f_delta = plsq_ring_e[0:2]
    #     r_ellipse = plsq_ring_e[2]

    #     ovalization, d2 = self.compute_dis_fitted_ellipse(xyz, r, xy0_ring, v, f_delta, r_ellipse)
    #     d2 = d2.reshape(-1, 1)
    #     d.append(d2)
        
    #     if num_seg > 1:
    #         d3 = np.zeros(xyz.shape[0])
    #         for i in range(1, num_seg + 1):
    #             index = label[:] == i
    #             if index.any() == True:
    #                 xyz_seg = xyz[index, :]
    #                 plsq_seg = optimize.leastsq(self.fit_circle, xy0_ring, args=(xyz_seg, r, v))
    #                 plsq_seg = plsq_seg[0]
    #                 xy0_seg = plsq_seg
    #                 d3[index] = self.compute_dis_fitted_circle(xyz_seg, r, xy0_ring, v, xy0_seg)[:]
    #         d3 = d3.reshape(-1, 1)
    #         d.append(d3)
        
    #     d = np.hstack(d)
        
    #     self.v = v
    #     self.xy0_ring = xy0_ring
    #     self.ovalization = ovalization
    #     self.d = d

    #     return self.d
    
    # @staticmethod
    # def compute_dis_fitted_circle(xyz, r, xy0_ring, v, xy0_seg):
        
    #     xy_p = Ring.project2plane(xyz, v)
    #     xy_p_fitted = xy_p - xy0_seg.reshape(1, -1)
    #     xy_p_fitted = xy_p_fitted / np.linalg.norm(xy_p_fitted, axis=1).reshape(-1, 1) * r
    #     xy_p_fitted = xy_p_fitted + xy0_seg.reshape(1, -1)
        
    #     d = xy_p_fitted - xy0_ring
    #     d = np.linalg.norm(d, axis=1) - r
        
    #     return d