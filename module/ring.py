import copy
import numpy as np
from scipy import optimize

from module.utils import compute_normal, fit_circle, fit_circle_v, fit_ellipse_v, project2plane, rotate_xy, solve_contradiction


class Ring():
    
    def __init__(self, xyz, intensity, label, r, length, width, num_seg, angles_b, angles_m, angles_f, v0_dir):
        
        self.offset = np.mean(xyz, axis=0)
        
        self.xyz = xyz - self.offset
        self.intensity = intensity
        self.label = label
        self.num_point = self.xyz.shape[0]
        
        self.r = r
        self.length = length
        self.width = width
        self.num_seg = num_seg
        self.angles_b = angles_b
        self.angles_m = angles_m
        self.angles_f = angles_f
        
        normal = compute_normal(xyz, 16)
        self.v0 = solve_contradiction(normal)
        if np.dot(self.v0, v0_dir) < 0:
            self.v0 = - self.v0
        
        self.d_circle = None
        self.d_ellipse = None
        self.d_seg_circle = None
    
    def compute_d_circle(self):
        
        param = np.zeros(5)
        param[2:] = self.v0[:]
        param_ls = optimize.least_squares(fit_circle_v, param, args=(self.xyz, self.r))
        param_ls = param_ls.x
        xy_o = param_ls[0:2]
        v = param_ls[2:]
        if np.dot(v, self.v0) < 0:
            v = - v
            xy_o[0] = - xy_o[0]
        xyz_p = project2plane(self.xyz, v)
        xyz_p[:, 0:2] = xyz_p[:, 0:2] - xy_o
        xy_p = xyz_p[:, 0:2]
        
        d = np.linalg.norm(xy_p, axis=1) - self.r
        d = d.reshape(-1, 1)
        
        error = d
        
        self.d_circle = (xyz_p, d, error)
        
        return xyz_p, d, error
    
    def compute_d_ellipse(self):
        
        param = np.zeros(8)
        param[2:5] = self.v0[:]
        param[7] = self.r
        param_ls = optimize.least_squares(fit_ellipse_v, param, args=(self.xyz,))
        param_ls = param_ls.x
        xy_o = param_ls[0:2]
        v = param_ls[2:5]
        f_delta = param_ls[5:7]
        r_ellipse = param_ls[7]
        
        if np.dot(v, self.v0) < 0:
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
        
        xyz_p = project2plane(self.xyz, v)
        xyz_p[:, 0:2] = xyz_p[:, 0:2] - xy_o
        xy_p = xyz_p[:, 0:2]
        theta = np.arctan2(xy_p[:, 1], xy_p[:, 0])
        
        theta = theta - theta_ellipse
        
        x_ellipse = a * np.cos(theta)
        y_ellipse = b * np.sin(theta)
        xy_ellipse = np.hstack((x_ellipse.reshape(-1, 1), y_ellipse.reshape(-1, 1)))
        xy_ellipse = rotate_xy(xy_ellipse, theta_ellipse)
        
        d = np.linalg.norm(xy_ellipse, axis=1) - self.r
        d = d.reshape(-1, 1)
        
        error = np.linalg.norm(xy_p, axis=1) - np.linalg.norm(xy_ellipse, axis=1)
        error = error.reshape(-1, 1)
        
        self.d_ellipse = (xyz_p, d, error, ovalisation)
        
        return xyz_p, d, error, ovalisation
    
    def compute_d_seg_circle(self):
        
        if self.d_ellipse is not None:
            xyz_p = self.d_ellipse[0]
        else:
            xyz_p, _, _, _ = self.compute_d_ellipse()
        
        xy_o_all = np.zeros((self.num_seg, 2))
        d = np.zeros(self.xyz.shape[0])
        error = np.zeros(self.xyz.shape[0])
        
        for i in range(self.num_seg):
            index = np.where(self.label == i + 1)
            index = index[0]
            if index.shape[0] == 0:
                continue
            
            xy_p_seg = xyz_p[index, 0:2]
            param = np.zeros(2)
            param_ls = optimize.least_squares(fit_circle, param, args=(xy_p_seg, self.r))
            param_ls = param_ls.x
            xy_o_seg = param_ls
            xy_o_all[i, :] = xy_o_seg[:]
            
            theta_seg = np.arctan2(xy_p_seg[:, 1] - xy_o_seg[1], xy_p_seg[:, 0] - xy_o_seg[0])
            x_circle_seg = self.r * np.cos(theta_seg)
            y_circle_seg = self.r * np.sin(theta_seg)
            xy_circle_seg = np.hstack((x_circle_seg.reshape(-1, 1), y_circle_seg.reshape(-1, 1)))
            
            d[index] = np.linalg.norm(xy_circle_seg + xy_o_seg, axis=1) - self.r
            error[index] = np.linalg.norm(xy_p_seg - xy_o_seg, axis=1) - self.r
        
        d = d.reshape(-1, 1)
        error = error.reshape(-1, 1)
        
        self.d_seg_circle = (xyz_p, d, error, xy_o_all)
        
        return xyz_p, d, error
    
    
    # def compute_d_joint(self):
        
    #     if self.d_seg_circle is None:
    #         self.compute_d_seg_circle()
    #     xyz_p = self.d_seg_circle[0]
    #     xy_o_all = self.d_seg_circle[3]
        
    #     param_0 = np.zeros(2)
    #     bounds = ([- 0.1, - np.pi], [0.1, np.pi])
        
    #     param_0_ls = optimize.least_squares(Ring.fit_0, param_0, bounds=bounds, args=(self.num_seg, self.r, self.length, self.width, self.angles_b, self.angles_m, self.angles_f, xyz_p, self.label), xtol=None)
        
        
    
    
    def compute_d_seg_lin(self):
        
        if self.d_seg_circle is None:
            self.compute_d_seg_circle()
        xyz_p = self.d_seg_circle[0]
        xy_o_key = self.d_seg_circle[3][0, :]        
        
        param_0 = np.zeros(2)
        bounds = ([- 0.1, - np.pi], [0.1, np.pi])
        
        param_0_ls = optimize.least_squares(Ring.fit_0, param_0, bounds=bounds, args=(self.num_seg, self.r, self.length, self.width, self.angles_b, self.angles_m, self.angles_f, xyz_p, self.label), xtol=None)
        
        # print(param_0_ls.message)
        # print(param_0_ls.success)
        # print(param_0_ls.cost)
        # print(np.max(param_0_ls.fun))
        # print(np.min(param_0_ls.fun))
        # print(param_0_ls.x)
        param_0_ls = param_0_ls.x
        
        delta_z = param_0_ls[0]
        delta_theta = param_0_ls[1]
        
        xyz_p[:, 2] = xyz_p[:, 2] + delta_z
        xyz_p = rotate_xy(xyz_p, delta_theta)
        xy_o_key = rotate_xy(xy_o_key.reshape(1, 2), delta_theta)
        xyz_p[:, 0:2] = xyz_p[:, 0:2] - xy_o_key
        
        param_1 = np.zeros(2 * self.num_seg)
        
        lbs = []
        ubs = []
        for i in range(self.num_seg):
            lbs.append(-0.02)
            ubs.append(0.02)
        for i in range(self.num_seg):
            lbs.append(-np.pi / 36)
            ubs.append(np.pi / 36)
        bounds = (lbs, ubs)
        
        param_1_ls = optimize.least_squares(Ring.fit_1, param_1, bounds=bounds, args=(self.num_seg, self.r, self.length, self.width, self.angles_b, self.angles_m, self.angles_f, xyz_p, self.label))
        # print(param_1_ls.success)
        # print(param_1_ls.message)
        # print(param_1_ls.x)
        param_1_ls = param_1_ls.x
        
        dislocation_all = param_1_ls[0:self.num_seg]
        rotation_all = param_1_ls[self.num_seg:2 * self.num_seg]
        print(dislocation_all)
        print(rotation_all)
        
        param_seg_ls = param_1_ls[0:2 * self.num_seg]
        
        position_all = Ring.compute_seg_position(param_seg_ls, self.num_seg, self.r, self.width, self.angles_m)
        print(position_all[self.num_seg, 0:2])
        print(position_all[self.num_seg, 2] / np.pi * 180)
        xy_o_all = position_all[:, 0:2]
        
        d = np.zeros(self.xyz.shape[0])
        error = np.zeros(self.xyz.shape[0])
        
        for i in range(self.num_seg):
            
            index = np.where(self.label == i + 1)[0]
            xy_p_seg = xyz_p[index, 0:2]
            xy_p_seg = xy_p_seg - xy_o_all[i]
            theta_seg = np.arctan2(xy_p_seg[:, 1], xy_p_seg[:, 0])
            xy_circle_seg = self.r * np.hstack((np.cos(theta_seg).reshape(-1, 1), np.sin(theta_seg).reshape(-1, 1)))
            
            error_seg = np.linalg.norm(xy_p_seg, axis=1) - np.linalg.norm(xy_circle_seg, axis=1)
            error[index] = error_seg[:]
            
            xy_circle_seg = xy_circle_seg + xy_o_all[i] + xy_o_key
            xy_circle_seg = rotate_xy(xy_circle_seg, - delta_theta)
            d_seg = np.linalg.norm(xy_circle_seg, axis=1) - self.r
            d[index] = d_seg[:]
        
        d = d.reshape(-1, 1)
        error = error.reshape(-1, 1)
        
        return xyz_p, d, error, dislocation_all, rotation_all
    
    @staticmethod
    def compute_seg_position(param_seg, num_seg, r, width, angles_m):
        
        dislocation_all = param_seg[0:num_seg]
        rotation_all = param_seg[num_seg:2 * num_seg]
        
        position_all = np.zeros((num_seg + 1, 3))
        position_all[0, 2] = np.pi
        
        for i in range(1, num_seg + 1):
            
            xy_o_0, theta_mc_0 = position_all[i - 1, 0:2], position_all[i - 1, 2]
            
            theta_dislocation = theta_mc_0 + angles_m[i - 1][1]
            xy_o_1 = xy_o_0 - dislocation_all[i - 1] * np.asarray([np.cos(theta_dislocation), np.sin(theta_dislocation)])
            
            rotation_centre = xy_o_0 + (r + width / 2 - dislocation_all[i - 1]) * np.asarray([np.cos(theta_dislocation), np.sin(theta_dislocation)])
            xy_o_1 = xy_o_1 - rotation_centre
            xy_o_1 = xy_o_1.reshape(1, 2)
            xy_o_1 = rotate_xy(xy_o_1, rotation_all[i - 1])
            xy_o_1 = xy_o_1.reshape(2)
            xy_o_1 = xy_o_1 + rotation_centre
            
            if i < num_seg:
                theta_mc_1 = theta_dislocation - angles_m[i][0] + rotation_all[i - 1]
            else:
                theta_mc_1 = theta_dislocation - angles_m[0][0] + rotation_all[i - 1]
                
            position_all[i, 0:2], position_all[i, 2] = xy_o_1[:], theta_mc_1
        
        return position_all
    
    @staticmethod
    def fit_0(param_0, num_seg, r, length, width, angles_b, angles_m, angles_f, xyz_p, label):
        
        xyz_p = copy.deepcopy(xyz_p)
        
        delta_z = param_0[0]
        delta_theta = param_0[1]
        
        xyz_p[:, 2] = xyz_p[:, 2] + delta_z
        xyz_p = rotate_xy(xyz_p, delta_theta)
        
        param_seg = np.zeros(2 * num_seg)
        positioan_all = Ring.compute_seg_position(param_seg, num_seg, r, width, angles_m)
        
        error_all = np.zeros(xyz_p.shape[0])
        
        for i in range(num_seg):
            
            index = np.where(label == i + 1)[0]
            if index.shape[0] == 0:
                continue
            
            xy_seg_o = positioan_all[i, 0:2]
            theta_mc_seg = positioan_all[i, 2]
            
            xyz_p_seg = xyz_p[index, :]
            xy_p_seg = xyz_p_seg[:, 0:2]
            z_p_seg = xyz_p_seg[:, 2]
            xy_p_seg = xy_p_seg - xy_seg_o            
            theta_seg = np.arctan2(xy_p_seg[:, 1], xy_p_seg[:, 0])
            
            error = np.zeros(xyz_p_seg.shape[0])

            vector_mc_seg = np.asarray([np.cos(theta_mc_seg), np.sin(theta_mc_seg)]).reshape(1, 2)
            vector_theta_seg = np.hstack((np.cos(theta_seg).reshape(-1, 1), np.sin(theta_seg).reshape(-1, 1)))
            delta_theta_mc_seg = np.arccos(np.dot(vector_theta_seg, vector_mc_seg.T)).reshape(-1)
            flag_delta_theta_mc_seg = np.cross(vector_mc_seg, vector_theta_seg)
            index_flag = np.where(flag_delta_theta_mc_seg < 0)[0]
            delta_theta_mc_seg[index_flag] = - delta_theta_mc_seg[index_flag]
            
            boundary_l = np.interp(z_p_seg, [- length / 2, length / 2], [angles_b[i][1], angles_f[i][1]])
            xy_l = np.hstack((np.cos(theta_mc_seg + boundary_l).reshape(-1, 1), np.sin(theta_mc_seg + boundary_l).reshape(-1, 1))) * r
            boundary_u = np.interp(z_p_seg, [- length / 2, length / 2], [angles_b[i][0], angles_f[i][0]])
            xy_u = np.hstack((np.cos(theta_mc_seg + boundary_u).reshape(-1, 1), np.sin(theta_mc_seg + boundary_u).reshape(-1, 1))) * r
            
            index_out = np.where(delta_theta_mc_seg < boundary_l)[0]
            error[index_out] = np.linalg.norm(xy_l[index_out, :] - xy_p_seg[index_out, :], axis=1)
            index_out = np.where(delta_theta_mc_seg > boundary_u)[0]
            error[index_out] = np.linalg.norm(xy_p_seg[index_out, :] - xy_u[index_out, :], axis=1)
            
            error_all[index] = error[:]

        return error_all
    
    @staticmethod
    def fit_1(param_1, num_seg, r, length, width, angles_b, angles_m, angles_f, xyz_p, label):
        
        xyz_p = copy.deepcopy(xyz_p)
        
        param_seg = param_1
        
        positioan_all = Ring.compute_seg_position(param_seg, num_seg, r, width, angles_m)
        
        error_all = np.zeros(xyz_p.shape[0])
        
        for i in range(1, num_seg + 1):
            
            if i < num_seg:
                index = np.where(label == i + 1)[0]
            else:
                index = np.where(label == 1)[0]
            if index.shape[0] == 0:
                continue
            
            xy_seg_o = positioan_all[i, 0:2]
            
            xy_p_seg = xyz_p[index, 0:2]
            xy_p_seg = xy_p_seg - xy_seg_o            
            
            error = np.linalg.norm(xy_p_seg, axis=1) - r
            
            error_all[index] = error[:]
            
        return error_all
    
