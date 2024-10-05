import copy
import numpy as np
from scipy import optimize

from module.utils import compute_normal, fit_circle, fit_ellipse, fit_ellipse_v, fit_fourier, fit_polynomial_residual, fit_polynomial_residual_zone, project2plane, rotate_xy, solve_contradiction


class Ring():
    
    def __init__(self, xyz, intensity, label, r, length, width, num_seg, angle_joint_width, angles_b, angles_m, angles_f, v0_dir):
        
        self.offset = np.mean(xyz, axis=0)
        
        self.xyz = copy.deepcopy(xyz)
        self.xyz = self.xyz - self.offset
        self.intensity = intensity
        self.label = label
        self.num_point = self.xyz.shape[0]
        self.angle_joint_width = angle_joint_width
        
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
        
        self.d_ellipse = None
        
        self.delta = None
    
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
        
        d = np.linalg.norm(xy_ellipse, axis=1)
        d0 = np.linalg.norm(xy_p, axis=1)
        error = d0 - np.linalg.norm(xy_ellipse, axis=1)
        d0 -= self.r
        d -= self.r
        
        d0 = d0.reshape(-1, 1)
        d = d.reshape(-1, 1)
        error = error.reshape(-1, 1)
        
        self.d_ellipse = (xyz_p, d0, d, error, ovalisation)
        
        return xyz_p, d0, d, error, ovalisation
    
    def compute_d_seg_circle(self):
    
        if self.d_ellipse is not None:
            xyz_p = self.d_ellipse[0]
        else:
            xyz_p, _, _, _ = self.compute_d_ellipse()
        
        xy_o_all = np.zeros((self.num_seg, 2))
        d = np.zeros(self.xyz.shape[0])
        error = np.zeros(self.xyz.shape[0])
        
        for i in range(self.num_seg):
            index = np.where(self.label == i + 1)[0]
            if index.shape[0] < 48:
                xy_o_all[i, :] = np.nan
                continue
            xy_p_seg = xyz_p[index, 0:2]
            
            param = np.zeros(2)
            param_ls = optimize.least_squares(fit_circle, param, xtol=None, gtol=None, loss='soft_l1', f_scale=0.001, args=(xy_p_seg, self.r))
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
        
        return xyz_p, d, error
    
    def compute_d_seg_ellipse_polynomial(self, cfg_e_p):
        
        if self.d_ellipse is not None:
            xyz_p = self.d_ellipse[0]
        else:
            xyz_p, _, _, _, _ = self.compute_d_ellipse()
        
        if self.delta is None:
            param_0 = np.zeros(2)
            bounds_0 = ([- 0.1, - np.pi], [0.1, np.pi])
            param_1 = np.zeros(2 * self.num_seg)
            param_0_ls = optimize.least_squares(Ring.fit_0, param_0, bounds=bounds_0, args=(param_1, self.num_seg, self.r, self.length, self.width, self.angle_joint_width, self.angles_b, self.angles_m, self.angles_f, xyz_p, self.label))
            param_0_ls = param_0_ls.x
            self.delta = param_0_ls

        xyz_p[:, 2] = xyz_p[:, 2] + self.delta[0]
        xyz_p = rotate_xy(xyz_p, self.delta[1])
        
        d = np.zeros(self.num_point)
        error = np.zeros(self.num_point)
        
        r_length = cfg_e_p['r_length']
        
        ovalisation_seg_all = []
        
        polynomial_seg_all = []
        k_polynomial_max = cfg_e_p['k_polynomial_max']
        
        polynomial_zone_seg_all = np.zeros((self.num_seg * 2, 3))
        angle_zone = cfg_e_p['angle_zone'] / 180 * np.pi
        
        theta_seg_m = np.pi
        for i in range(self.num_seg):
            
            if i == 0:
                k_polynomial = 3
            else:
                k_polynomial = k_polynomial_max
            
            index = np.where(self.label == i + 1)[0]
            
            if index.shape[0] < 48:
                ovalisation_seg_all.append(np.full(3, np.nan))
                polynomial_seg_all.append(np.full(k_polynomial_max + 1, np.nan))
                theta_seg_m += self.angles_m[i][1]
                if i == self.num_seg - 1:
                    theta_seg_m -= self.angles_m[0][0]
                else:
                    theta_seg_m -= self.angles_m[i + 1][0]
                continue
            
            xyz_p_seg = xyz_p[index, 0:3]
            xy_p_seg = xyz_p_seg[:, 0:2]
            index_middle = np.where((xyz_p_seg[:, 2] < self.length / r_length) & (xyz_p_seg[:, 2] > - self.length / r_length))[0]
            xy_p_seg_middle = xyz_p_seg[index_middle, 0:2]
            
            if cfg_e_p['flag_ellipse']:
                param = np.zeros(3)
                param[2] = self.r
                param_ls = optimize.least_squares(fit_ellipse, param, xtol=None, gtol=None, loss='soft_l1', f_scale=0.001, args=(xy_p_seg_middle,))
                # param_ls = optimize.least_squares(fit_ellipse, param, xtol=None, gtol=None, loss='soft_l1', f_scale=1, args=(xy_p_seg_middle,))
                param_ls = param_ls.x
                f_delta = param_ls[0:2]
                r_ellipse = param_ls[2]
                if f_delta[0] < 0:
                    f_delta = - f_delta
                a = r_ellipse
                c = np.linalg.norm(f_delta)
                b = np.sqrt(a ** 2 - c ** 2)
                theta_ellipse = np.arctan2(f_delta[1], f_delta[0])
            else:
                a = 0
                b = 0
                theta_ellipse = 0
                
            ovalisation_seg_all.append([a, b, theta_ellipse])
            
            theta_seg_middle = np.arctan2(xy_p_seg_middle[:, 1], xy_p_seg_middle[:, 0])
            if i == 0:
                theta_seg_middle[np.where(theta_seg_middle < 0)[0]] += 2 * np.pi
            
            theta_seg_middle = theta_seg_middle - theta_ellipse
            
            x_seg_middle_ellipse = a * np.cos(theta_seg_middle)
            y_seg_middle_ellipse = b * np.sin(theta_seg_middle)
            xy_seg_middle_ellipse = np.hstack((x_seg_middle_ellipse.reshape(-1, 1), y_seg_middle_ellipse.reshape(-1, 1)))
            xy_seg_middle_ellipse = rotate_xy(xy_seg_middle_ellipse, theta_ellipse)
            
            theta_seg_middle = theta_seg_middle + theta_ellipse
            
            residual = np.linalg.norm(xy_p_seg_middle, axis=1) - np.linalg.norm(xy_seg_middle_ellipse, axis=1)
            
            param = np.zeros(k_polynomial_max + 1)
            if not cfg_e_p['flag_ellipse']:
                param[0] = self.r

            if cfg_e_p['flag_polynomial']:
                param_ls = optimize.least_squares(fit_polynomial_residual, param, xtol=None, gtol=None, loss='soft_l1', f_scale=0.001, args=(k_polynomial, theta_seg_middle, residual))
                # param_ls = optimize.least_squares(fit_polynomial_residual, param, xtol=None, gtol=None, loss='soft_l1', f_scale=1, args=(k_polynomial, theta_seg_middle, residual))
                param_ls = param_ls.x
                param_polynomial = param_ls
            else:
                param_polynomial = param
            
            polynomial_seg_all.append(param_polynomial)
            
            residual -= param_polynomial[0]
            for j in range(k_polynomial):
                residual -= param_polynomial[j + 1] * (theta_seg_middle ** (j + 1))
            
            theta_joints = [theta_seg_m + self.angles_m[i][1], theta_seg_m + self.angles_m[i][0]]
            
            if cfg_e_p['flag_zone']:
                index_zone = np.where(theta_seg_middle < theta_joints[0] + angle_zone)[0]
                if index_zone.shape[0] < 12:
                    pass
                else:
                    param = np.zeros(3)
                    param_ls = optimize.least_squares(fit_polynomial_residual_zone, param, xtol=None, gtol=None, loss='soft_l1', f_scale=0.001, args=(theta_joints[0] + angle_zone, theta_seg_middle[index_zone], residual[index_zone]))
                    # param_ls = optimize.least_squares(fit_polynomial_residual_zone, param, xtol=None, gtol=None, loss='soft_l1', f_scale=1, args=(theta_joints[0] + angle_zone, theta_seg_middle[index_zone], residual[index_zone]))
                    param_ls = param_ls.x
                    polynomial_zone_seg_all[2 * i, :] = param_ls
                index_zone = np.where(theta_seg_middle > theta_joints[1] - angle_zone)[0]
                if index_zone.shape[0] < 12:
                    pass
                else:
                    param = np.zeros(3)
                    param_ls = optimize.least_squares(fit_polynomial_residual_zone, param, xtol=None, gtol=None, loss='soft_l1', f_scale=0.001, args=(theta_joints[1] - angle_zone, theta_seg_middle[index_zone], residual[index_zone]))
                    # param_ls = optimize.least_squares(fit_polynomial_residual_zone, param, xtol=None, gtol=None, loss='soft_l1', f_scale=1, args=(theta_joints[1] - angle_zone, theta_seg_middle[index_zone], residual[index_zone]))
                    param_ls = param_ls.x
                    polynomial_zone_seg_all[2 * i + 1, :] = param_ls
                
            theta_seg = np.arctan2(xy_p_seg[:, 1], xy_p_seg[:, 0])
            if i == 0:
                theta_seg[np.where(theta_seg < 0)[0]] += 2 * np.pi
            
            theta_seg = theta_seg - theta_ellipse
            x_seg_ellipse = a * np.cos(theta_seg)
            y_seg_ellipse = b * np.sin(theta_seg)
            xy_seg_ellipse = np.hstack((x_seg_ellipse.reshape(-1, 1), y_seg_ellipse.reshape(-1, 1)))
            xy_seg_ellipse = rotate_xy(xy_seg_ellipse, theta_ellipse)
            theta_seg = theta_seg + theta_ellipse
            d[index] = np.linalg.norm(xy_seg_ellipse, axis=1)
            d[index] += param_polynomial[0]
            for j in range(k_polynomial):
                d[index] += param_polynomial[j + 1] * (theta_seg ** (j + 1))
            
            index_zone = np.where(theta_seg < theta_joints[0] + angle_zone)[0]
            # d[index][index_zone] += polynomial_zone_seg_all[2 * i, 0] * ((theta_seg[index_zone] - (theta_joints[0] + angle_zone)) ** 2) + polynomial_zone_seg_all[2 * i, 1] * ((theta_seg[index_zone] - (theta_joints[0] + angle_zone)) ** 4)
            # d[index][index_zone] += polynomial_zone_seg_all[2 * i, 0] * ((theta_seg[index_zone] - (theta_joints[0] + angle_zone)) ** 1) + polynomial_zone_seg_all[2 * i, 1] * ((theta_seg[index_zone] - (theta_joints[0] + angle_zone)) ** 2)
            d[index][index_zone] += polynomial_zone_seg_all[2 * i, 0] * ((theta_seg[index_zone] - (theta_joints[0] + angle_zone)) ** 1) + polynomial_zone_seg_all[2 * i, 1] * ((theta_seg[index_zone] - (theta_joints[0] + angle_zone)) ** 2) + polynomial_zone_seg_all[2 * i, 2] * ((theta_seg[index_zone] - (theta_joints[0] + angle_zone)) ** 3)
            
            index_zone = np.where(theta_seg > theta_joints[1] - angle_zone)[0]
            # d[index][index_zone] += polynomial_zone_seg_all[2 * i + 1, 0] * ((theta_seg[index_zone] - (theta_joints[1] - angle_zone)) ** 2) + polynomial_zone_seg_all[2 * i + 1, 1] * ((theta_seg[index_zone] - (theta_joints[1] - angle_zone)) ** 4)
            # d[index][index_zone] += polynomial_zone_seg_all[2 * i + 1, 0] * ((theta_seg[index_zone] - (theta_joints[1] - angle_zone)) ** 1) + polynomial_zone_seg_all[2 * i + 1, 1] * ((theta_seg[index_zone] - (theta_joints[1] - angle_zone)) ** 2)
            d[index][index_zone] += polynomial_zone_seg_all[2 * i + 1, 0] * ((theta_seg[index_zone] - (theta_joints[1] - angle_zone)) ** 1) + polynomial_zone_seg_all[2 * i + 1, 1] * ((theta_seg[index_zone] - (theta_joints[1] - angle_zone)) ** 2) + polynomial_zone_seg_all[2 * i + 1, 2] * ((theta_seg[index_zone] - (theta_joints[1] - angle_zone)) ** 3)
            
            error[index] = np.linalg.norm(xy_p_seg, axis=1) - d[index]
            d[index] = d[index] - self.r
            
            theta_seg_m += self.angles_m[i][1]
            if i == self.num_seg - 1:
                theta_seg_m -= self.angles_m[0][0]
            else:
                theta_seg_m -= self.angles_m[i + 1][0]
        
        ovalisation_seg_all = np.asarray(ovalisation_seg_all)
        polynomial_seg_all = np.asarray(polynomial_seg_all)

        d = d.reshape(-1, 1)
        error = error.reshape(-1, 1)
        
        dislocation_all = np.zeros(self.num_seg)
        rotation_all = np.zeros(self.num_seg)
        theta_seg_m = np.pi
        for i in range(self.num_seg):
            
            if i == 0:
                k_polynomial = 3
            else:
                k_polynomial = k_polynomial_max
            
            theta_joints = [theta_seg_m + self.angles_m[i][1], theta_seg_m + self.angles_m[i][0]]
            
            a = ovalisation_seg_all[i, 0]
            b = ovalisation_seg_all[i, 1]
            theta_ellipse = ovalisation_seg_all[i, 2]
            param_polynomial = polynomial_seg_all[i, :]
            param_polynomial_zone = polynomial_zone_seg_all[2 * i, :]
            
            '''ellipse'''
            theta_joint_last = theta_joints[0] + self.angle_joint_width - theta_ellipse
            
            xy_joint_last = np.asarray([a * np.cos(theta_joint_last), b * np.sin(theta_joint_last)])
            xy_joint_last = rotate_xy(xy_joint_last.reshape(1, 2), theta_ellipse).reshape(2)
            vector_joint_last = np.asarray([- a * np.sin(theta_joint_last), b * np.cos(theta_joint_last)])
            vector_joint_last = rotate_xy(vector_joint_last.reshape(1, 2), theta_ellipse).reshape(2)
            
            theta_joint_last += theta_ellipse
            '''ellipse'''
            
            '''polynomial'''
            r_joint_last = param_polynomial[0]
            for j in range(k_polynomial):
                r_joint_last += param_polynomial[j + 1] * (theta_joint_last ** (j + 1))
            xy_joint_last += np.asarray([r_joint_last * np.cos(theta_joint_last), r_joint_last * np.sin(theta_joint_last)])
            
            vector_joint_last += np.asarray([- r_joint_last * np.sin(theta_joint_last), r_joint_last * np.cos(theta_joint_last)])
            for j in range(k_polynomial):
                vector_joint_last += np.asarray([np.cos(theta_joint_last), np.sin(theta_joint_last)]) * (j + 1) * param_polynomial[j + 1] * (theta_joint_last ** j)
            '''polynomial'''
            
            '''polynomial zone'''
            theta_joint_zone = theta_joints[0] + angle_zone
            # r_joint_last = param_polynomial_zone[0] * ((theta_joint_last - theta_joint_zone) ** 2) + param_polynomial_zone[1] * ((theta_joint_last - theta_joint_zone) ** 4)
            # r_joint_last = param_polynomial_zone[0] * (theta_joint_last - theta_joint_zone)
            # r_joint_last = param_polynomial_zone[0] * ((theta_joint_last - theta_joint_zone) ** 1) + param_polynomial_zone[1] * ((theta_joint_last - theta_joint_zone) ** 2)
            r_joint_last = param_polynomial_zone[0] * ((theta_joint_last - theta_joint_zone) ** 1) + param_polynomial_zone[1] * ((theta_joint_last - theta_joint_zone) ** 2) + param_polynomial_zone[2] * ((theta_joint_last - theta_joint_zone) ** 3)
            xy_joint_last += np.asarray([r_joint_last * np.cos(theta_joint_last), r_joint_last * np.sin(theta_joint_last)])
            
            vector_joint_last += np.asarray([- r_joint_last * np.sin(theta_joint_last), r_joint_last * np.cos(theta_joint_last)])
            # vector_joint_last += np.asarray([np.cos(theta_joint_last), np.sin(theta_joint_last)]) * (2 * param_polynomial_zone[0] * (theta_joint_last - theta_joint_zone) + 4 * param_polynomial_zone[1] * (theta_joint_last - theta_joint_zone) ** 3)
            # vector_joint_last += np.asarray([np.cos(theta_joint_last), np.sin(theta_joint_last)]) * param_polynomial_zone[0]
            # vector_joint_last += np.asarray([np.cos(theta_joint_last), np.sin(theta_joint_last)]) * (1 * param_polynomial_zone[0] + 2 * param_polynomial_zone[1] * (theta_joint_last - theta_joint_zone) ** 1)
            vector_joint_last += np.asarray([np.cos(theta_joint_last), np.sin(theta_joint_last)]) * (1 * param_polynomial_zone[0] + 2 * param_polynomial_zone[1] * (theta_joint_last - theta_joint_zone) ** 1 + 3 * param_polynomial_zone[2] * (theta_joint_last - theta_joint_zone) ** 2)
            '''polynomial zone'''
            
            vector_joint_last = vector_joint_last / np.linalg.norm(vector_joint_last)
            
            if i == self.num_seg - 1:
                k_polynomial = 3
            else:
                k_polynomial = k_polynomial_max
            
            if i == self.num_seg - 1:
                a = ovalisation_seg_all[0, 0]
                b = ovalisation_seg_all[0, 1]
                theta_ellipse = ovalisation_seg_all[0, 2]
                param_polynomial = polynomial_seg_all[0, :]
                param_polynomial_zone = polynomial_zone_seg_all[1, :]
            else:
                a = ovalisation_seg_all[i + 1, 0]
                b = ovalisation_seg_all[i + 1, 1]
                theta_ellipse = ovalisation_seg_all[i + 1, 2]
                param_polynomial = polynomial_seg_all[i + 1, :]
                param_polynomial_zone = polynomial_zone_seg_all[2 * i + 3, :]
            
            '''ellipse'''
            theta_joint_next = theta_joints[0] - self.angle_joint_width - theta_ellipse
            
            if i == self.num_seg - 1:
                theta_joint_next += 2 * np.pi
            
            xy_joint_next = np.asarray([a * np.cos(theta_joint_next), b * np.sin(theta_joint_next)])
            xy_joint_next = rotate_xy(xy_joint_next.reshape(1, 2), theta_ellipse).reshape(2)
            vector_joint_next = np.asarray([- a * np.sin(theta_joint_next), b * np.cos(theta_joint_next)])
            vector_joint_next = rotate_xy(vector_joint_next.reshape(1, 2), theta_ellipse).reshape(2)
            
            theta_joint_next += theta_ellipse
            '''ellipse'''
            
            '''polynomial'''
            r_joint_next = param_polynomial[0]
            for j in range(k_polynomial):
                r_joint_next += param_polynomial[j + 1] * (theta_joint_next ** (j + 1))
            xy_joint_next += np.asarray([r_joint_next * np.cos(theta_joint_next), r_joint_next * np.sin(theta_joint_next)])
            
            vector_joint_next += np.asarray([- r_joint_next * np.sin(theta_joint_next), r_joint_next * np.cos(theta_joint_next)])
            for j in range(k_polynomial):
                vector_joint_next += np.asarray([np.cos(theta_joint_next), np.sin(theta_joint_next)]) * (j + 1) * param_polynomial[j + 1] * (theta_joint_next ** j)
            '''polynomial'''
            
            '''polynomial zone'''
            theta_joint_zone = theta_joints[0] - angle_zone
            if i == self.num_seg - 1:
                theta_joint_zone += 2 * np.pi
            # r_joint_next = param_polynomial_zone[0] * ((theta_joint_next - theta_joint_zone) ** 2) + param_polynomial_zone[1] * ((theta_joint_next - theta_joint_zone) ** 4)
            # r_joint_next = param_polynomial_zone[0] * (theta_joint_next - theta_joint_zone)
            # r_joint_next = param_polynomial_zone[0] * ((theta_joint_next - theta_joint_zone) ** 1) + param_polynomial_zone[1] * ((theta_joint_next - theta_joint_zone) ** 2)
            r_joint_next = param_polynomial_zone[0] * ((theta_joint_next - theta_joint_zone) ** 1) + param_polynomial_zone[1] * ((theta_joint_next - theta_joint_zone) ** 2) + param_polynomial_zone[2] * ((theta_joint_next - theta_joint_zone) ** 3)
            xy_joint_next += np.asarray([r_joint_next * np.cos(theta_joint_next), r_joint_next * np.sin(theta_joint_next)])
            
            vector_joint_next += np.asarray([- r_joint_next * np.sin(theta_joint_next), r_joint_next * np.cos(theta_joint_next)])
            # vector_joint_next += np.asarray([np.cos(theta_joint_next), np.sin(theta_joint_next)]) * (2 * param_polynomial_zone[0] * (theta_joint_next - theta_joint_zone) + 4 * param_polynomial_zone[1] * (theta_joint_next - theta_joint_zone) ** 3)
            # vector_joint_next += np.asarray([np.cos(theta_joint_next), np.sin(theta_joint_next)]) * param_polynomial_zone[0]
            # vector_joint_next += np.asarray([np.cos(theta_joint_next), np.sin(theta_joint_next)]) * (1 * param_polynomial_zone[0] + 2 * param_polynomial_zone[1] * (theta_joint_next - theta_joint_zone) ** 1)
            vector_joint_next += np.asarray([np.cos(theta_joint_next), np.sin(theta_joint_next)]) * (1 * param_polynomial_zone[0] + 2 * param_polynomial_zone[1] * (theta_joint_next - theta_joint_zone) ** 1 + 3 * param_polynomial_zone[2] * (theta_joint_next - theta_joint_zone) ** 2)
            '''polynomial zone'''
            
            vector_joint_next = vector_joint_next / np.linalg.norm(vector_joint_next)
            
            dislocation_all[i] = np.linalg.norm(xy_joint_last) - np.linalg.norm(xy_joint_next)
            
            rotation = np.arccos(np.dot(vector_joint_last, vector_joint_next))
            if np.cross(vector_joint_last, vector_joint_next) < 0:
                rotation = - rotation
            rotation_all[i] = rotation
            
            theta_seg_m += self.angles_m[i][1]
            if i == self.num_seg - 1:
                theta_seg_m -= self.angles_m[0][0]
            else:
                theta_seg_m -= self.angles_m[i + 1][0]
        
        xy_p_ellipse_polynomial_all = []
        theta_seg_m = np.pi
        for i in range(self.num_seg):
            
            if i == 0:
                k_polynomial = 3
            else:
                k_polynomial = k_polynomial_max
            
            theta_joints = [theta_seg_m + self.angles_m[i][1], theta_seg_m + self.angles_m[i][0]]
            
            a = ovalisation_seg_all[i, 0]
            b = ovalisation_seg_all[i, 1]
            theta_ellipse = ovalisation_seg_all[i, 2]
            
            param_polynomial = polynomial_seg_all[i, :]
            param_polynomial_zone = polynomial_zone_seg_all[2 * i: 2 * i + 2, :]
            
            for theta_per in np.arange(theta_joints[0], theta_joints[1], step=np.pi/1800):
                
                theta_per = theta_per - theta_ellipse
                xy_per = np.asarray([a * np.cos(theta_per), b * np.sin(theta_per)]).reshape(1, 2)
                xy_per = rotate_xy(xy_per, theta_ellipse).reshape(2)
                theta_per = theta_per + theta_ellipse
                
                r_per = param_polynomial[0]
                for j in range(k_polynomial):
                    r_per += param_polynomial[j + 1] * (theta_per ** (j + 1))
                    
                if theta_per < theta_joints[0] + angle_zone:
                    # r_per += param_polynomial_zone[0, 0] * (theta_per - (theta_joints[0] + angle_zone)) ** 2 + param_polynomial_zone[0, 1] * (theta_per - (theta_joints[0] + angle_zone)) ** 4
                    # r_per += param_polynomial_zone[0, 0] * (theta_per - (theta_joints[0] + angle_zone))
                    # r_per += param_polynomial_zone[0, 0] * (theta_per - (theta_joints[0] + angle_zone)) ** 1 + param_polynomial_zone[0, 1] * (theta_per - (theta_joints[0] + angle_zone)) ** 2
                    r_per += param_polynomial_zone[0, 0] * (theta_per - (theta_joints[0] + angle_zone)) ** 1 + param_polynomial_zone[0, 1] * (theta_per - (theta_joints[0] + angle_zone)) ** 2 + param_polynomial_zone[0, 2] * (theta_per - (theta_joints[0] + angle_zone)) ** 3
                elif theta_per > theta_joints[1] - angle_zone:
                    # r_per += param_polynomial_zone[1, 0] * (theta_per - (theta_joints[1] - angle_zone)) ** 2 + param_polynomial_zone[1, 1] * (theta_per - (theta_joints[1] - angle_zone)) ** 4
                    # r_per += param_polynomial_zone[1, 0] * (theta_per - (theta_joints[1] - angle_zone))
                    # r_per += param_polynomial_zone[1, 0] * (theta_per - (theta_joints[1] - angle_zone)) ** 1 + param_polynomial_zone[1, 1] * (theta_per - (theta_joints[1] - angle_zone)) ** 2
                    r_per += param_polynomial_zone[1, 0] * (theta_per - (theta_joints[1] - angle_zone)) ** 1 + param_polynomial_zone[1, 1] * (theta_per - (theta_joints[1] - angle_zone)) ** 2 + param_polynomial_zone[1, 2] * (theta_per - (theta_joints[1] - angle_zone)) ** 3
                
                xy_per += np.asarray([r_per * np.cos(theta_per), r_per * np.sin(theta_per)])
                xy_p_ellipse_polynomial_all.append([xy_per[0], xy_per[1], i + 1])
                
            theta_seg_m += self.angles_m[i][1]
            if i == self.num_seg - 1:
                theta_seg_m -= self.angles_m[0][0]
            else:
                theta_seg_m -= self.angles_m[i + 1][0]
        
        xy_p_ellipse_polynomial_all = np.asarray(xy_p_ellipse_polynomial_all)
        xy_p_norm_all = np.hstack((xyz_p[:, 0:2], self.label.reshape(-1, 1)))
        
        xy_p_ellipse_polynomial_all_10 = copy.deepcopy(xy_p_ellipse_polynomial_all)
        d_ellipse_polynomial_all_10 = np.linalg.norm(xy_p_ellipse_polynomial_all_10[:, 0:2], axis=1)
        xy_p_ellipse_polynomial_all_10[:, 0] = ((d_ellipse_polynomial_all_10 - self.r) * 10 + self.r) * xy_p_ellipse_polynomial_all[:, 0] / d_ellipse_polynomial_all_10
        xy_p_ellipse_polynomial_all_10[:, 1] = ((d_ellipse_polynomial_all_10 - self.r) * 10 + self.r) * xy_p_ellipse_polynomial_all[:, 1] / d_ellipse_polynomial_all_10
        xy_p_ellipse_polynomial_all = np.hstack((xy_p_ellipse_polynomial_all, xy_p_ellipse_polynomial_all_10))
        
        xy_p_norm_all_10 = copy.deepcopy(xy_p_norm_all)
        d_norm_all_10 = np.linalg.norm(xy_p_norm_all_10[:, 0:2], axis=1)
        xy_p_norm_all_10[:, 0] = ((d_norm_all_10 - self.r) * 10 + self.r) * xy_p_norm_all_10[:, 0] / d_norm_all_10
        xy_p_norm_all_10[:, 1] = ((d_norm_all_10 - self.r) * 10 + self.r) * xy_p_norm_all_10[:, 1] / d_norm_all_10
        xy_p_norm_all = np.hstack((xy_p_norm_all, xy_p_norm_all_10))
        
        '''joint edge'''
        label_dislocation = np.zeros(self.num_point)
        theta_seg_m = np.pi
        theta_p = np.arctan2(xyz_p[:, 1], xyz_p[:, 0])
        for i in range(self.num_seg):
            theta_joint = theta_seg_m + self.angles_m[i][1]
            index_edge = np.where((theta_p > (theta_joint - angle_zone)) & (theta_p < (theta_joint + angle_zone)))[0]
            if not np.isnan(dislocation_all[i]):
                label_dislocation[index_edge] = np.abs(dislocation_all[i])
            
            theta_seg_m += self.angles_m[i][1]
            if i == self.num_seg - 1:
                theta_seg_m -= self.angles_m[0][0]
            else:
                theta_seg_m -= self.angles_m[i + 1][0]
        
        label_dislocation = label_dislocation.reshape(-1, 1)
        '''joint edge'''
        
        return xyz_p, d, error, dislocation_all, rotation_all, xy_p_norm_all, xy_p_ellipse_polynomial_all, label_dislocation
    
    def compute_d_seg_fourier(self):
        
        if self.d_ellipse is not None:
            xyz_p = self.d_ellipse[0]
        else:
            xyz_p, _, _, _, _ = self.compute_d_ellipse()
        
        if self.delta is None:
            param_0 = np.zeros(2)
            bounds_0 = ([- 0.1, - np.pi], [0.1, np.pi])
            param_1 = np.zeros(2 * self.num_seg)
            param_0_ls = optimize.least_squares(Ring.fit_0, param_0, bounds=bounds_0, args=(param_1, self.num_seg, self.r, self.length, self.width, self.angle_joint_width, self.angles_b, self.angles_m, self.angles_f, xyz_p, self.label))
            param_0_ls = param_0_ls.x
            self.delta = param_0_ls
        
        xyz_p[:, 2] = xyz_p[:, 2] + self.delta[0]
        xyz_p = rotate_xy(xyz_p, self.delta[1])
        
        d = np.zeros(self.num_point)
        error = np.zeros(self.num_point)
        
        fourier_seg_all = []
        
        k_fourier = 16
        
        r_length = 4
        
        theta_seg_m = np.pi
        
        for i in range(self.num_seg):
            
            index = np.where(self.label == i + 1)[0]
            
            if index.shape[0] < 48:
                fourier_seg_all.append(np.full(2 * k_fourier + 1, np.nan))
                theta_seg_m += self.angles_m[i][1]
                if i == self.num_seg - 1:
                    theta_seg_m -= self.angles_m[0][0]
                else:
                    theta_seg_m -= self.angles_m[i + 1][0]
                continue
            
            xyz_p_seg = xyz_p[index, 0:3]
            xy_p_seg = xyz_p_seg[:, 0:2]
            index_middle = np.where((xyz_p_seg[:, 2] < self.length / r_length) & (xyz_p_seg[:, 2] > - self.length / r_length))[0]
            xy_p_seg_middle = xyz_p_seg[index_middle, 0:2]     
            
            param = np.zeros(2 * k_fourier + 1)
            param[-1] = self.r
            param_ls = optimize.least_squares(fit_fourier, param, loss='soft_l1', f_scale=0.001, args=(k_fourier, xy_p_seg_middle, self.r, theta_seg_m))
            param_ls = param_ls.x
            param_ls[-1] = self.r
            fourier_seg_all.append(param_ls)
            
            theta_seg = np.arctan2(xy_p_seg[:, 1], xy_p_seg[:, 0])
            
            d[index] = param_ls[-1]
            for j in range(k_fourier):
                d[index] += param_ls[j] * np.cos((j + 1) * (theta_seg - theta_seg_m)) + param_ls[k_fourier + j] * np.sin((j + 1) * (theta_seg - theta_seg_m))
            error[index] = np.linalg.norm(xy_p_seg, axis=1) - d[index]
            d[index] = d[index] - self.r
            
            theta_seg_m += self.angles_m[i][1]
            if i == self.num_seg - 1:
                theta_seg_m -= self.angles_m[0][0]
            else:
                theta_seg_m -= self.angles_m[i + 1][0]
        
        fourier_seg_all = np.asarray(fourier_seg_all)
        
        d = d.reshape(-1, 1)
        error = error.reshape(-1, 1)
        
        dislocation_all = np.zeros(self.num_seg)
        rotation_all = np.zeros(self.num_seg)
        
        theta_seg_m = np.pi
        
        for i in range(self.num_seg):
            
            theta_joints = [theta_seg_m + self.angles_m[i][1], theta_seg_m + self.angles_m[i][0]]
            
            param_last = fourier_seg_all[i, :]
            theta_joint_last = theta_joints[0] + self.angle_joint_width
            
            r_joint_last = param_last[-1]
            for j in range(k_fourier):
                r_joint_last += param_last[j] * np.cos((j + 1) * (theta_joint_last - theta_seg_m)) + param_last[k_fourier + j] * np.sin((j + 1) * (theta_joint_last - theta_seg_m))
            vector_joint_last = np.asarray([- np.sin(theta_joint_last) * r_joint_last, np.cos(theta_joint_last) * r_joint_last])
            for j in range(k_fourier):
                vector_joint_last[0] += np.cos(theta_joint_last) * (- param_last[j] * (j + 1) * np.sin((j + 1) * (theta_joint_last - theta_seg_m)) + param_last[k_fourier + j] * (j + 1) * np.cos((j + 1) * (theta_joint_last - theta_seg_m)))
                vector_joint_last[1] += np.sin(theta_joint_last) * (- param_last[j] * (j + 1) * np.sin((j + 1) * (theta_joint_last - theta_seg_m)) + param_last[k_fourier + j] * (j + 1) * np.cos((j + 1) * (theta_joint_last - theta_seg_m)))
            vector_joint_last = vector_joint_last / np.linalg.norm(vector_joint_last)
            
            if i == self.num_seg - 1:
                param_next = fourier_seg_all[0, :]
            else:
                param_next = fourier_seg_all[i + 1, :]
            theta_joint_next = theta_joints[0] - self.angle_joint_width
            
            r_joint_next = param_next[-1]
            for j in range(k_fourier):
                r_joint_next += param_next[j] * np.cos((j + 1) * (theta_joint_next - theta_seg_m)) + param_next[k_fourier + j] * np.sin((j + 1) * (theta_joint_next - theta_seg_m))
            vector_joint_next = np.asarray([- np.sin(theta_joint_next) * r_joint_next, np.cos(theta_joint_next) * r_joint_next])
            for j in range(k_fourier):
                vector_joint_next[0] += np.cos(theta_joint_next) * (- param_next[j] * (j + 1) * np.sin((j + 1) * (theta_joint_next - theta_seg_m)) + param_next[k_fourier + j] * (j + 1) * np.cos((j + 1) * (theta_joint_next - theta_seg_m)))
                vector_joint_next[1] += np.sin(theta_joint_next) * (- param_next[j] * (j + 1) * np.sin((j + 1) * (theta_joint_next - theta_seg_m)) + param_next[k_fourier + j] * (j + 1) * np.cos((j + 1) * (theta_joint_next - theta_seg_m)))
            vector_joint_next = vector_joint_next / np.linalg.norm(vector_joint_next)
            
            dislocation_all[i] = r_joint_last - r_joint_next
            
            rotation = np.arccos(np.dot(vector_joint_last, vector_joint_next))
            if np.cross(vector_joint_last, vector_joint_next) < 0:
                rotation = - rotation
            rotation_all[i] = rotation
                
            theta_seg_m += self.angles_m[i][1]
            if i == self.num_seg - 1:
                theta_seg_m -= self.angles_m[0][0]
            else:
                theta_seg_m -= self.angles_m[i + 1][0]
        
        xy_p_fourier_all = []
        
        theta_seg_m = np.pi
        
        for i in range(self.num_seg):
            
            theta_joints = [theta_seg_m + self.angles_m[i][1], theta_seg_m + self.angles_m[i][0]]
            
            param_fourier = fourier_seg_all[i, :]
            
            for theta_per in np.arange(theta_joints[0], theta_joints[1], step=np.pi/1800):
                r_per = param_fourier[-1]
                for j in range(k_fourier):
                    r_per += param_fourier[j] * np.cos((j + 1) * (theta_per - theta_seg_m)) + param_fourier[k_fourier + j] * np.sin((j + 1) * (theta_per - theta_seg_m))
                xy_p_fourier_all.append([r_per * np.cos(theta_per), r_per * np.sin(theta_per), i + 1])
                
            theta_seg_m += self.angles_m[i][1]
            if i == self.num_seg - 1:
                theta_seg_m -= self.angles_m[0][0]
            else:
                theta_seg_m -= self.angles_m[i + 1][0]
        
        xy_p_fourier_all = np.asarray(xy_p_fourier_all)
        xy_p_norm_all = np.hstack((xyz_p[:, 0:2], self.label.reshape(-1, 1)))
        
        xy_p_fourier_all_10 = copy.deepcopy(xy_p_fourier_all)
        d_fourier_all_10 = np.linalg.norm(xy_p_fourier_all_10[:, 0:2], axis=1)
        xy_p_fourier_all_10[:, 0] = ((d_fourier_all_10 - self.r) * 10 + self.r) * xy_p_fourier_all_10[:, 0] / d_fourier_all_10
        xy_p_fourier_all_10[:, 1] = ((d_fourier_all_10 - self.r) * 10 + self.r) * xy_p_fourier_all_10[:, 1] / d_fourier_all_10
        xy_p_fourier_all = np.hstack((xy_p_fourier_all, xy_p_fourier_all_10))
        
        xy_p_norm_all_10 = copy.deepcopy(xy_p_norm_all)
        d_norm_all_10 = np.linalg.norm(xy_p_norm_all_10[:, 0:2], axis=1)
        xy_p_norm_all_10[:, 0] = ((d_norm_all_10 - self.r) * 10 + self.r) * xy_p_norm_all_10[:, 0] / d_norm_all_10
        xy_p_norm_all_10[:, 1] = ((d_norm_all_10 - self.r) * 10 + self.r) * xy_p_norm_all_10[:, 1] / d_norm_all_10
        xy_p_norm_all = np.hstack((xy_p_norm_all, xy_p_norm_all_10))
        
        return xyz_p, d, error, dislocation_all, rotation_all, xy_p_norm_all, xy_p_fourier_all

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
    def fit_0(param_0, param_1, num_seg, r, length, width, angle_joint_width, angles_b, angles_m, angles_f, xyz_p, label):
        
        xyz_p = copy.deepcopy(xyz_p)
        
        delta_z = param_0[0]
        delta_theta = param_0[1]
        
        xyz_p[:, 2] = xyz_p[:, 2] + delta_z
        xyz_p = rotate_xy(xyz_p, delta_theta)
        
        param_seg = param_1
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
            
            boundary_l = np.interp(z_p_seg, [- length / 2, length / 2], [angles_b[i][1] + angle_joint_width, angles_f[i][1] + angle_joint_width])
            # xy_l = np.hstack((np.cos(theta_mc_seg + boundary_l).reshape(-1, 1), np.sin(theta_mc_seg + boundary_l).reshape(-1, 1))) * r
            boundary_u = np.interp(z_p_seg, [- length / 2, length / 2], [angles_b[i][0] - angle_joint_width, angles_f[i][0] - angle_joint_width])
            # xy_u = np.hstack((np.cos(theta_mc_seg + boundary_u).reshape(-1, 1), np.sin(theta_mc_seg + boundary_u).reshape(-1, 1))) * r
            
            index_out = np.where(delta_theta_mc_seg < boundary_l)[0]
            # error[index_out] = np.linalg.norm(xy_l[index_out, :] - xy_p_seg[index_out, :], axis=1)
            error[index_out] = delta_theta_mc_seg[index_out] - boundary_l[index_out]
            index_out = np.where(delta_theta_mc_seg > boundary_u)[0]
            # error[index_out] = np.linalg.norm(xy_p_seg[index_out, :] - xy_u[index_out, :], axis=1)
            error[index_out] = delta_theta_mc_seg[index_out] - boundary_u[index_out]
            
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
    
