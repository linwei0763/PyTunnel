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
        
        self.d_seg_circle = (xyz_p, d, error)
        
        return xyz_p, d, error
     
    def compute_d_seg_lin(self):
        
        if self.d_ellipse is not None:
            xyz_p = self.d_ellipse[0]
        else:
            xyz_p, _, _, _ = self.compute_d_ellipse()
        
        param = np.zeros(2)
        bounds = ([-0.1, -np.pi], [0.1, np.pi])
        
        param_ls = optimize.least_squares(Ring.fit_seg_angle, param, bounds=bounds, args=(self.num_seg, self.r, self.length, self.width, self.angles_b, self.angles_m, self.angles_f, xyz_p, self.label))
        param_ls = param_ls.x
        
        delta_z = param_ls[0]
        
        delta_theta = param_ls[1]
        print(delta_theta)
        
        param = np.zeros(2 * self.num_seg + 4)
        param[2 * self.num_seg + 2] = delta_z
        param[2 * self.num_seg + 3] = delta_theta
        
        # constraints = {'type': 'eq', 'fun': Ring.compute_seg_position, 'args': (self.num_seg, self.r, self.width, self.angles_m, True)}
        # constraints_0 = optimize.NonlinearConstraint(lambda x: Ring.compute_seg_position(x, self.num_seg, self.r, self.width, self.angles_m, True)[0], - 1e-4, 1e-4)
        # constraints_1 = optimize.NonlinearConstraint(lambda x: Ring.compute_seg_position(x, self.num_seg, self.r, self.width, self.angles_m, True)[1], - 1e-4, 1e-4)
        # constraints_2 = optimize.NonlinearConstraint(lambda x: Ring.compute_seg_position(x, self.num_seg, self.r, self.width, self.angles_m, True)[2], - 1e-4, 1e-4)
        # constraints = (constraints_0, constraints_1, constraints_2)
        
        lbs = []
        ubs = []
        for i in range(self.num_seg):
            lbs.append(-0.025)
            ubs.append(0.025)
        for i in range(self.num_seg):
            lbs.append(-np.pi / 72)
            ubs.append(np.pi / 72)
        for i in range(3):
            lbs.append(-0.1)
            ubs.append(0.1)
        lbs.append(-np.pi)
        ubs.append(np.pi)
        bounds = (lbs, ubs)
        
        # bounds = optimize.Bounds(bounds)

        # param = optimize.minimize(Ring.fit_seg, param, args=(self.num_seg, self.r, self.length, self.width, self.angles_b, self.angles_m, self.angles_f, xyz_p, self.label), method='trust-constr', constraints=constraints)
        # param = optimize.minimize(Ring.fit_seg, param, args=(self.num_seg, self.r, self.length, self.width, self.angles_b, self.angles_m, self.angles_f, xyz_p, self.label), bounds=bounds, constraints=constraints)
        # param = optimize.differential_evolution(Ring.fit_seg, bounds=bounds, args=(self.num_seg, self.r, self.length, self.width, self.angles_b, self.angles_m, self.angles_f, xyz_p, self.label), constraints=constraints)
        param_ls = optimize.least_squares(Ring.fit_seg, param, bounds=bounds, args=(self.num_seg, self.r, self.length, self.width, self.angles_b, self.angles_m, self.angles_f, xyz_p, self.label))
        print(param_ls.success)
        print(param_ls.message)
        param_ls = param_ls.x
        
        position_all = Ring.compute_seg_position(param_ls, self.num_seg, self.r, self.width, self.angles_m, False)
        xy_o_all = position_all[:, 0:2]
        
        dislocation_all = param_ls[0:self.num_seg]
        rotation_all = param_ls[self.num_seg:2 * self.num_seg]
        delta_xy = param_ls[2 * self.num_seg:2 * self.num_seg + 2]
        delta_theta = param_ls[2 * self.num_seg + 3]
        
        norm_xy_p = xyz_p[:, 0:2] + delta_xy
        norm_xy_p = rotate_xy(norm_xy_p, delta_theta)
        
        d = np.zeros(self.xyz.shape[0])
        error = np.zeros(self.xyz.shape[0])
        
        for i in range(self.num_seg):
            
            index = np.where(self.label == i + 1)[0]
            norm_xy_p_seg = norm_xy_p[index, :]
            norm_xy_p_seg = norm_xy_p_seg - xy_o_all[i]
            norm_theta_seg = np.arctan2(norm_xy_p_seg[:, 1], norm_xy_p_seg[:, 0])
            xy_circle_seg = np.hstack((self.r * np.cos(norm_theta_seg).reshape(-1, 1), self.r * np.sin(norm_theta_seg).reshape(-1, 1)))
            error_seg = np.linalg.norm(norm_xy_p[index, :], axis=1) - np.linalg.norm(xy_circle_seg, axis=1)
            error[index] = error_seg[:]
            
            xy_circle_seg = xy_circle_seg + xy_o_all[i]
            d_seg = np.linalg.norm(xy_circle_seg, axis=1) - self.r
            d[index] = d_seg[:]
        
        d = d.reshape(-1, 1)
        error = error.reshape(-1, 1)
        
        return xyz_p, d, error, dislocation_all, rotation_all
    
    
    @staticmethod
    def compute_error(param_ring, param_seg, num_seg, r, length, width, angles_b, angles_m, angles_f, xyz_p, label):
        
        delta_xy = param_ring[0:2]
        delta_z = param_ring[2]
        delta_theta = param_ring[3]
        
        xyz_p[:, 0:2] = xyz_p[:, 0:2] + delta_xy
        xyz_p[:, 2] = xyz_p[:, 2] + delta_z
        xyz_p = rotate_xy(xyz_p, delta_theta)
        
        positioan_all = Ring.compute_seg_position(param_seg, num_seg, r, width, angles_m, False)
        
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
            
            boundary_down = np.interp(z_p_seg, [- length / 2, length / 2], [angles_b[i][1], angles_f[i][1]])
            xy_down = np.hstack((np.cos(theta_mc_seg + boundary_down).reshape(-1, 1), np.sin(theta_mc_seg + boundary_down).reshape(-1, 1))) * r
            # print(xy_down)
            boundary_up = np.interp(z_p_seg, [- length / 2, length / 2], [angles_b[i][0], angles_f[i][0]])
            xy_up = np.hstack((np.cos(theta_mc_seg + boundary_up).reshape(-1, 1), np.sin(theta_mc_seg + boundary_up).reshape(-1, 1))) * r
            
            # print((delta_theta_mc_seg - boundary_down).shape[0])
            # print(boundary_down.shape[0])
            
            index_out = np.where(delta_theta_mc_seg < boundary_down)[0]
            # print(delta_theta_mc_seg.shape)
            # print(boundary_down.shape)
            # print(index_out)
            # print(index_out.shape[0])
            error[index_out] = np.linalg.norm(xy_p_seg[index_out, :] - xy_down[index_out, :], axis=1)
            # print(index_out.shape[0])
            index_out = np.where((delta_theta_mc_seg - boundary_up) > 0)[0]
            error[index_out] = np.linalg.norm(xy_p_seg[index_out, :] - xy_up[index_out, :], axis=1)
            # print(index_out.shape[0])
            index_in = np.where(((delta_theta_mc_seg - boundary_down) >= 0) & ((delta_theta_mc_seg - boundary_up) <= 0))[0]
            error[index_in] = np.linalg.norm(xy_p_seg[index_in, :], axis=1) - r
            # print(index_in.shape[0])
            # print(error[index_in])
            
            error_all[index] = error[:]
            
        # error_all_2 = np.power(error_all, 2)
        # error_all_2 = np.mean(error_all_2)
        # print(error_all_2)
            
        return error_all
    
    
    @staticmethod
    def compute_seg_position(param, num_seg, r, width, angles_m, flag_constrain):
        
        dislocation_all = param[0:num_seg]
        rotation_all = param[num_seg:2 * num_seg]
        
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
            
            if i == num_seg:
                theta_mc_1 = theta_dislocation - angles_m[0][0] + rotation_all[i - 1]                
            else:
                theta_mc_1 = theta_dislocation - angles_m[i][0] + rotation_all[i - 1]
            
            position_all[i, 0:2], position_all[i, 2] = xy_o_1[:], theta_mc_1
        
        if flag_constrain:
            delta_vector = np.asarray([np.cos(np.pi), np.sin(np.pi)]) - np.asarray([np.cos(position_all[num_seg, 2]), np.sin(position_all[num_seg, 2])])
            delta_distance = np.linalg.norm(delta_vector)
            vector_cons = np.zeros(3)
            vector_cons[0:2], vector_cons[2] = position_all[num_seg, 0:2], delta_distance
            # print(vector_cons)
            return vector_cons
        else:
            return position_all[0:num_seg, :]
    
    
    @staticmethod
    def fit_seg_angle(param, num_seg, r, length, width, angles_b, angles_m, angles_f, xyz_p, label):
        
        delta_z = param[0]
        delta_theta = param[1]
        
        fake_param = np.zeros(2 * num_seg + 4)
        fake_param[2 * num_seg + 2] = delta_z
        fake_param[2 * num_seg + 3] = delta_theta
        
        error_all = Ring.fit_seg(fake_param, num_seg, r, length, width, angles_b, angles_m, angles_f, xyz_p, label)
        
        return error_all    
    
