import numpy as np
from scipy import optimize

from module.utils import compute_normal, fit_circle, fit_circle_v, fit_ellipse_v, project2plane, rotate_xy, solve_contradiction


class Ring():
    
    def __init__(self, xyz, intensity, label, r, length, width, num_seg, angles_b, angles_m, angles_f, v0_dir):
        
        self.offset = np.mean(xyz, axis=0)
        
        self.xyz = xyz - self.offset
        self.intensity = intensity
        self.label = label
        
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
        
        param = np.zeros(2 * self.num_seg + 4)
        
        cons = {'type': 'eq', 'fun': Ring.compute_seg_position, 'args': (self.num_seg, self.r, self.width, self.angles_m, True)}

        param = optimize.minimize(Ring.fit_seg, param, args=(self.num_seg, self.r, self.length, self.width, self.angles_b, self.angles_m, self.angles_f, xyz_p, self.label), method='trust-constr', constraints=cons)
        print(param.success)
        print(param.message)
        param = param.x
        
        position_all = Ring.compute_seg_position(param, self.num_seg, self.r, self.width, self.angles_m, False)
        xy_o_all = position_all[:, 0:2]
        
        dislocation_all = param[0:self.num_seg]
        rotation_all = param[self.num_seg:2 * self.num_seg]
        delta_xy = param[2 * self.num_seg:2 * self.num_seg + 2]
        delta_theta = param[2 * self.num_seg + 2]
        
        norm_xy_p = xyz_p[:, 0:2] + delta_xy
        norm_xy_p = rotate_xy(norm_xy_p, delta_theta)
        
        d = np.zeros(norm_xy_p.shape[0])
        error = np.zeros(norm_xy_p.shape[0])
        
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
    def compute_seg_position(param, num_seg, r, width, angles_m, flag_constrain):
        
        dislocation_all = param[0:num_seg]
        rotation_all = param[num_seg:2 * num_seg]
        
        position_all = np.zeros((num_seg + 1, 3))
        position_all[0, 2] = np.pi
        
        for i in range(1, num_seg + 1):
            
            xy_o_0, theta_mc_0 = position_all[i - 1, 0:2], position_all[i - 1, 2]
            xy_o_1, theta_mc_1 = np.zeros(2), 0
            
            theta_dislocation = theta_mc_0 + angles_m[i - 1][1]
            xy_o_1 = xy_o_0 - dislocation_all[i - 1] * np.hstack((np.cos(theta_dislocation).reshape(-1, 1), np.sin(theta_dislocation).reshape(-1, 1)))
                                                                 
            rotation_centre = xy_o_0 + (r + width / 2 - dislocation_all[i - 1]) * np.hstack((np.cos(theta_dislocation).reshape(-1, 1), np.sin(theta_dislocation).reshape(-1, 1)))
            xy_o_1 = xy_o_1 - rotation_centre
            xy_o_1 = xy_o_1.reshape(1, 2)
            xy_o_1 = rotate_xy(xy_o_1, rotation_all[i - 1])
            xy_o_1 = xy_o_1.reshape(2)
            xy_o_1 = xy_o_1 + rotation_centre
            
            if i == num_seg:
                theta_mc_1 = theta_dislocation - angles_m[0][0] + rotation_all[i - 1]
                # while theta_mc_1 <= -np.pi:
                #     theta_mc_1 += 2 * np.pi
                # while theta_mc_1 > np.pi:
                #     theta_mc_1 -= 2 * np.pi                    
            else:
                theta_mc_1 = theta_dislocation - angles_m[i][0] + rotation_all[i - 1]
            
            position_all[i, 0:2], position_all[i, 2] = xy_o_1[:], theta_mc_1
        
        if flag_constrain:
            delta_vector = np.asarray([np.cos(np.pi), np.sin(np.pi)]), np.asarray([np.cos(position_all[num_seg, 2]), np.sin(position_all[num_seg, 2])])
            delta_theta = np.linalg.norm(delta_vector)
            vector_cons = np.zeros(3)
            vector_cons[0:2], vector_cons[2] = position_all[num_seg, 0:2], delta_theta
            return vector_cons
        else:
            return position_all[0:num_seg, :]
    
    
    @staticmethod
    def fit_seg(param, num_seg, r, length, width, angles_b, angles_m, angles_f, xyz_p, label):
        
        delta_xy = param[2 * num_seg:2 * num_seg + 2]
        delta_theta = param[2 * num_seg + 2]
        delta_z = param[2 * num_seg + 3]
        
        xyz_p[:, 0:2] = xyz_p[:, 0:2] + delta_xy
        xyz_p[:, 2] = xyz_p[:, 2] + delta_z
        xyz_p = rotate_xy(xyz_p, delta_theta)
        
        positioan_all = Ring.compute_seg_position(param[0:2 * num_seg], num_seg, r, width, angles_m, False)
        
        error_all = np.zeros(xyz_p.shape[0])
        
        for i in range(num_seg):
            
            index = np.where(label == i + 1)
            index = index[0]
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
            delta_theta_mc_seg = np.arccos(np.dot(vector_theta_seg, vector_mc_seg.T))
            flag_delta_theta_mc_seg = np.cross(vector_mc_seg, vector_theta_seg)
            index_flag = np.where(flag_delta_theta_mc_seg < 0)[0]
            delta_theta_mc_seg[index_flag] = - delta_theta_mc_seg[index_flag]
            
            boundary_down = np.interp(z_p_seg, [- length / 2, length / 2], [angles_b[i][1], angles_f[i][1]])
            xy_down = np.vstack((np.cos(theta_mc_seg + boundary_down).reshape(-1, 1), np.sin(theta_mc_seg + boundary_down).reshape(-1, 1)))
            boundary_up = np.interp(z_p_seg, [- length / 2, length / 2], [angles_b[i][0], angles_f[i][0]])
            xy_up = np.vstack((np.cos(theta_mc_seg + boundary_up).reshape(-1, 1), np.sin(theta_mc_seg + boundary_up).reshape(-1, 1)))
            
            index_out = np.where(delta_theta_mc_seg - boundary_down < 0)[0]
            error[index_out] = np.linalg.norm(xy_p_seg[index_out, :] - xy_down[index_out, :], axis=1)
            index_out = np.where(delta_theta_mc_seg - boundary_up > 0)[0]
            error[index_out] = np.linalg.norm(xy_p_seg[index_out, :] - xy_up[index_out, :], axis=1)
            index_in = np.where(error == 0)[0]
            error[index_in] = np.linalg.norm(xy_p_seg[index_in, :], axis=1) - r
            
            error_all[index] = error[:]
            
            error_2 = np.power(error_all, 2)
            error_2 = np.sum(error_2)
            
        return error_2
        

            
            
        
            
        
        
        
    
    
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