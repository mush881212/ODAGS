#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.back_proj_points = torch.empty(0).cuda()
        self.back_proj_color = torch.empty(0).cuda()
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum,
        xyz_gradient_accum_abs, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, max_grad_abs, min_opacity, extent, max_screen_size, iter, path=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads_abs, max_grad_abs, extent)
        # self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size # 檢查渲染出的splat大小是否大於20 pixel
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent # 檢查splat的大小(經過exp function)是否大於10%的場景大小
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        
    def prune_opacity(self, min_opacity):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
        
    def prune_needle(self, extent, needle_thres):
        scales = self.get_scaling
        max_scale = scales.max(dim=1).values

        scale_factor_x = (max_scale/scales[:, 0])
        scale_factor_y = (max_scale/scales[:, 1])
        scale_factor_z = (max_scale/scales[:, 2])
        
        # plate-like pruning
        large_max_scale = max_scale > (10 * needle_thres*extent)
        needle_mask = large_max_scale
        # needle-like pruning
        is_needle_x = (scale_factor_x > 30) | (scale_factor_x == 1.0)
        is_needle_y = (scale_factor_y > 30) | (scale_factor_y == 1.0)
        is_needle_z = (scale_factor_z > 30) | (scale_factor_z == 1.0)
        large_max_scale = max_scale > (needle_thres*extent)
        needle_mask_2 = is_needle_x & is_needle_y & is_needle_z & large_max_scale
        needle_mask = torch.logical_or(needle_mask, needle_mask_2)
        
        self.prune_points(needle_mask)
        torch.cuda.empty_cache()

    
    def add_densification_stats_org(self, viewspace_point_tensor, update_filter):
        grad = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum[update_filter] += grad
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        return grad
        
    def add_densification_stats_margin(self, viewspace_point_tensor, update_filter, depths, point_xy, depth_map, margin):

        point_xy = point_xy[update_filter] # visible and in image range
        point_xy =  torch.floor(point_xy).long()
        
        depth_map_values = depth_map[point_xy[:, 1], point_xy[:, 0]]
        depth_values = depths[update_filter]
        
        occlusion_mask = torch.zeros(viewspace_point_tensor.shape[0], device="cuda", dtype=bool)
        occlusion_mask[update_filter] = (depth_values <= (depth_map_values + margin)) # copy the occlusion mask to the visible points
        
        grad = torch.norm(viewspace_point_tensor.grad[occlusion_mask,:2], dim=-1, keepdim=True)
        grad_abs = torch.norm(viewspace_point_tensor.grad[occlusion_mask, 2:], dim=-1, keepdim=True)

        self.xyz_gradient_accum[occlusion_mask] += grad
        self.xyz_gradient_accum_abs[occlusion_mask] += grad_abs

        self.denom[occlusion_mask] += 1
        return grad, occlusion_mask
    
    def accum_from_depth_projection_v2(self, viewpoint_cam, depth, error_matrix, gt_image, iteration=0, idx=0,  path=None, render_error_thres=0.1, extent=0.01):
        # inverse project pixels into 3D scenes
        K = viewpoint_cam.K
        cam2world = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
        height, width = depth.shape # Get the shape of the depth image
        # Create a grid of 2D pixel coordinates
        y, x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        # Stack the 2D and depth coordinates to create 3D homogeneous coordinates
        coordinates = torch.stack([x.to(depth.device), y.to(depth.device), torch.ones_like(depth)], dim=-1)
        coordinates = coordinates.view(-1, 3).to(K.device).to(torch.float32)
        # Reproject the 2D coordinates to 3D coordinates
        coordinates_3D = (K.inverse() @ coordinates.T).T
        coordinates_3D *= depth.view(-1, 1)
        world_coordinates_3D = (cam2world[:3, :3] @ coordinates_3D.T).T + cam2world[:3, 3]

        # Get the shape of the error matrix
        height, width = error_matrix.shape
        block_size = 8
        # padding
        pad_y = (block_size - (height % block_size)) % block_size
        pad_x = (block_size - (width % block_size)) % block_size
        error_matrix = torch.nn.functional.pad(error_matrix, (0, pad_x, 0, pad_y), mode='constant', value=0.0)

        # Reshape the error matrix to have an extra dimension for blocks
        blocks = error_matrix.view(height // block_size, block_size, width // block_size, block_size)
        # Move the block dimensions to the front and merge them
        blocks = blocks.permute(0, 2, 1, 3).contiguous().view(-1, block_size, block_size)
        # Find the index of the maximum error in each block
        max_error_indices = torch.argmax(blocks.view(blocks.size(0), -1), dim=1) # indices: 0 ~ 63
        max_error_values = torch.max(blocks.view(blocks.size(0), -1), dim=1).values
        valid_block = max_error_values > render_error_thres
        # torchvision.utils.save_image(blocks[0], "block.png")

        max_error_indices_x = max_error_indices % block_size
        max_error_indices_y = max_error_indices // block_size
        max_error_coords = torch.stack((max_error_indices_y, max_error_indices_x), dim=1)

        # Compute the global coordinates
        y_indices = torch.arange(0, height, block_size, device=depth.device).repeat_interleave(width // block_size)
        x_indices = torch.arange(0, width, block_size, device=depth.device).repeat(height // block_size)
        global_coords = torch.stack((y_indices, x_indices), dim=1) + max_error_coords
        global_coords = global_coords[valid_block]
        global_coords = global_coords[:, 0] * width + global_coords[:, 1]

        # Reshape the result to match the block grid layout
        fused_point_cloud = world_coordinates_3D[global_coords]
        color_downsampled = gt_image.permute(1, 2, 0).view(-1, 3)[global_coords]

        # initialize gaussians
        fused_color = RGB2SH(color_downsampled)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).to(fused_color.device)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        all_points = torch.concat([fused_point_cloud, self.back_proj_points, self.get_xyz], dim=0)
        dist2 = torch.clamp_min(distCUDA2(all_points), 0.0000001)[:fused_point_cloud.shape[0]]
        
        if path:
            os.makedirs(os.path.join(path, "ply_i"), exist_ok=True)
            np.save(os.path.join(path, "ply_i", f"{iteration}_{idx}.npy"), fused_point_cloud.cpu().numpy())
            # dist_map = torch.zeros((height*width), device=depth.device)
            # dist2_norm = dist2 / torch.max(dist2)
            # dist_map[global_coords] = dist2_norm
            # dist_map = dist_map.view(1, height, width)
            # torchvision.utils.save_image(colorize(dist_map), os.path.join(path,f"{iteration}_{idx}_dist.png"))
            np.save(os.path.join(path, "ply_i", f"{iteration}_{idx}_sparsity.npy"), dist2.cpu().numpy())

        # sparsity_mask = dist2 > extent
        # self.back_proj_points = torch.concat([self.back_proj_points, fused_point_cloud[sparsity_mask]], dim=0)
        # self.back_proj_color = torch.concat([self.back_proj_color, features[sparsity_mask]], dim=0)
        # print(f"fuse point: {fused_point_cloud.shape}->{fused_point_cloud[sparsity_mask].shape}")
        # print(f"fuse color: {features.shape}->{features[sparsity_mask].shape}")
        # print(f"total point: {self.back_proj_points.shape}")
        # print(f"total color: {self.back_proj_color.shape}")
        self.back_proj_points = torch.concat([self.back_proj_points, fused_point_cloud], dim=0)
        self.back_proj_color = torch.concat([self.back_proj_color, features], dim=0)
        print(f"fuse point: {fused_point_cloud.shape}")
        print(f"fuse color: {features.shape}")
        print(f"total point: {self.back_proj_points.shape}")
        print(f"total color: {self.back_proj_color.shape}")
        torch.cuda.empty_cache()
       
    def back_proj(self, iteration, opacity=0.1, path=None):
        original_point_cloud = self.get_xyz
        fused_shape = self.back_proj_points.shape[0]
        if fused_shape == 0:
            print(f"No splat to add")
            return
        all_point_cloud = torch.concat([self.back_proj_points, original_point_cloud], dim=0)
        all_dist2 = torch.clamp_min(distCUDA2(all_point_cloud), 0.0000001)
        dist2 = all_dist2[:fused_shape]
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((self.back_proj_points.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(opacity * torch.ones((self.back_proj_points.shape[0], 1), dtype=torch.float, device="cuda"))
        print(f"adding fused point {self.back_proj_points.shape} + {original_point_cloud.shape} -> {all_point_cloud.shape}")
        if path:
            os.makedirs(os.path.join(path, "ply"), exist_ok=True)
            np.save(os.path.join(path, "ply", f"fused_point_cloud_{iteration}.npy"), self.back_proj_points.cpu().numpy())

        new_xyz = nn.Parameter(self.back_proj_points.requires_grad_(True))
        new_features_dc = nn.Parameter(self.back_proj_color[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(self.back_proj_color[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        #update gaussians
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        self.back_proj_points = torch.empty(0).cuda()
        self.back_proj_color = torch.empty(0).cuda()
        torch.cuda.empty_cache()
        