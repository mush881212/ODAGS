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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def getdepth(xyz, view_matrix):
    ones = torch.ones((xyz.shape[0], 1), device=xyz.device)
    xyz = torch.cat((xyz, ones), dim=1)
    depth = torch.matmul(xyz, view_matrix)
    depth = depth[:, 2]
    return depth

def getpointxy(xyz, proj_view_matrix, width, height):
    ones = torch.ones((xyz.shape[0], 1), device=xyz.device)
    xyz = torch.cat((xyz, ones), dim=1)
    p_hom = torch.matmul(xyz, proj_view_matrix)
    p_w = 1.0 / (p_hom[:, 3:4] + 0.0000001)
    p_hom = p_hom * p_w
    p_hom_x = ((p_hom[:, 0]+ 1.0) * width - 1.0) * 0.5
    p_hom_y = ((p_hom[:, 1] + 1.0) * height - 1.0) * 0.5
    point_x = torch.clamp(p_hom_x, min = 0, max = width - 1)
    point_y = torch.clamp(p_hom_y, min=0, max = height - 1)
    point_xy = torch.cat([point_x.unsqueeze(-1), point_y.unsqueeze(-1)], dim=-1)
    return point_xy
    