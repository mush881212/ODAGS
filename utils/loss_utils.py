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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn
from torchvision import transforms as T

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
        self.blur_op = T.GaussianBlur(3, sigma=(0.1, 2.0))
        self.max_pooling = nn.MaxPool2d(5, stride=1, padding=2)
        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1).cuda()
        self.filter.weight = nn.Parameter(G, requires_grad=False)
    
    def rgb_to_gray(self, rgb):
        rgb = rgb.permute(1, 2, 0)
        return torch.sum(rgb * torch.tensor([0.2989, 0.5870, 0.1140], device=rgb.device), dim=2)

    def forward(self, img):
        img = self.rgb_to_gray(img)
        img = self.blur_op(img.unsqueeze(0))
        img = F.pad(img.unsqueeze(0), (1, 1, 1, 1), mode='reflect')

        x = self.filter(img.float())
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        x = self.max_pooling(x)
        return x.squeeze(0).squeeze(0)
    

class Sobel_grad(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
        self.blur_op = T.GaussianBlur(3, sigma=(0.1, 2.0))
        self.max_pooling = nn.MaxPool2d(5, stride=1, padding=2)
        Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1).cuda()
        self.filter.weight = nn.Parameter(G, requires_grad=False)
    
    def rgb_to_gray(self, rgb):
        rgb = rgb.permute(1, 2, 0)
        return torch.sum(rgb * torch.tensor([0.2989, 0.5870, 0.1140], device=rgb.device), dim=2)

    def forward(self, img):
        img = self.rgb_to_gray(img)
        img = F.pad(img.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')

        x = self.filter(img.float())
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        # x = self.max_pooling(x)
        return x.squeeze(0)

def grad_sigmoid(grad):
    return 1.0 / (1.0 + torch.exp(-10.0 *grad))

def edge_aware_logl1_loss_v2(pred, gt, rgb, mask):
    pred, gt, rgb, mask = pred.permute(1, 2, 0), gt.permute(1, 2, 0), rgb.permute(1, 2, 0), mask.permute(1, 2, 0)
    logl1 = torch.log(1 + torch.abs(pred - gt))
    grad_img_x = torch.mean(
        torch.abs(rgb[ :, :-1, :] - rgb[ :, 1:, :]), -1, keepdim=True
    )
    grad_img_y = torch.mean(
        torch.abs(rgb[ :-1, :, :] - rgb[ 1:, :, :]), -1, keepdim=True
    )
    lambda_x = torch.exp(-grad_img_x)
    lambda_y = torch.exp(-grad_img_y)
    lambda_xy = lambda_x + lambda_y
    loss = lambda_xy * logl1
    loss = loss[mask]
    return loss.mean()

def edge_aware_logl1_loss(pred, gt, rgb, mask):
    pred, gt, rgb, mask = pred.permute(1, 2, 0), gt.permute(1, 2, 0), rgb.permute(1, 2, 0), mask.permute(1, 2, 0)
    # print(pred.shape, gt.shape, rgb.shape, mask.shape)

    logl1 = torch.log(1 + torch.abs(pred - gt))
    grad_img_x = torch.mean(torch.abs(rgb[ :, :-1, :] - rgb[ :, 1:, :]), -1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(rgb[ :-1, :, :] - rgb[ 1:, :, :]), -1, keepdim=True)
    lambda_x = torch.exp(-grad_img_x)
    lambda_y = torch.exp(-grad_img_y)

    loss_x = lambda_x * logl1[ :, :-1, :]
    loss_y = lambda_y * logl1[ : -1, :, :]
    
    if mask is not None:
        assert mask.shape[:2] == pred.shape[:2]
        loss_x = loss_x[mask[ :, :-1, :]]
        loss_y = loss_y[mask[ :-1, :, :]]

    return loss_x.mean() + loss_y.mean()

def TV_loss(pred):
    pred = pred.permute(1, 2, 0)
    # pred = [batch, H, W, 1]
    h_diff = pred[:, :-1, :] - pred[:, 1:, :]
    w_diff = pred[:-1, :, :] - pred[1:, :, :]
    return torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))

def TV_pixel_loss(pred):
    # print(pred.shape)
    pred_h = F.pad(pred, (1, 0, 0, 0), mode='reflect')
    pred_w = F.pad(pred, (0, 0, 1, 0), mode='reflect')
    pred_h = pred_h.permute(1, 2, 0)
    pred_w = pred_w.permute(1, 2, 0)
    h_diff = pred_h[:, :-1, :] - pred_h[:, 1:, :]
    w_diff = pred_w[:-1, :, :] - pred_w[1:, :, :]
    return torch.abs(h_diff) + torch.abs(w_diff)

def edge_aware_TV_loss(depth, rgb):
    
    grad_depth_x = torch.abs(depth[:, :-1, :] - depth[:, 1:, :])
    grad_depth_y = torch.abs(depth[:-1, :, :] - depth[1:, :, :])

    grad_img_x = torch.mean(torch.abs(rgb[:, :-1, :] - rgb[:, 1:, :]), -1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(rgb[ :-1, :, :] - rgb[ 1:, :, :]), -1, keepdim=True)

    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)

    return grad_depth_x.mean() + grad_depth_y.mean()

def corr_loss(pred, target):
    pred = pred.reshape(1, -1)
    target = target.reshape(1, -1)

    pred_centred = pred - torch.unsqueeze(torch.mean(pred, 1), 1)
    target_centred = target - torch.unsqueeze(torch.mean(target, 1), 1)
    
    pred_std = torch.unsqueeze(torch.sqrt(torch.mean(pred_centred ** 2, 1)), 1)
    target_std = torch.unsqueeze(torch.sqrt(torch.mean(target_centred ** 2, 1)), 1)
    
    corr_torch = pred_centred * target_centred / (pred_std * target_std)
    return 1.0 - corr_torch.mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_pixel_loss(network_output, gt):
    return torch.abs((network_output - gt))

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

