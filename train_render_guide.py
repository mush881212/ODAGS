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

import os
# os.environ['CUDA_VISIBLE_DEVICES'] ='0'
import torch
import torchvision
# from torchmetrics import PearsonCorrCoef
# from pynvml import *
import random
from random import randint
from utils.loss_utils import l1_loss, ssim, corr_loss, sparse_loss, edge_aware_logl1_loss, edge_aware_TV_loss, TV_loss, Sobel
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, colorize
from utils.graphics_utils import getdepth, getpointxy
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import math

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

log = False
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    max_psnr = 0

    ### log here ###
    if log:
        os.makedirs(f"{scene.model_path}/log", exist_ok=True)
        f_img_idx = open(f"{scene.model_path}/log/img_log.txt", 'w')
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        np.save(os.path.join(scene.model_path, "log", f"0_splat.npy"), xyz)
    ###############
    # add weight decade
    can_back_proj = False

    
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        is_reload = False
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            is_reload = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        # [0413] add densify from rgb-guidance
        if iteration >= opt.propagated_iteration_begin and iteration <= opt.propagated_iteration_after and (iteration % opt.propagation_interval == 0):
                can_back_proj = True
        if can_back_proj:
            if is_reload:
                start_back_proj = True
        if start_back_proj:
            with torch.no_grad():
                print(f"start back proj: {iteration}")
                print(len(viewpoint_stack))
                psnrs = []
                for i in range(0, len(viewpoint_stack), 10):
                    psnrs.append(viewpoint_stack[i].psnr)
                average_psnr = sum(psnrs)/len(psnrs)
                std = math.sqrt(sum([(x - average_psnr) ** 2 for x in psnrs]) / len(psnrs))
                # std
                print(f"Average psnr {average_psnr} std {std}")

                for i in range(0, len(viewpoint_stack), 20):
                    view_cam = viewpoint_stack[i]
                    render_pkg = render(view_cam, gaussians, pipe, bg)
                    image, render_depth, alpha = render_pkg["render"], render_pkg["depth_map"], render_pkg["alpha"]
                    gt_image = view_cam.original_image.cuda()
                    rgb_abs_error = torch.mean(torch.abs(image - gt_image), dim=0)
                    
                    valid_mask = (render_depth > opt.depth_tolerance).squeeze(0)
                    # rgb_densify_mask = (rgb_abs_error > opt.rgb_densify_threshold_min) & (rgb_abs_error <= opt.rgb_densify_threshold_max) # [H, W]
                    rgb_densify_mask = rgb_abs_error > opt.rgb_densify_threshold_min # [H, W]
                    rgb_densify_mask = rgb_densify_mask & valid_mask
                    # edge_mask = sobel(gt_image) < 0.3 # add edge as condition
                    # rgb_densify_mask = rgb_densify_mask & edge_mask
                    rgb_abs_error[~rgb_densify_mask] = 0.0

                    torchvision.utils.save_image(image, os.path.join(scene.model_path, f"{iteration}_{view_cam.image_name}_render.png"))
                    if view_cam.psnr < (average_psnr - 3*std):
                        # print(view_cam.psnr)
                        continue
                    torchvision.utils.save_image(rgb_densify_mask.float(), os.path.join(scene.model_path, f"{iteration}_{view_cam.image_name}_mask.png"))
                    gaussians.accum_from_depth_projection_v2(view_cam, render_depth.squeeze(0), rgb_abs_error, gt_image, path=scene.model_path, iteration=iteration, idx=view_cam.image_name, render_error_thres=opt.rgb_densify_threshold_min, extent=scene.cameras_extent * opt.rgb_densify_thres)

                gaussians.back_proj(iteration, path=scene.model_path)
                can_back_proj = False
                start_back_proj = False
                print(f"Finish back proj {iteration}")

        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        if log:
            f_img_idx.write(viewpoint_cam.image_name+"\n")

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii, render_depth, alpha = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth_map"], render_pkg["alpha"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        viewpoint_cam.psnr = psnr(image, gt_image).mean().float().detach()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            test_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration % saving_iterations[0] == 0) or (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if test_psnr > max_psnr:
                print("\n[ITER {}] Saving Gaussians with Max PSNR".format(iteration))
                scene.save(0) # save max gaussian at the first scene

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                
                # [0401] add manual depth and point_xy computation
                depth_manual = getdepth(gaussians.get_xyz, viewpoint_cam.world_view_transform)
                point_xy_manual = getpointxy(gaussians.get_xyz, viewpoint_cam.full_proj_transform, viewpoint_cam.image_width, viewpoint_cam.image_height)
                
                # [0317] add occlusion [0606] add depth scaling
                if iteration > opt.start_occlusion_test:
                    grad, occlusion_mask = gaussians.add_densification_stats_margin(viewspace_point_tensor, visibility_filter, depth_manual, point_xy_manual, render_depth.squeeze(0), 0.15*scene.cameras_extent)
                else:
                    grad = gaussians.add_densification_stats_org(viewspace_point_tensor, visibility_filter)
                if log:
                    occlusion_mask = occlusion_mask.detach().cpu().numpy()
                    visibility_idx = np.where(occlusion_mask==1)[0].astype(np.uint32)
                    np.save(os.path.join(scene.model_path, "log", f"{iteration}_filter.npy"), visibility_idx)
                    np.save(os.path.join(scene.model_path, "log", f"{iteration}_grad.npy"), grad.detach().cpu().numpy())

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_grad_abs_threshold, 0.005, scene.cameras_extent, size_threshold, iteration, path = scene.model_path)

                    if log:
                        xyz = gaussians.get_xyz.detach().cpu().numpy()
                        np.save(os.path.join(scene.model_path,"log",  f"{iteration}_splat.npy"), xyz)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            # [0401] add pruning
            if (iteration > opt.prune_opacity_start) & (iteration % opt.prune_opacity_interval == 0):
                gaussians.prune_opacity(0.005)
            
            # [0607] add needle_prune
            if (iteration > opt.prune_needle_start) & (iteration % opt.prune_needle_interval == 0):
                gaussians.prune_needle(scene.cameras_extent)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations) or (iteration % checkpoint_iterations[0] == 0):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            if test_psnr > max_psnr:
                max_psnr = test_psnr
                print("\n[ITER {}] Saving Checkpoint with Max PSNR".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + "_best" + ".pth") 
    
    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    if log:
        f_img_idx.close()

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        print("Logging progress to Tensorboard")
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    cur_test_psnr = 0
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations or iteration % testing_iterations[0] == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        cur_test_psnr = 0
        for config in validation_configs:
            # if config["name"] == "train":
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                # l1_depth = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    depth = torch.clamp(colorize(render_pkg["depth_map"]), 0.0, 1.0)
                    # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # gt_depth = torch.clamp(colorize(viewpoint.depth.to("cuda")), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            # tb_writer.add_images(config['name'] + "_view_{}/ground_truth_depth".format(viewpoint.image_name), gt_depth[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    # l1_depth += depth_loss(render_pkg["depth_map"], viewpoint.depth.to("cuda")).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                if config['name'] == 'test':
                    cur_test_psnr = psnr_test
                l1_test /= len(config['cameras'])
                # l1_depth /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    # tb_writer.add_scalar(config['name'] + '/loss_viewpoint - depth_l1_loss', l1_depth, iteration)
                    # tb_writer.add_scalar(config['name'] + '/loss_viewpoint - sparse_loss', Lsparse, iteration)
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
    return cur_test_psnr

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1000_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--n_init_points", type=int, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    random.seed(20240307)
    torch.manual_seed(20240307)
    torch.cuda.manual_seed_all(20240307)
    np.random.seed(20240307)
    os.environ['PYTHONHASHSEED'] = str(20240307)

    training(lp.extract(args), op.extract(args), pp.extract(args), \
            args.test_iterations, args.save_iterations, args.checkpoint_iterations, \
            args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
