# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

from PoseExpNet import PoseExpNet

from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False



def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
       rays_flat: [b, 3+3+1+1] o,d,near,far
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret: # k must be the key value
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret} 
    return all_ret

# ndc-Normalized Device Coordinate
def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True) # 归一�?        
        viewdirs = torch.reshape(viewdirs, [-1,3]).float() # [b, 3]

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:]) # list(sh[:-1])-batch_size, 
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # k_extract = ['rgb_map', 'disp_map', 'acc_map']
    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'pt_map', 'disp_mask'] # revised by shu chen, 2022/4/22
    ret_list = [all_ret[k] for k in k_extract] # list
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, pt, disp_mask, extras = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)  
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args, pose_net):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)  

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 4  
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    # optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    if args.pose_extract:
        optim_params = [
            {'params': grad_vars, 'lr': args.lrate},
            {'params': pose_net.parameters(), 'lr': args.pose_lrate, 'weight_decay': args.weight_decay}
        ]
        if args.netpose_freeze:
            optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
        else:
            optimizer = torch.optim.Adam(optim_params, betas=(0.9, 0.999)) #,weight_decay=args.weight_decay
    else:
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    pose_start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        pose_start = ckpt['pose_global_step']
        
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 2000
        new_lrate = args.lrate * (decay_rate ** (start / decay_steps))
        optimizer.param_groups[0]['lr'] = new_lrate
        
        if args.pose_extract:
            if not args.netpose_freeze:  # 5/27
                decay_steps = args.lrate_decay * 2000
                new_lrate = args.pose_lrate * (decay_rate ** (pose_start / decay_steps))
                optimizer.param_groups[1]['lr'] = new_lrate
            
        # if not args.netpose_freeze:
        #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if args.pose_extract:
            pose_net.load_state_dict(ckpt['pose_exp_net_state_dict'])
    else:
        if args.pose_extract:
            pose_net.init_weights()

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, pose_start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4].[N_rays, N_samples, 3+1(rgb+alpha)] Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)      
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1] # [N_rays, N_samples]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1) 
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.max(1e-10 * torch.ones(weights.shape[0]), torch.sum(weights, -1))) 
    depth_threshold = np.full(disp_map.shape, 1e-10)
    depth_threshold = torch.from_numpy(depth_threshold).float()
    disp_mask = torch.where(torch.lt(depth_threshold.cuda(), torch.sum(weights, -1)), torch.ones_like(disp_map, dtype=torch.float32), torch.zeros_like(disp_map, dtype=torch.float32) )
    acc_map = torch.sum(weights, -1) # [N_rays]

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map, disp_mask

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.     
    N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals) 
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples]) 

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand  # 公式(2)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


    raw = network_query_fn(pts, viewdirs, network_fn) # [N_rays, N_samples, 3+1(rgb+alpha)]
    rgb_map, disp_map, acc_map, weights, depth_map, disp_mask = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    pt_map = rays_o + rays_d / disp_map[...,:,None] 
    

    if N_importance > 0: # refining procedure

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        pt_map_0 = pt_map  
        disp_mask_0 = disp_mask 

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach() # detach-Returns a new Variable, detached from the current graph. 

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)  
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)  # [N_rays, N_samples, 3+1(rgb+alpha)]

        rgb_map, disp_map, acc_map, weights, depth_map, disp_mask = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        # pt_map = rays_o + rays_d * depth_map[...,:,None]  
        pt_map = rays_o + rays_d / disp_map[...,:,None]  
        
    
    
    # ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'pt_map' : pt_map, 'disp_mask' : disp_mask} 
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['pt0'] = pt_map_0 
        ret['disp_mask_0'] = disp_mask_0 

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print("! [Numerical Error]: contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')  
    parser.add_argument("--expname", type=str, default='scene0316_00',  
                        help='experiment name')   
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs') 
    parser.add_argument("--datadir", type=str, default='../NoExtNeRF/data/ScanNet/scene0316_00', 
                        help='input data directory')  
    parser.add_argument("--ref_img_no", type=int, default=31,   # scene0000_01-23, scene0158_00-31, scene0316_00-31, scene0553_00-4
                        help="the index of refered image")  
    parser.add_argument("--siftdir", type=str, default='../NoExtNeRF/data/ScanNet/scene0316_00/orginal_sift_correspondences/', 
                        help='where to store ckpts and logs')   
    parser.add_argument('--pretrained-exppose', dest='pretrained_exp_pose', default=None, metavar='PATH',
                    help='path to pre-trained Exp Pose net model')
    
    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')     
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')     
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=16*16,  #default=32*32*4
                        help='batch size (number of random rays per gradient step)') # batch_size    
    parser.add_argument("--lrate", type=float, default=5e-4,  # 1e-5 for scene0316_00, 5e-4 for others
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,  
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,                          
                        help='number of rays processed in parallel, decrease if running out of memory')                        
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_false',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',   
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,   
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--image_weight", type=float, default=1.0, 
                        help='image weight in the loss')
    parser.add_argument("--sift_weight", type=float, default=1.0, 
                        help='sift weight in the loss')
    parser.add_argument("--project_weight", type=float, default=0.0, 
                        help='sift project_weight in the loss')   
    parser.add_argument("--train_pose_weight", type=float, default=0.0, 
                        help='train_pose_weight in the loss')     
    parser.add_argument('--pose_lrate', '--learning-rate', default=5e-4, type=float, # 1e-5 for scene0316_00, 5e-4 for others
                    metavar='LR', help='initial learning rate for pose net')
    parser.add_argument('--weight_decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')   
    parser.add_argument("--max_sift_num", type=int, default=20)  
    parser.add_argument('--netpose_freeze', type=bool, default=False)
    parser.add_argument('--pose_extract', type=bool, default=True)
    parser.add_argument('--use_default_ref', type=bool, default=False)    # True--ref_img_no is selected as the reference image
    parser.add_argument('--only_consective_frm', type=bool, default=False) # True-- the images without the sift correspondencs will not included
    parser.add_argument('--predict_test_pose', type=bool, default=False) # True-- include test images
 

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,                           
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=64,  
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=1e0, 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=2, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=25, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, #500
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,  #10000
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_predicted_pose", type=int, default=2000,  #10000
                        help='frequency of predicted pose saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--i_eval", type=int, default=5000,
                        help='frequency of testset measuring')
    parser.add_argument("--i_poses",     type=int, default=200, #500
                        help='frequency of saving predicted poses')
    parser.add_argument('--compute_extra_metrics', type=bool, default=True)
    

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    if args.pose_extract:
        pose_exp_net = PoseExpNet().to(device)
        if args.netpose_freeze:
            for i in pose_exp_net.parameters():
                i.requires_grad=False
    else: 
        pose_exp_net = None
    
    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        # images- [b, h, w, c]
        hwf = poses[0,:3,-1]  # [b, 3], intronics
        poses = poses[:,:3,:4] # [b, 3, 4], extronics
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
  
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, pose_start, grad_vars, optimizer = create_nerf(args, pose_exp_net)
    global_step = start
    pose_global_step = pose_start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:07d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0
        
    predicted_poses = load_predicted_pose(os.path.join(basedir, expname, 'predicted_poses.txt'))
    if len(predicted_poses)>0:
        poses[:, :3,:4] = predicted_poses
        original_predicted_poses = predicted_poses
        original_predicted_poses = torch.Tensor(original_predicted_poses).to(device)  

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 20000000 + 1  # 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    
    tb_writer = SummaryWriter(log_dir=".//log")
    if args.pose_extract:
        pose_exp_net.train()
    object_pt = []
    if args.predict_test_pose:
        args.ref_img_no, valid_frames = get_training_frames(np.concatenate((i_train, i_test)), images.shape[0], args.siftdir, args.factor, args.only_consective_frm, args.use_default_ref, args.ref_img_no)
    else:
        args.ref_img_no, valid_frames = get_training_frames(i_train, images.shape[0], args.siftdir, args.factor, args.only_consective_frm, args.use_default_ref, args.ref_img_no)
    p_eye = torch.eye(3)            
    t_zero = torch.zeros([3, 1])
    poses[args.ref_img_no, :3,:4] = torch.cat([p_eye, t_zero], dim=1)  # [3, 4]
    eval_poses = transform_pose_to_ref(args.ref_img_no, poses, i_val)  
    best_eval_psnr = 0
    best_ssim = 0
    start = start + 1

    for i in range(start, N_iters): 
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2] 

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from two image
            sample_img = []
            sample_pose = []
            
            valid_frame_inds = np.random.randint(len(valid_frames))
            img_inds = valid_frames[valid_frame_inds] 
            sample_pose.append(poses[img_inds[0], :3,:4].detach()) 
            
            for j in img_inds:
                dst_img = images[j]
                dst_img = torch.Tensor(dst_img).to(device)
                sample_img.append(dst_img)
            
            temp_img = torch.stack(sample_img, dim=0)
            temp_img = temp_img * 2. - 1  
            temp_img = temp_img.permute(0, 3, 1, 2)  # [b, c, h, w]
            if args.pose_extract:
                predicted_pose = pose_exp_net(temp_img[0, ...].unsqueeze(0), temp_img[1:, ...].unsqueeze(1)) 
                predicted_pose = torch.squeeze(predicted_pose, 0) # [2, 6]
                predicted_pose = pose_vec2mat(predicted_pose) # [2, 3, 4]
                predicted_pose = pose_transform(predicted_pose, sample_pose[0]) 
                poses[img_inds[1], :3,:4] = predicted_pose[0, ...]  
                poses[img_inds[2], :3,:4] = predicted_pose[1, ...]
            for j in range(2):
                if args.pose_extract:
                    sample_pose.append(predicted_pose[j])
                else:
                    sample_pose.append(poses[img_inds[j+1], :3,:4])

            sample_pose_inv = get_pose_inverse( torch.stack(sample_pose, dim=0))           
            if N_rand is not None: 
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                batch_ray_o = []
                batch_ray_d = []
                batch_s = []
                for j in range(len(sample_img)):
                    img = sample_img[j]
                    pose = sample_pose[j]
                    rays_o, rays_d = get_rays(H, W, K, pose)  # (H, W, 3), (H, W, 3)
                    
                    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()  # (N_rand, 2)
                    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_ray_o.append(rays_o)
                    batch_ray_d.append(rays_d)
                    # batch_rays = torch.stack([rays_o, rays_d], 0) # (2, N_rand, 3)
                    batch_target_s = img[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_s.append(batch_target_s)

                rays_o = torch.cat(batch_ray_o, 0) # (3*N_rand, 3)
                rays_d = torch.cat(batch_ray_d, 0) # (3*N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0) # (2, 3*N_rand, 3)
                target_s = torch.cat(batch_s, 0) # (3*N_rand, 3)    

                sift_file = args.siftdir + 'sift_{:0>2d}_{:0>2d}_{:0>2d}_{:0>2d}.txt'.format(args.factor, img_inds[0], img_inds[1], img_inds[2])                        
                sift_correspondences = load_sift_correspondences(sift_file) # [n, 6]
                sift_correspondences = np.array(sift_correspondences) 
                if len(sift_correspondences) > args.max_sift_num:                   
                    select_inds = np.random.choice(len(sift_correspondences), size=[args.max_sift_num], replace=False)  # (N_rand,)
                    sift_correspondences = sift_correspondences[select_inds]                    
                                  
                sift_correspondences = np.reshape(sift_correspondences, [-1, 3, 2]) # [n, 3, 2]
                n_sift = len(sift_correspondences)
                if n_sift == 0:
                    print("zero")
                sift_correspondences = np.transpose(sift_correspondences, [1,0,2]).astype(np.float32) # [3, n, 2]
                sift_correspondences = torch.Tensor(sift_correspondences).to(device)
             
                sift_ray_o = []
                sift_ray_d = []

                for j in range(3):
                    sift_point = sift_correspondences[j, :, :] # [n, 2]
                    
                    rays_o, rays_d = get_sift_rays(sift_point[:, 0], sift_point[:, 1], K, sample_pose[j])
                    sift_ray_o.append(rays_o)
                    sift_ray_d.append(rays_d)

                sift_rays_o = torch.cat(sift_ray_o, 0) # (3*N_sift, 3)
                sift_rays_d = torch.cat(sift_ray_d, 0) # (3*N_sift, 3)
                batch_sift_rays = torch.stack([sift_rays_o, sift_rays_d], 0) # (2, 3*N_sift, 3)
                batch_rays = torch.cat([batch_rays, batch_sift_rays], dim=1) # [2, 3*N_rand+3*N_sift, 3]

                # interplating
                coords_x_norm = 2.*sift_correspondences[..., 0]/(W-1.) - 1.  # [b=3, N_sift]
                coords_y_norm = 2.*sift_correspondences[..., 1]/(H-1.) - 1.
                coords_x_y_norm = torch.stack([coords_x_norm, coords_y_norm], dim=-1)  # [b=3, N_sift, 2]
                coords_x_y_norm = coords_x_y_norm.unsqueeze(1) # [b=3, 1, n, 2]

                # grid_sample [b, 3, h, w]
                interplated_image = torch.nn.functional.grid_sample(torch.stack(sample_img, dim=0).permute(0, 3, 1, 2), coords_x_y_norm, padding_mode='zeros', align_corners=True) 
                interplated_image = interplated_image.permute(0, 2, 3, 1)  # [b, 1, N_sift, c]
                interplated_image = torch.squeeze(interplated_image, 1) # [b=3, N_sift, 3]
                interplated_image = torch.reshape(interplated_image, [-1, 3]) # [3*N_sift, 3]
                target_s = torch.cat([target_s, interplated_image], dim=0) # [3*N_rand + 3*N_sift, 3]

        #####  Core optimization loop  #####
        rgb, disp, acc, pt, disp_mask, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = args.image_weight * img_loss
        psnr = mse2psnr(img_loss)        

        # 计算极线约束, pt-[3*N_rand + 3*N_sift, 3]
        sift_pts = pt[3*N_rand: 3*N_rand + 3*n_sift,...] # [3*N_sift, 3]
        sift_pts = torch.reshape(sift_pts, [3, n_sift, 3])   # 为世界坐标系下的坐标 
        mask = disp_mask[3*N_rand: 3*N_rand + 3*n_sift] # [3*N_sift]
        mask = torch.reshape(mask, [3, n_sift])  # [3, N_sift]   
        epipolar_loss = epipolar_constraint(sift_pts, mask)
        loss = loss + args.sift_weight * epipolar_loss         

        project_loss = sift_project_constraint(sift_correspondences[1:, ...], sift_pts[0, ...], sample_pose_inv[1:, ...], torch.Tensor(K).to(device))    
        loss = loss + args.project_weight * project_loss 
      
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + args.image_weight * img_loss0
            psnr0 = mse2psnr(img_loss0)

            sift_pts0 = extras['pt0'][3*N_rand: 3*N_rand + 3*n_sift,...] # [3*N_sift, 3]
            sift_pts0 = torch.reshape(sift_pts0, [3, n_sift, 3])    
            disp_mask_0 = extras['disp_mask_0'][3*N_rand: 3*N_rand + 3*n_sift] # [3*N_sift]
            disp_mask_0 = torch.reshape(disp_mask_0, [3, n_sift])  # [3, N_sift]    
            epipolar_loss0 = epipolar_constraint(sift_pts0, disp_mask_0)
            loss = loss + args.sift_weight * epipolar_loss0
            
            # if args.project_weight > 1.0e-8:
            project_loss0 = sift_project_constraint(sift_correspondences[1:, ...], sift_pts0[0, ...], sample_pose_inv[1:, ...], torch.Tensor(K).to(device))    
            loss = loss + args.project_weight * project_loss0                 
   
        loss.backward()  
        optimizer.step()        

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay  * 2000 
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        optimizer.param_groups[0]['lr'] = new_lrate
        
        if args.pose_extract:
            if not args.netpose_freeze:  
                decay_steps = args.lrate_decay  * 2000
                new_lrate = args.pose_lrate * (decay_rate ** (pose_global_step / decay_steps))
                optimizer.param_groups[1]['lr'] = new_lrate
       
        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:07d}.tar'.format(i))
            if args.N_importance > 0:
                if args.pose_extract:
                    torch.save({
                        'global_step': global_step,
                        'pose_global_step': pose_global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                        'pose_exp_net_state_dict':pose_exp_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
                else:
                    torch.save({
                        'global_step': global_step,
                        'pose_global_step': pose_global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),                        
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
            else:
                if args.pose_extract:
                    torch.save({
                        'global_step': global_step,
                        'pose_global_step': pose_global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'pose_exp_net_state_dict':pose_exp_net.state_dict(),                    
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
                else:
                    torch.save({
                        'global_step': global_step,
                        'pose_global_step': pose_global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),                                            
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
            print('Saved checkpoints at', path)
        if i%args.i_predicted_pose==0:    
            if args.predict_test_pose:
                predicted_poses_name = os.path.join(basedir, expname, 'predicted_poses_{:07d}.txt'.format(i))
                save_predicted_pose(predicted_poses_name, poses[:, :3,:4])
        
        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:07d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:07d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            # tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            print("[TRAIN] Iter: {} Loss: {}  PSNR: {}".format(i, loss.item(), psnr.item()))
            tb_writer.add_scalar("loss",loss,global_step)
            tb_writer.add_scalar("epipolar_loss",epipolar_loss,global_step) 
            tb_writer.add_scalar("img_loss",img_loss,global_step)   
            tb_writer.add_scalar("project_loss",project_loss,global_step)                       
            tb_writer.add_scalar("psnr",psnr,global_step)            
            tb_writer.add_scalar("lr",optimizer.param_groups[0]['lr'],global_step)
            if args.pose_extract:
                if not args.netpose_freeze:
                    tb_writer.add_scalar("pose_lr",optimizer.param_groups[1]['lr'],pose_global_step)
            if args.N_importance > 0:
                tb_writer.add_scalar("psnr0",psnr0,global_step)

        
        if i%args.i_img==0:
            
            # Log a rendered validation view to Tensorboard
            img_i=np.random.choice(i_val)
            target = images[img_i]
            pose = poses[img_i, :3,:4]

            with torch.no_grad():
                rgb, disp, acc, pt, disp_mask, extras = render(H, W, K, chunk=args.chunk, c2w=pose,
                                                    **render_kwargs_test)
            
            psnr = mse2psnr(img2mse(rgb, torch.from_numpy(target).to(device)))
            temp = to8b(rgb.cpu().numpy())
            # temp = torch.from_numpy(temp)[np.newaxis].to(device)
            tb_writer.add_image('rgb', temp, dataformats='HWC')
            tb_writer.add_image('disp', disp[np.newaxis,...])
            tb_writer.add_image('acc', acc[np.newaxis,...])
            
            tb_writer.add_scalar('psnr_holdout', psnr, global_step)
            tb_writer.add_image('rgb_holdout', target, dataformats='HWC')
            if args.N_importance > 0:
                temp = to8b(extras['rgb0'].cpu().numpy())                
                tb_writer.add_image('rgb0', temp, dataformats='HWC')
                tb_writer.add_image('disp0', extras['disp0'][np.newaxis,...])
                tb_writer.add_image('z_std', extras['z_std'][np.newaxis,...])      
        
        if i%args.i_eval==0:
            PSNRs = []
            ssims,l_alex,l_vgg=[],[],[]
            for i, v in enumerate(i_val):
                target = images[v]
                pose = poses[v, :3,:4]
                with torch.no_grad():
                    rgb, disp, acc, pt, disp_mask, extras = render(H, W, K, chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)
                
                psnr = mse2psnr(img2mse(rgb, torch.from_numpy(target).to(device)))
                PSNRs.append(psnr)

                if args.compute_extra_metrics:
                    this_type_str = type(rgb)
                    if this_type_str is not np.ndarray:
                        rgb = np.array(rgb.cpu())
                    
                    ssim = rgb_ssim(rgb, target, 1)
                    l_a = rgb_lpips(target, rgb, 'alex', device)
                    l_v = rgb_lpips(target, rgb, 'vgg', device)
                    ssims.append(ssim)
                    l_alex.append(l_a)
                    l_vgg.append(l_v)
            if PSNRs:
                eval_psnr = torch.stack(PSNRs, -1).mean()                
                if args.compute_extra_metrics:
                    eval_ssim = np.mean(np.asarray(ssims))
                    eval_l_a = np.mean(np.asarray(l_alex))
                    eval_l_v = np.mean(np.asarray(l_vgg))
                    if eval_ssim > best_ssim:
                        best_ssim = eval_ssim
                        mean_name = os.path.join(basedir, expname, 'mean.txt')
                        # header = ('psnr', 'sim', 'l_a', 'l_v')
                        with open(mean_name, "a") as f:
                            np.savetxt(f, np.asarray([eval_psnr.cpu(), eval_ssim, eval_l_a, eval_l_v]), newline=' ', delimiter=',')
                            f.write("\n")
                    if eval_psnr > best_eval_psnr:
                        mean_name = os.path.join(basedir, expname, 'mean.txt')
                        with open(mean_name, "a") as f:
                            np.savetxt(f, np.asarray([eval_psnr.cpu(), eval_ssim, eval_l_a, eval_l_v]), newline=' ', delimiter=',')
                            f.write("\n")
                else:
                    if eval_psnr> best_eval_psnr:
                        np.savetxt(mean_name, np.asarray([eval_psnr.cpu()])) 
                if eval_psnr> best_eval_psnr:
                    best_eval_psnr = eval_psnr      
                print("eval_psnr: {}".format(eval_psnr))
                tb_writer.add_scalar('eval_psnr', eval_psnr,global_step)
        
        global_step += 1
        if not args.netpose_freeze:  # 5/27
            pose_global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
