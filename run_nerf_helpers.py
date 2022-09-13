# -*- coding: utf-8 -*-
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.linalg import det
import os
import scipy.signal


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')
        # this_type_str = type(z)
        # if this_type_str is np.ndarray:
        #     return scipy.signal.convolve2d(z, f, mode='valid')
        # else:
        #     return scipy.signal.convolve2d(z.cpu(), f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim

__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def transform_pose_to_ref(ref_img_no, poses, i_test):
    ref_r = poses[ref_img_no, :, :3] # [3, 3]
    ref_t = poses[ref_img_no, :, 3].unsqueeze(-1) # [3, 1]     
    ref_r_inv = invmat(ref_r) # 求姿态的逆矩阵      
    trans_pose = []
    for i in i_test:
        loc_r = poses[i, :, :3] # [3, 3]
        loc_t = poses[i, :, 3].unsqueeze(-1) # [3, 1]
        trans_r = torch.matmul(ref_r_inv, loc_r) # [3, 3]
        trans_t = torch.matmul(ref_r_inv, loc_t-ref_t) # [3, 1]
        pose = torch.cat([trans_r, trans_t], 1) # [3, 4]
        trans_pose.append(pose)
    trans_pose = torch.stack(trans_pose, 0) # [n, 3, 4]
    return trans_pose
        

def load_sift_correspondences(data_file):
    if os.stat(data_file).st_size == 0:
        return []
    with open(data_file,'r') as in_file:
        txt = in_file.readlines()
        txt = [txt[i].split(' ') for i in range(len(txt))]                       
    for i in range(len(txt)):    
        for j in range(len(txt[0])):                                    
            txt[i][j] = txt[i][j].strip('[\n,\t]') 
            txt[i][j] = float(txt[i][j])
    return txt

def get_frm_no_in_set(valid_frames):
    frm_no = []
    for i in range(len(valid_frames)):
        frm_no.append(valid_frames[i][0])
        frm_no.append(valid_frames[i][1])
        frm_no.append(valid_frames[i][2])
    return list(set(frm_no)) 

def save_predicted_pose(file_name, poses):
    with open(file_name, 'w') as f:
            for m in range(poses.shape[0]):
                for i in range(3):
                    for j in range(4):
                        if i ==2 and j==3:
                            f.write('%f' %(poses[m][i][j]))     
                        else:
                            f.write('%f ' %(poses[m][i][j]))
                f.write('\n')           
    # poses[img_inds[1], :3,:4]

def load_predicted_pose(data_file):
    if not os.path.exists(data_file):
        return []
    if os.stat(data_file).st_size == 0:
        return []
    with open(data_file,'r') as in_file:
        txt = in_file.readlines()
        txt = [txt[i].split(' ') for i in range(len(txt))]                        
    for i in range(len(txt)):    
        for j in range(len(txt[0])):
            txt[i][j] = txt[i][j].strip('[\n,\t]')
            txt[i][j] = float(txt[i][j])
    poses = np.array(txt)
    poses = np.reshape(poses, [-1, 3, 4]) # [n, 3, 4]
    return poses

def get_training_frames(i_train, total_num, siftdir, factor, only_consective_frm = False, use_default_ref= False, input_ref_no = -1):
    i_train.sort()  
    ref_img_no = i_train[0]
    best_sift_num = 0
    frm_num = 0
    best_frm_list = []
    best_valid_frames = []
    if use_default_ref:
        valid_frames, valid_sift_num = get_valid_frames(i_train, input_ref_no, total_num, siftdir, factor)
        valid_frm_list = get_frm_no_in_set(valid_frames)  
        best_frm_list = valid_frm_list
        best_valid_frames = valid_frames
        ref_img_no = input_ref_no
    else:        
        for i in i_train:        
            valid_frames, valid_sift_num = get_valid_frames(i_train, i, total_num, siftdir, factor)
            valid_frm_list = get_frm_no_in_set(valid_frames)  
            if len(valid_frm_list) > frm_num or (len(valid_frm_list) == frm_num and valid_sift_num > best_sift_num):
                best_sift_num = valid_sift_num
                ref_img_no = i
                best_frm_list = valid_frm_list
                frm_num = len(valid_frm_list)
                best_valid_frames = valid_frames
    if only_consective_frm:
        print("ref_img_no:{}\n".format(ref_img_no))
        print("valid frame:\n")
        for i in range(len(best_valid_frames)): 
            print("{},{},{}\n".format(best_valid_frames[i][0], best_valid_frames[i][1], best_valid_frames[i][2]))
        return ref_img_no, best_valid_frames           
    invalid_frames_no = np.setdiff1d(i_train, best_frm_list)
    
    new_valid_frams = []
    for i in invalid_frames_no: 
        if i in new_valid_frams: 
            continue
        index = i_train.tolist().index(i) 
        best_frames = [] 
        valid_sift_num = 0
        for j in best_frm_list: 
            if j == ref_img_no:
                    continue           
            for k in range(index):  
                if i_train[k] == ref_img_no or i_train[k] == j or j == i:
                    continue
                sift_file = siftdir + 'sift_{:0>2d}_{:0>2d}_{:0>2d}_{:0>2d}.txt'.format(factor, j, i_train[k], i) 
                sift_correspondences = load_sift_correspondences(sift_file) 
                if len(sift_correspondences) > valid_sift_num:
                    best_frames = [ j, i_train[k], i]
                    valid_sift_num = len(sift_correspondences)
                    
            for k in range(index+1, len(i_train)): 
                if i_train[k] == ref_img_no  or i_train[k] == j or j == i:
                    continue
                sift_file = siftdir + 'sift_{:0>2d}_{:0>2d}_{:0>2d}_{:0>2d}.txt'.format(factor, j, i, i_train[k]) 
                sift_correspondences = load_sift_correspondences(sift_file) 
                if len(sift_correspondences) > valid_sift_num:
                    best_frames = [ j, i, i_train[k]]
                    valid_sift_num = len(sift_correspondences)
        if valid_sift_num > 3:  
            best_valid_frames.append(best_frames)
            new_valid_frams.append(best_frames[1])
            new_valid_frams.append(best_frames[2])
            
    best_frm_list = get_frm_no_in_set(best_valid_frames)  
    invalid_frames_no = np.setdiff1d(i_train, best_frm_list)
    
    new_valid_frams = []
    for i in invalid_frames_no: 
        if i in new_valid_frams:
            continue
        index = i_train.tolist().index(i) 
        best_frames = [] 
        valid_sift_num = 0
        for j in best_frm_list: 
            if j == ref_img_no:
                    continue           
            for k in range(index):  
                if i_train[k] == ref_img_no or i_train[k] == j or j == i:
                    continue
                sift_file = siftdir + 'sift_{:0>2d}_{:0>2d}_{:0>2d}_{:0>2d}.txt'.format(factor, j, i_train[k], i) 
                sift_correspondences = load_sift_correspondences(sift_file) 
                if len(sift_correspondences) > valid_sift_num:
                    best_frames = [ j, i_train[k], i]
                    valid_sift_num = len(sift_correspondences)
                    
            for k in range(index+1, len(i_train)): 
                if i_train[k] == ref_img_no  or i_train[k] == j or j == i:
                    continue
                sift_file = siftdir + 'sift_{:0>2d}_{:0>2d}_{:0>2d}_{:0>2d}.txt'.format(factor, j, i, i_train[k]) 
                sift_correspondences = load_sift_correspondences(sift_file) 
                if len(sift_correspondences) > valid_sift_num:
                    best_frames = [ j, i, i_train[k]]
                    valid_sift_num = len(sift_correspondences)
        if valid_sift_num > 3:  
            best_valid_frames.append(best_frames)
            new_valid_frams.append(best_frames[1])
            new_valid_frams.append(best_frames[2])
            
    
    print("ref_img_no:{}\n".format(ref_img_no))
    print("valid frame:\n")
    for i in range(len(best_valid_frames)): 
        print("{},{},{}\n".format(best_valid_frames[i][0], best_valid_frames[i][1], best_valid_frames[i][2]))
                
    return ref_img_no, best_valid_frames

def get_valid_frames(i_train, ref_img_no, total_num, siftdir, factor):
    valid_frames = []
    valid_idx = [[-1, -1, -1]] * total_num  
    valid_sift_num = [0] * total_num
    for i in range(total_num):
        if i == ref_img_no:
            continue
        if not (i in  i_train):  
            continue
        for j in range(i+1, total_num):
            if j == ref_img_no:
                continue
            if not (j in  i_train) : 
                continue
            sift_file = siftdir + 'sift_{:0>2d}_{:0>2d}_{:0>2d}_{:0>2d}.txt'.format(factor, ref_img_no, i, j)                
            sift_correspondences = load_sift_correspondences(sift_file) 
            if len(sift_correspondences) > valid_sift_num[i]:
                valid_idx[i] = [ref_img_no, i, j]
                valid_sift_num[i] = len(sift_correspondences)
            if len(sift_correspondences) > valid_sift_num[j]:
                valid_idx[j] = [ref_img_no, i, j]
                valid_sift_num[j] = len(sift_correspondences)
    total_sift_num = 0
    for i in range(total_num): 
        if  valid_sift_num[i] > 3:      
            valid_frames.append(valid_idx[i])
            total_sift_num = total_sift_num + valid_sift_num[i]          

    return valid_frames, total_sift_num
            
def in_invalid_frames(frm_idx_1, frm_idx_2, invalid_frames):
    for i, j in invalid_frames:
        if (i == frm_idx_1 and j == frm_idx_2) or (j == frm_idx_1 and i == frm_idx_2):
            return True
    return False

def in_valid_frames(frm_idx_1, frm_idx_2, valid_frames):
    for i, j in valid_frames:
        if (i == frm_idx_1 and j == frm_idx_2) or (j == frm_idx_1 and i == frm_idx_2):
            return True
    return False
    
def create_invalide_sift_idx(max_image_num):
    invalide_sift_idx ={}
    for i in range(max_image_num):
        for j in range(i+1, max_image_num):
            invalide_sift_idx["{}_{}".format(i,j)] = []
    return invalide_sift_idx

def cof1(M,index):
    zs = M[:index[0]-1,:index[1]-1]
    ys = M[:index[0]-1,index[1]:]
    zx = M[index[0]:,:index[1]-1]
    yx = M[index[0]:,index[1]:]
    s = torch.cat((zs,ys),axis=1)
    x = torch.cat((zx,yx),axis=1)
    return det(torch.cat((s,x),axis=0))

def alcof(M,index):
    return pow(-1,index[0]+index[1])*cof1(M,index)

def adj(M):
    result = torch.zeros((M.shape[0],M.shape[1]))
    for i in range(1,M.shape[0]+1):
        for j in range(1,M.shape[1]+1):
            result[j-1][i-1] = alcof(M,[i,j])
    return result


def invmat(M):
    return 1.0/det(M)*adj(M)  

def pose_transform(poses, ref_pose):
    temp_pose = ref_pose.unsqueeze(0).repeat(poses.shape[0], 1, 1) # [2, 3, 4]
    ref_r = temp_pose[:, :, :3] # [2, 3, 3]
    ref_t = temp_pose[:, :, 3].unsqueeze(-1) # [2, 3, 1]
    loc_r = poses[:, :, :3] # [2, 3, 3]
    loc_t = poses[:, :, 3].unsqueeze(-1) # [2, 3, 1]
    r = torch.matmul(ref_r, loc_r) # [2, 3, 3]
    t = torch.matmul(ref_r, loc_t) + ref_t # [2, 3, 1]
    ret_pose = torch.cat([r, t], dim=2)  #[2, 3, 4]
    return ret_pose

def get_pose_inverse(pose):
    # poses-[n, 3, 4]
    temp_pose = []
    for i in range(pose.shape[0]):
        pose_i = pose[i, ...]  # [3, 4]
        R_inv = invmat(pose_i[:, :3])
        t = pose_i[:, 3] # [3, 1]
        t_inv = -(R_inv @ t)
        pose_i_inv = torch.cat([R_inv, t_inv.unsqueeze(-1)], dim=1)  #[3, 4]
        temp_pose.append(pose_i_inv)
        
    pose_inv = torch.stack(temp_pose, dim=0) # [n, 3, 4]
    return pose_inv      

def get_valid_flag(pixel_coord, img_h, img_w):
    coords_x, coords_y = torch.split(pixel_coord, 1, dim=2)
    img_h_threshold = np.full(coords_y.shape, img_h) # np.full(coords_y.get_shape().as_list(), img_h)
    img_h_threshold = torch.from_numpy(img_h_threshold).float() # tf.constant(img_h_threshold, dtype=tf.float32)
    img_w_threshold = np.full(coords_x.shape, img_w) # np.full(coords_x.get_shape().as_list(), img_w)
    img_w_threshold = torch.from_numpy(img_w_threshold).float() # tf.constant(img_w_threshold, dtype=tf.float32)   
    
    valid_flag_x = torch.where(torch.lt(torch.abs(coords_x), img_w_threshold.cuda()), torch.ones_like(coords_x, dtype=torch.float32), torch.zeros_like(coords_x, dtype=torch.float32) )
    valid_flag_y = torch.where(torch.lt(torch.abs(coords_y), img_h_threshold.cuda()), torch.ones_like(coords_x, dtype=torch.float32), torch.zeros_like(coords_x, dtype=torch.float32) )
    valid_flag = valid_flag_x * valid_flag_y
    return valid_flag

def cam2pixel(cam_coords, intrinsics):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: [B, 3]
        intrinsics - [3, 3]
    Returns:
        coordinates -- [B, 2]
    """
    b, _ = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1) # [b, 3, 1]
    b_intrinsics = intrinsics.unsqueeze(0).repeat(b, 1, 1) # [b, 3, 3]
    pcoords = b_intrinsics @ cam_coords_flat  # [b, 3, 1]
    X = pcoords[:, 0] # [B, 1]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X = X / Z # [B, 1]
    Y = Y / Z

    pixel_coords = torch.cat([X, Y], dim=1)  # [B, 2]
    return pixel_coords

def pose2proj_cam(poses, intrinsics):
    '''
    poses-[b=3, 3, 4]
    intrinsics - [3, 3]    '''
    b_intrinsics = intrinsics.unsqueeze(0).repeat(poses.shape[0], 1, 1) # [b, 3, 3]
    proj_cam = b_intrinsics @ poses  # [B, 3, 4]
    return proj_cam

 
def sift_project_constraint(correspondences, sift_pts,  poses, intrinsics):
    # correspondences-[2, N_sift , 2]
    # sift_pts - [N_sift, 3]
    # poses-[2, 3 , 4]
    # intrinsics-[3, 3]
    # temp = sift_pts[:, 0].unsqueeze(1)   

    
    if sift_pts.shape[0] == 0:
        print("no sift_pts")
        return 0
    rot, tr = poses[:,:,:3], poses[:,:,-1:]  # [2, 3, 3], [2, 3, 1]
    rot0 = rot[0,...].unsqueeze(0).repeat(sift_pts.shape[0], 1, 1) # [N_sift, 3, 3]
    rot1 = rot[1,...].unsqueeze(0).repeat(sift_pts.shape[0], 1, 1) # [N_sift, 3, 3]
    rot = torch.cat([rot0, rot1], dim=0) # [2* N_sift, 3, 3]
    tr0 = tr[0,...].unsqueeze(0).repeat(sift_pts.shape[0], 1, 1) # [N_sift, 3, 1]
    tr1 = tr[1,...].unsqueeze(0).repeat(sift_pts.shape[0], 1, 1) # [N_sift, 3, 1]
    tr = torch.cat([tr0, tr1], dim=0)  # [2* N_sift, 3, 1]
    batch_pts = sift_pts.unsqueeze(0).repeat(poses.shape[0], 1, 1) # [2, N_sift, 3]
    batch_pts = batch_pts.reshape(-1, 3, 1) # [b=2* N_sift, 3, 1]
    pixel_coords = rot @ batch_pts + tr  
    # print(pixel_coords)
    pixel_coords = torch.cat([pixel_coords[:, 0, :], -pixel_coords[:, 1, :], -pixel_coords[:, 2, :]], dim = -1)  
    # print(pixel_coords)
    pixel_coords = cam2pixel(pixel_coords,intrinsics)  # [b=2* N_sift, 2]
    pixel_coords = pixel_coords.reshape(poses.shape[0], -1, 2) # [2, N_sift, 2]
    project_loss = torch.mean(torch.nn.functional.pairwise_distance(correspondences.reshape(-1, 2), pixel_coords.reshape(-1, 2), p=2)) 
    return project_loss
    

def get_projected_pt(pts, poses, intrinsics):    
    
    # pts-[3* N_sift, 3]
    # poses-[3, 3, 4]
    # intrinsics-[3, 3]
    # return-[3, *3*N_sift, 2]    

    rot, tr = poses[:,:,:3], poses[:,:,-1:]  # [3, 3, 3], [3, 3, 1]
    rot0 = rot[0,...].unsqueeze(0).repeat(pts.shape[0], 1, 1) # [3*N_sift, 3, 3]
    rot1 = rot[1,...].unsqueeze(0).repeat(pts.shape[0], 1, 1) # [3*N_sift, 3, 3]
    rot2 = rot[2,...].unsqueeze(0).repeat(pts.shape[0], 1, 1) # [3*N_sift, 3, 3]
    rot = torch.cat([rot0, rot1, rot2], dim=0) # [3* 3*N_sift, 3, 3]

    tr0 = tr[0,...].unsqueeze(0).repeat(pts.shape[0], 1, 1) # [3*N_sift, 3, 1]
    tr1 = tr[1,...].unsqueeze(0).repeat(pts.shape[0], 1, 1) # [3*N_sift, 3, 1]
    tr2 = tr[2,...].unsqueeze(0).repeat(pts.shape[0], 1, 1) # [3*N_sift, 3, 1]
    tr = torch.cat([tr0, tr1, tr2], dim=0)  # [3* 3*N_sift, 3, 1]
    
    batch_pts = pts.unsqueeze(0).repeat(poses.shape[0], 1, 1) # [3, 3*N_sift, 3]
    batch_pts = batch_pts.reshape(-1, 3, 1) # [b=3* 3*N_sift, 3, 1]
    
    pixel_coords = rot @ batch_pts + tr    # [b=3* 3*N_sift, 3]     
    pixel_coords = torch.cat([pixel_coords[:, 0, :], -pixel_coords[:, 1, :], -pixel_coords[:, 2, :]], dim = -1)  
    pixel_coords = cam2pixel(pixel_coords,intrinsics)  # [b=3* 3*N_sift, 2]
    
    pixel_coords = pixel_coords.reshape(poses.shape[0], -1, 2) # [3, 3*N_sift, 2]
    return pixel_coords

def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat

def pose_vec2mat(vec):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    rot_mat = euler2mat(rot)  # [B, 3, 3]    
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat

def epipolar_constraint(pts, mask):
    # pts-[3, n_sift, 3]
    # mask-[3, n_sift]
    # loss = img2mse(pts[0, ...], pts[1, ...])
    # loss = loss + img2mse(pts[0, ...], pts[2, ...])
    # loss = loss + img2mse(pts[1, ...], pts[2, ...])
    loss = torch.mean(((pts[0, ...] - pts[1, ...]) * mask[0].unsqueeze(-1).repeat(1, 3) * mask[1].unsqueeze(-1).repeat(1, 3)) ** 2)
    loss = loss + torch.mean(((pts[0, ...] - pts[2, ...]) * mask[0].unsqueeze(-1).repeat(1, 3) * mask[2].unsqueeze(-1).repeat(1, 3)) ** 2)
    loss = loss + torch.mean(((pts[1, ...] - pts[2, ...]) * mask[1].unsqueeze(-1).repeat(1, 3) * mask[2].unsqueeze(-1).repeat(1, 3)) ** 2)
    return loss


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class NeRF(nn.Module):
    
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.output_alpha_linear = nn.Linear(W, 1)        
     
        if use_viewdirs:
            color_ch = input_ch + input_ch_views
        else:
            color_ch = input_ch
        self.color_linears = nn.ModuleList(
            [nn.Linear(color_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + color_ch, W) for i in range(D-1)])
        
        self.output_color_linear = nn.Linear(W, 3)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.output_alpha_linear(h)
        
        if self.use_viewdirs:
            h = x
        else:
            h = input_pts
        for i, l in enumerate(self.color_linears):
            h = self.color_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        rgb = self.output_color_linear(h)
        outputs = torch.cat([rgb, alpha], -1)
        return outputs   
    
    
    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1) 
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_o = c2w[:3,-1].expand(rays_d.shape) 
    return rays_o, rays_d

def get_sift_rays(i, j, K, c2w):
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1) 
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape) 
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d # [n, 3]

# added by shu chen
def get_sift_rays_np(i, j, K, c2w):    
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

# ndc -- normal device coordinate
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True) 
    cdf = torch.cumsum(pdf, -1)  
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)  
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
