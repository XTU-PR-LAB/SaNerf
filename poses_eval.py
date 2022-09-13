# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import scipy.misc
import numpy as np
from glob import glob
# from joblib import Parallel, delayed
import os

from load_llff import load_llff_data
from run_nerf_helpers import *



parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, default='./data/nerf_llff_data/orchids', help="where the dataset is stored")
parser.add_argument("--factor", type=int, default=8, help="downsample factor for LLFF images")
parser.add_argument("--ref_img_no", type=int, default=13, help="the index of refered image") # orchids-13, leaves-18, fern-6
parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs') 
parser.add_argument("--expname", type=str, default='orchids',  
                        help='experiment name')  
parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes') 
parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

args = parser.parse_args()

def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420

    Args
    X: array NxM of targets, with N number of points and M point dimensionality
    Y: array NxM of inputs
    compute_optimal_scale: whether we compute optimal scale or force it to be 1

    Returns:
    d: squared error after transformation
    Z: transformed Y
    T: computed rotation
    b: scaling
    c: translation
    """
    import numpy as np

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:,-1] *= np.sign( detT )
    s[-1]   *= np.sign( detT )
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA**2
        Z = normX*traceTA*np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    c = muX - b*np.dot(muY, T)

    return d, Z, T, b, c

def transform_poses(ref_img_no, poses):
    ref_r = poses[ref_img_no, :, :3] # [3, 3]
    ref_t = np.expand_dims(poses[ref_img_no, :, 3], -1) # [3, 1]     
    ref_r_inv = np.linalg.inv (ref_r) 
    trans_pose = []
    for i in range(len(poses)):
        loc_r = poses[i, :, :3] # [3, 3]
        # loc_t = poses[i, :, 3].unsqueeze(-1) # [3, 1]
        loc_t = np.expand_dims(poses[i, :, 3], -1) # [3, 1]
        trans_r = np.matmul(ref_r_inv, loc_r) # [3, 3]
        trans_t = np.matmul(ref_r_inv, loc_t-ref_t) # [3, 1]
        pose = np.concatenate([trans_r, trans_t], 1) # [3, 4]
        if i == ref_img_no:
            print("hlll")
        trans_pose.append(pose)
    trans_pose = np.stack(trans_pose, 0) # [n, 3, 4]
    return trans_pose
def save_eval_poses_t(eval_poses_t, predicted_poses_t, error):
    dir_name = os.path.join(args.basedir, args.expname, 'eval_poses')
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    eval_poses_t_file = os.path.join(dir_name, 'eval_poses_t.txt')
    with open(eval_poses_t_file, 'w') as f:
            for m in range(len(eval_poses_t)):
                f.write('%f\t%f\t%f\n' %(eval_poses_t[m][0][0], eval_poses_t[m][1][0], eval_poses_t[m][2][0]))   
    predicted_poses_t_file = os.path.join(dir_name, 'predicted_poses_t.txt')
    with open(predicted_poses_t_file, 'w') as f:
            for m in range(len(predicted_poses_t)):
                f.write('%f\t%f\t%f\n' %(predicted_poses_t[m][0][0], predicted_poses_t[m][1][0], predicted_poses_t[m][2][0])) 
    error_file = os.path.join(dir_name, 'error.txt')     
    with open(error_file, 'w') as f:      
        f.write('%f\n' %(error))   
        

def eval_poses():
    images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
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
    eval_poses = transform_poses(args.ref_img_no, poses)
    eval_poses_r = eval_poses[:, :, :3] # [n, 3, 3]
    eval_poses_t = np.expand_dims(eval_poses[:, :, 3], -1) #.unsqueeze(-1) # [n, 3, 1]
    eval_poses_t = np.delete(eval_poses_t, args.ref_img_no, axis=0) # [n-1, 3, 1]
    predicted_poses = load_predicted_pose(os.path.join(args.basedir, args.expname, 'predicted_poses.txt'))
    predicted_poses_r = predicted_poses[:, :, :3] # [n, 3, 3]
    predicted_poses_t = np.expand_dims(predicted_poses[:, :, 3], -1) #.unsqueeze(-1) # [n, 3, 1]
    predicted_poses_t = np.delete(predicted_poses_t, args.ref_img_no, axis=0) # [n-1, 3, 1]
    predicted_poses_t_transform = np.zeros_like(predicted_poses_t)
    
    for j in range(len(eval_poses)-1):
        gt = eval_poses_t[j, :, :]  # [16, 3]
        out = predicted_poses_t[j, :, :]
        _, Z, T, b, c = compute_similarity_transform(gt, out, compute_optimal_scale=True)  
        out = (b * out.dot(T)) + c

        predicted_poses_t_transform[j, :, :] = out  # np.reshape(out, [-1, 3])

        sqerr_transform = (predicted_poses_t_transform - eval_poses_t) ** 2  # [N, 16, 3]Squared error between prediction and expected output
        dist_transform = np.sqrt(np.sum(sqerr_transform, axis=2))  # [N, 16]
        total_err_transform = np.mean(dist_transform)
    
     # Compute Euclidean distance error per joint
    sqerr = (predicted_poses_t_transform - eval_poses_t) ** 2  # [N, 16, 3]Squared error between prediction and expected output
    dist = np.sqrt(np.sum(sqerr, axis=2))  # [N, 16] 
    total_err = np.mean(dist)
    save_eval_poses_t(eval_poses_t, predicted_poses_t_transform, total_err)

    
eval_poses()