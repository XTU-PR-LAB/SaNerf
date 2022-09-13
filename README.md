# SaNerf
Structure-Aware NeRF without Posed Camera via Epipolar Constraint
### Dependencies

Make sure you have the following dependencies installed before proceeding:

-torch>=1.8
-torchvision>=0.9.1
-imageio
-imageio-ffmpeg
-matplotlib
-configargparse
-tensorboard>=2.0
-opencv-python

### Training from scratch
Download the correspondeing datasets just like LLFF, and  Put them into data folder.
Preprocessing
     run make_correspondences.py to extract the SIFT matches.
The training script is in run_nerf.py, to train a NoExtNeRF:
     python run_nerf.py
### Acknowledgments
The code is adapted from https://github.com/yenchenlin/nerf-pytorch
