import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms as tf
from torch.optim import AdamW
import cv2
import numpy as np
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import argparse
import math as m
from tqdm import tqdm
from os import listdir
from lib.network import IONet, VONet, VIONet
import pandas as pd
import cv2
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from lib.helpers import *


def test(batches, Checkpointpath, device):
    
    #initialize model and load the checkpoint
    model = VIONet(6,64,2,4)
    model = model.to(device)
    Checkpoint = torch.load(Checkpointpath)
    model.load_state_dict(Checkpoint['model_state_dict'])
    model.eval()
    
    #initializing q, t
    q = []
    t = []
    
    #inferencing model
    for batch in batches:
        batch = batch.to(device)
        q_pred, t_pred = model(batch)
        
        q.append(q_pred)
        t.append(t_pred)

    q = torch.cat(q, dim=0)
    t = torch.cat(t, dim=0)
    
    relative_pose = torch.cat((t, q), dim=1)
    
    return relative_pose


def relative_to_absolutepose_T(relative_pose, absolute_pose_gt):
    
    print(relative_pose)
    absolute_pose = np.zeros_like(relative_pose)
    row_of_zeros = np.zeros((1, absolute_pose.shape[1]), dtype=absolute_pose.dtype)
    absolute_pose = np.concatenate((absolute_pose, row_of_zeros), axis=0)
    
    absolute_pose[0] = absolute_pose_gt[0,1:]
    
    for i in range(1, len(relative_pose)):
        d_translation = relative_pose[i, :3]
        d_rotation_q = relative_pose[i, 3:]
        
        q_prev = absolute_pose[i-1, 3:]
        
        rot_mat = convert_R_to_rotmat(convert_quaternion_to_R(d_rotation_q))
        rot_mat_prev = convert_R_to_rotmat(convert_quaternion_to_R(q_prev))
        
        
        
        absolute_pose[i, :3] = absolute_pose[i-1, :3] + np.dot(rot_mat_prev, d_translation)
        rot_mat_abs = np.dot(rot_mat_prev, rot_mat)
        absolute_pose[i, 3:] = convert_euler_to_quaternion(convert_rotmat_to_R(rot_mat_abs))
    
    return absolute_pose

def save_to_txt(absolute_pose):
    
    with open('absolute_pose_new_1038.txt', 'w') as f:
        f.write("# Timestamp tx ty tz qx qy qz qw\n")
        
        for i,pose in enumerate(absolute_pose):
            line = f"{i} {pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f} {pose[4]:.6f} {pose[5]:.6f} {pose[6]:.6f}\n"
            f.write(line)
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="Path to the images folder", default=r"data_t\images")
    parser.add_argument("--IMU", type=str, help="path to imu data", default=r"data_t\imu.csv")
    parser.add_argument("--rotations", type=str, help="path to relative pose data", default=r"data_t\relative_pose.csv")
    parser.add_argument("--groundtruth", type=str, help="path to ground truth data", default=r"data_t\trajectory_pose.csv")
    parser.add_argument("--minibatch", type=int, help="Minibatch size", default=1)
    parser.add_argument("--latestmodelpath", type=str, help="folder to load checkpoint", default=r"checkpoints\savedcheckpoints\checkpoint_ionet499.ckpt")
    args = parser.parse_args()

    IMU = pd.read_csv(args.IMU)
    IMU_array = IMU.values
    
    batch_size = args.minibatch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #ground truth data
    gt_traj = pd.read_csv(args.groundtruth)
    gt_traj = gt_traj.values
    
    batches = BatchgenIMU(IMU_array, batch_size)
    
    print("Batches Shape: ", len(batches))
    relative_pose = test(batches, args.latestmodelpath, device)
    
    relative_pose = relative_pose.cpu().detach().numpy()
    
    absolute_pose = relative_to_absolutepose_T(relative_pose, gt_traj)
    
    save_to_txt(absolute_pose)
    
    
    
    
    
if __name__ == "__main__":
    main()