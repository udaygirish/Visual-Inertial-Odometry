import math
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch 



def convert_to_euler(rotation):
    return (
        math.radians(rotation[0]),
        math.radians(rotation[1]),
        math.radians(rotation[2]),
    )


def convert_euler_to_quaternion(rotation):
    r = R.from_euler("xyz", rotation, degrees=False)
    return r.as_quat()


def convert_R_to_rotmat(rotation):
    r = R.from_euler("xyz", rotation, degrees=False)
    return r.as_matrix()


def convert_quaternion_to_R(quaternion):
    r = R.from_quat(quaternion)
    return r.as_euler("xyz", degrees=False)


def convert_rotmat_to_R(rotmat):
    r = R.from_matrix(rotmat)
    return r.as_euler("xyz", degrees=False)


def get_relative_pose(pose1, pose2):
    # Get relative pose of pose2 wrt pose1
    pose1 = np.array(pose1)
    pose2 = np.array(pose2)
    relative_pose = np.dot(np.linalg.inv(pose1), pose2)
    return relative_pose

def loss_function_1(q_pred, t_pred, q_true, t_true):
    # Translation Loss 
    criterion_t = torch.nn.MSELoss()
    loss_t = criterion_t(t_true, t_pred)

    # Quaternion Loss
    dot_product = torch.sum(q_pred * q_true, dim=1)
    q_norm_pred = torch.norm(q_pred, p=2, dim=1)
    q_norm_true = torch.norm(q_true, p=2, dim=1)
    loss_q = 1 - torch.abs(dot_product) / (q_norm_pred * q_norm_true)

    loss_q = torch.mean(loss_q)
    # Merge Translation and Quaternion Loss
    beta = 100
    loss = loss_t + beta * loss_q
    return loss

# def loss_function_2(q_pred, t_pred, q_true, t_true):
#     # Translation Loss 
#     criterion_t = torch.nn.MSELoss()
#     loss_t = criterion_t(t_true, t_pred)

#     # Quaternion Loss - Yet to define cos loss
#     cri 