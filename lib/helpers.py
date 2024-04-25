    
import math 
from scipy.spatial.transform import Rotation as R
import numpy as np

def convert_to_euler(rotation):
    return math.radians(rotation[0]), math.radians(rotation[1]), math.radians(rotation[2])
    
def convert_euler_to_quaternion(rotation):
    r = R.from_euler('xyz', rotation, degrees=False)
    return r.as_quat()

def convert_R_to_rotmat(rotation):
    r = R.from_euler('xyz', rotation, degrees=False)
    return r.as_matrix()

def convert_quaternion_to_R(quaternion):
    r = R.from_quat(quaternion)
    return r.as_euler('xyz', degrees=False)

def convert_rotmat_to_R(rotmat):
    r = R.from_matrix(rotmat)
    return r.as_euler('xyz', degrees=False)

def get_relative_pose(pose1, pose2):
    # Get relative pose of pose2 wrt pose1
    pose1 = np.array(pose1)
    pose2 = np.array(pose2)
    relative_pose = np.dot(np.linalg.inv(pose1), pose2)
    return relative_pose
