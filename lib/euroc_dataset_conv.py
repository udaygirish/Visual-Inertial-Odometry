import numpy as np 
import pandas 
import matplotlib.pyplot as plt
import yaml
import os 
import shutil
from tqdm import tqdm
from helpers import *

def get_relative_pose_quaternion(pose1, pose2):
    # Convert euler angles to rotation matrix
    R1 = convert_R_to_rotmat(convert_quaternion_to_R(pose1[3:]))
    R2 = convert_R_to_rotmat(convert_quaternion_to_R(pose2[3:]))

    T1 = np.array(pose1[:3])
    T2 = np.array(pose2[:3])

    # Get relative pose - Relative Rotation and Relative Translation
    relative_rot = np.dot(R1.T, R2)
    relative_pose = np.zeros(7)
    relative_pose[:3] = np.dot(R1.T, T2 - T1)
    relative_pose[3:] = convert_euler_to_quaternion(convert_rotmat_to_R(relative_rot))

    return relative_pose

BASE_PATH = "../../V1_01_easy/mav0/"

cam0_csv_path = BASE_PATH + "cam0/data.csv"
cam1_csv_path = BASE_PATH + "cam1/data.csv"

cam0_data_path = BASE_PATH + "cam0/data/"
cam1_data_path = BASE_PATH + "cam1/data/"

cam0_sensor = BASE_PATH + "cam0/sensor.yaml"
cam1_sensor = BASE_PATH + "cam1/sensor.yaml"

imu0_csv_path = BASE_PATH + "imu0/data.csv" 

imu0_sensor = BASE_PATH + "imu0/sensor.yaml"

state_groundtruth = BASE_PATH + "state_groundtruth_estimate0/data.csv"
state_groundyaml = BASE_PATH + "state_groundtruth_estimate0/sensor.yaml"

body_yaml = BASE_PATH + "body.yaml"

vicon0_csv_path = BASE_PATH + "vicon0/data.csv"

vicon0_sensor = BASE_PATH + "vicon0/sensor.yaml"

OUT_DATA_BASE_PATH = "../../V1_01_easy/out_data/"
out_img_path = OUT_DATA_BASE_PATH + "images/"
out_imu_path = OUT_DATA_BASE_PATH + "imu.csv"
out_gt_path = OUT_DATA_BASE_PATH + "trajectory_pose.csv"
out_rel_path = OUT_DATA_BASE_PATH + "relative_pose.csv"

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)

def load_data(csv_path):
    return pandas.read_csv(csv_path)

def load_yaml(yaml_path):
    with open(yaml_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)

# Create Directory
create_dir(OUT_DATA_BASE_PATH)

# For now lets use camera 0 only 
cam0_data = load_data(cam0_csv_path)

# Sort the data by first column
cam0_data = cam0_data.sort_values(by=[cam0_data.columns[0]])


imu0_data = load_data(imu0_csv_path)

# Sort the data by first column
imu0_data = imu0_data.sort_values(by=[imu0_data.columns[0]])


state_ground_data = load_data(state_groundtruth)

# Sort the data by first column
state_ground_data = state_ground_data.sort_values(by=[state_ground_data.columns[0]])


vicon0_data = load_data(vicon0_csv_path)

# Sort the data by first column
vicon0_data = vicon0_data.sort_values(by=[vicon0_data.columns[0]])


cam0_sensor_config = load_yaml(cam0_sensor)
cam1_sensor_config = load_yaml(cam1_sensor)
imu0_sensor_config = load_yaml(imu0_sensor)
state_ground_config = load_yaml(state_groundyaml)
vicon0_sensor_config = load_yaml(vicon0_sensor)
body_config = load_yaml(body_yaml)

# Print length of data
print("Length of cam0 data: ", len(cam0_data))
print("Length of imu0 data: ", len(imu0_data))
print("Length of state ground data: ", len(state_ground_data))
print("Length of vicon0 data: ", len(vicon0_data))


# Ingest at full fps 
ingest_fps_factor = 1

# Get all the timestamps from the data

cam0_timestamps = cam0_data[cam0_data.columns[0]].values

# Get all images from the data
cam0_images = cam0_data[cam0_data.columns[1]].values

# Copy the images to the out folder
create_dir(out_img_path)

for i in range(len(cam0_images)):
    shutil.copy(cam0_data_path + cam0_images[i], out_img_path)


# In IMU values shift the first three columns to last and last three columns to first
# 2, 3, 4 - 5, 6, 7, and 5, 6, 7 - 2, 3, 4

imu0_data = imu0_data[[imu0_data.columns[0], imu0_data.columns[4], imu0_data.columns[5], imu0_data.columns[6], imu0_data.columns[1], imu0_data.columns[2], imu0_data.columns[3]]]

# Change the column names - #timestamp, ax, ay, az, gx, gy, gz

imu0_data.columns = ["# timestamp", "ax", "ay", "az", "gx", "gy", "gz"]


# Find all poses from the vicon data for all timestamps of cam0 data

vicon0_timestamps = vicon0_data[vicon0_data.columns[0]].values

vicon0_poses = []

for i in range(len(cam0_timestamps)):
    cam0_time = cam0_timestamps[i]
    vicon0_time = vicon0_timestamps[np.argmin(np.abs(vicon0_timestamps - cam0_time))]
    vicon0_pose = vicon0_data[vicon0_data[vicon0_data.columns[0]] == vicon0_time]
    vicon0_poses.append(vicon0_pose)


# Convert to dataframe
vicon0_poses = pandas.concat(vicon0_poses)


# Change the column names - # timestamp, tx, ty, tz, qx, qy, qz, qw

vicon0_poses.columns = ["# timestamp", "tx", "ty", "tz", "qw", "qx", "qy", "qz"]



# Swap qw qx qy qz to qx qy qz qw
vicon0_poses = vicon0_poses[[vicon0_poses.columns[0], vicon0_poses.columns[1], vicon0_poses.columns[2], vicon0_poses.columns[3], vicon0_poses.columns[5], vicon0_poses.columns[6], vicon0_poses.columns[7], vicon0_poses.columns[4]]]

# Sort the data by first column
vicon0_poses = vicon0_poses.sort_values(by=[vicon0_poses.columns[0]])

# print("Vicon poses head: ", vicon0_poses.head())
# Find relative poses from the vicon poses

relative_poses = []

timestamps = vicon0_poses[vicon0_poses.columns[0]].values
for i in range(1, len(vicon0_poses)):
    pose1 = vicon0_poses.iloc[i-1].values
    pose2 = vicon0_poses.iloc[i].values
    # leave first timestamp
    pose1 = pose1[1:]
    pose2 = pose2[1:]

    relative_pose = get_relative_pose_quaternion(pose1, pose2)

    relative_poses.append(relative_pose)

# Convert to dataframe
relative_poses = pandas.DataFrame(relative_poses)

# Change the column names - tx, ty, tz, qx, qy, qz, qw and add timestamp column

relative_poses.columns = ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]

relative_poses["# timestamp"] = timestamps[1:]


# Shift timestamps to first column

relative_poses = relative_poses[["# timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]]


# print("Relative Poses head: ", relative_poses.head())



# Output Shape

print("Output Shape: ")

print("Images: ", len(cam0_images))
print("IMU: ", len(imu0_data))
print("Ground Truth: ", len(vicon0_poses))
print("Relative Poses: ", len(relative_poses))

# Save the data to csv

cam0_data.to_csv(out_img_path + "data.csv", index=False)
imu0_data.to_csv(out_imu_path, index=False)
vicon0_poses.to_csv(out_gt_path, index=False)
relative_poses.to_csv(out_rel_path, index=False)

print("Data saved to out folder...")