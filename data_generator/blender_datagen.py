# # Blender Data Generation Script

# # Generate IMU Data and Images for Visual Inertia Odometry
# # Author: Uday Girish Maradana

import bpy 
import random 
import csv 
import os
import sys 

BASE_PATH = "/home/udaygirish/Projects/WPI/computer_vision/project4/p2/Visual-Inertial-Odometry/"
sys.path.append(BASE_PATH)
sys.path.append(os.path.join(BASE_PATH, "lib"))

LIB_PATH = "/home/udaygirish/Projects/WPI/computer_vision/project3/"
sys.path.append(LIB_PATH+"blender-4.0.2-linux-x64/blender_env/lib/python3.10/site-packages")

import shutil
import math
# Import pyquaternion using scipy
# from scipy.spatial.transform import Rotation as R
from lib.helpers import *
from utils.imu_model import *   


# Create a large plane

class SceneGenerator:
    def __init__(self) -> None:
        self.description = "This class generates a scene with a floor plane and a camera"
        self.image_height = 480
        self.image_width = 480
        self.cam_focal_length = 500

    def delete_all_objects(self): 
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='MESH')
        bpy.ops.object.delete()
        # Delete all materials
        # bpy.ops.material.delete()
        # # Delete all textures
        # bpy.ops.texture.delete()

    def create_floor(self, location=(0, 0, 0), scale=(1, 1, 1)):
        bpy.ops.mesh.primitive_plane_add(enter_editmode=False, align="WORLD", location=location, scale=scale)
        bpy.context.object.name = "Floor"
        # Change scale of the floor
        bpy.data.objects["Floor"].scale = scale


    def create_material(self, texture_path):
        try:
            bpy.data.materials.new(name="FloorMaterial")
            directory = os.path.dirname(texture_path)
            bpy.ops.image.open(filepath=texture_path, directory=directory)
            texture_image = bpy.data.images.get(os.path.basename(texture_path))
            if texture_image:
                texture = bpy.data.textures.new(name="FloorTexture", type='IMAGE')
                texture.image = texture_image
                floor_material = bpy.data.materials.get("FloorMaterial")
                floor_material.use_nodes = True
                bsdf = floor_material.node_tree.nodes["Principled BSDF"]
                tex_image = floor_material.node_tree.nodes.new('ShaderNodeTexImage')
                tex_image.image = texture_image
                floor_material.node_tree.links.new(tex_image.outputs[0], bsdf.inputs[0])
                return floor_material
            else:
                print("Texture not found:", texture_path)
                return None
        except Exception as e:
            print("Error creating material:", e)
            return None

    # Create sequence of floor textures
    def create_multiple_floors(self, texture_paths, locations_list, scale_list):
        for texture_path, location,scale in zip(texture_paths, locations_list, scale_list):
            # Create floor
            self.create_floor(location, scale)
            floor_material = self.create_material(texture_path)
            # Assign material to the floor
            bpy.data.objects["Floor"].data.materials.append(floor_material)

    # Create a Rectangular Grid of Floor planes - x and y
    def create_grid_of_floors(self, texture_paths, grid_size=(10, 10), scale=(1, 1, 1)):
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # Create floor 
                self.create_floor((i*2, j*2, 0), scale)
                # print("Texture Paths: ", texture_paths[0])
                floor_material = self.create_material(texture_paths[0])
                # Assign material to the floor
                bpy.data.objects["Floor"].data.materials.append(floor_material)

    def get_calib_matrix(self):
        self.principal_point = (self.image_width/2, self.image_height/2)
        self.calib_matrix = [[self.cam_focal_length, 0, self.principal_point[0]], [0, self.cam_focal_length, self.principal_point[1]], [0, 0, 1]]

    def setup_camera_with_intrinsics(self):
        self.get_calib_matrix()
        calibration_matrix = self.calib_matrix
        # Set up camera
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 5), rotation=(0, 0, 0))
        
        # Set camera parameters based on calibration matrix
        bpy.context.object.data.type = 'PERSP'  # Change to perspective projection
        bpy.context.object.data.lens_unit = 'FOV'  # Set lens unit to field of view
        
        # Set focal length based on calibration matrix
        bpy.context.object.data.angle_x = 2 * math.atan(calibration_matrix[0][2] / calibration_matrix[0][0])  # Horizontal FOV
        bpy.context.object.data.angle_y = 2 * math.atan(calibration_matrix[1][2] / calibration_matrix[1][1])  # Vertical FOV
        
        # Set principal point
        bpy.context.object.data.shift_x = (calibration_matrix[0][2] - bpy.context.object.data.sensor_width / 2) / bpy.context.object.data.sensor_width
        bpy.context.object.data.shift_y = (calibration_matrix[1][2] - bpy.context.object.data.sensor_height / 2) / bpy.context.object.data.sensor_height

        # Set clipping distances
        bpy.context.object.data.clip_start = 0.1  # Near clipping distance
        bpy.context.object.data.clip_end = 100  # Far clipping distance

        # Set camera to look at the floor
        bpy.context.scene.camera = bpy.context.object
        bpy.context.view_layer.objects.active = bpy.context.object
        bpy.ops.view3d.camera_to_view_selected()

    # Set up camera such that it looks at the floor
    def setup_camera(self):
        # Set up camera
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 5), rotation=(0, 0, 0))
        bpy.context.object.data.type = 'ORTHO'
        bpy.context.object.data.ortho_scale = 2
        bpy.context.object.data.clip_end = 100

        # Set camera to look at the floor
        bpy.context.scene.camera = bpy.data.objects["Camera"]
        bpy.context.view_layer.objects.active = bpy.data.objects["Floor"]
        bpy.ops.view3d.camera_to_view_selected()

    # Move camera to a new location
    def move_camera(self, location):
        bpy.data.objects["Camera"].location = location

    def rotate_camera(self, rotation):
        # Rotate using Euler angles
        bpy.data.objects["Camera"].rotation_euler = rotation

    # Add a light source
    def add_sun_light(self):
        bpy.ops.object.light_add(type='SUN', location=(5, 5, 15))
        bpy.context.object.data.energy = 3
        bpy.context.object.data.angle = math.radians(45)
        bpy.context.object.data.use_shadow = True

    
    # Render settings 
    # Render both RGB and also in Material Preview mode
    # def render_and_save(self, render_path, render_rgb_path):
    #     bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    #     bpy.context.scene.render.resolution_x = 480
    #     bpy.context.scene.render.resolution_y = 480
    #     bpy.context.scene.render.image_settings.file_format = 'PNG'
    #     # Render in Material Preview mode
    #     bpy.context.space_data.shading.type = 'MATERIAL'
    #     bpy.context.scene.render.filepath = render_path
    #     bpy.ops.render.render(write_still=True)

    #     # Render in RGB mode
    #     bpy.context.space_data.shading.type = 'RENDERED'
    #     bpy.context.scene.render.filepath = render_rgb_path
    #     bpy.ops.render.render(write_still=True)
    def render_and_save(self, render_path, render_rgb_path):
        # Switch context to 3D viewport
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = 'MATERIAL'
                        break

        # Set rendering engine and resolution
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.render.resolution_x = 480
        bpy.context.scene.render.resolution_y = 480
        bpy.context.scene.render.image_settings.file_format = 'PNG'

        # Render in Material Preview mode
        bpy.context.scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

        # Switch context to 3D viewport again - Commented for now
        # for area in bpy.context.screen.areas:
        #     if area.type == 'VIEW_3D':
        #         for space in area.spaces:
        #             if space.type == 'VIEW_3D':
        #                 space.shading.type = 'RENDERED'
        #                 break

        # # Render in RGB mode
        # bpy.context.scene.render.filepath = render_rgb_path
        # bpy.ops.render.render(write_still=True)


    # Get Camera Intrinsic Parameters
    def get_camera_intrinsics(self):
        camera_object = bpy.context.scene.camera

        if camera_object is None:
            print("Error: No camera found in the scene.")
            return None

        if camera_object.data.type != 'ORTHO':
            print("Error: Camera is not in orthographic projection.")
            return None

        # Focal length (orthographic scale)
        focal_length = camera_object.data.ortho_scale

        # Principal point (assuming it's at the center of the image)
        principal_point_x = bpy.context.scene.render.resolution_x / 2
        principal_point_y = bpy.context.scene.render.resolution_y / 2

        # Image resolution
        image_width = bpy.context.scene.render.resolution_x
        image_height = bpy.context.scene.render.resolution_y

        # Construct the intrinsic matrix
        intrinsic_matrix = [
            [focal_length, 0, principal_point_x],
            [0, focal_length, principal_point_y],
            [0, 0, 1]
        ]

        return intrinsic_matrix

        
    


scenegen = SceneGenerator()

scenegen.delete_all_objects()

temp_texture_paths = "/home/udaygirish/Projects/WPI/computer_vision/project4/p2/Visual-Inertial-Odometry/textures/blue_floor_tiles_01_diff_4k.jpg"

texture_paths = [temp_texture_paths]*10

locations_list = [(0, 0, 0), (0, 2, 0), (0, 4, 0), (0, 6, 0), (0, 8, 0), (0, 10, 0), (0, 12, 0), (0, 14, 0), (0, 16, 0), (0, 18, 0)]
scale_list = [(1,1,1)]*10



# scenegen.create_multiple_floors(texture_paths, locations_list, scale_list)
scenegen.create_grid_of_floors(texture_paths, grid_size=(40, 40), scale=(1, 1, 1))

scenegen.setup_camera() # Sets up camera with Zero Radial distortion and Calibration matrix specified by focal length
scenegen.add_sun_light()

# # Generate a Random trajectory for the camera and save images and IMU data

trajectory_time = 1000

OUT_BASE_PATH =  "/home/udaygirish/Projects/WPI/computer_vision/project4/p2/Visual-Inertial-Odometry/data_t"
RENDER_BASE_PATH = "/home/udaygirish/Projects/WPI/computer_vision/project4/p2/Visual-Inertial-Odometry/data_t/images"
base_render_dir = "data_t"
render_dir = "render_images"
render_rgb_dir = "render_rgb_images"
imu_csv =  os.path.join(OUT_BASE_PATH, "imu.csv")
relative_pose_csv = os.path.join(OUT_BASE_PATH, "relative_pose.csv")
trajectory_pose_csv = os.path.join(OUT_BASE_PATH, "trajectory_pose.csv")


instrinsics = scenegen.get_camera_intrinsics()
print("Calibration Matrix (K): ", instrinsics)
global_gen_x = 0
global_gen_y = 0
global_gen_z = 5

global_gen_roll = 0
global_gen_pitch = 0
global_gen_yaw = 0

global_time = 0 

imu_rate = 100

linear_accel = 2 
angular_vel = 2

max_distance_per_traj_time = 2

# Lets not use this rather generate directly from the difference between final pose and initial pose 


# Here we limit the trajectory to not have roll and pitch
# We only have yaw

# Get the first image

temp_render_path = os.path.join(RENDER_BASE_PATH, "image_0.png")
temp_render_rgb_path = os.path.join(RENDER_BASE_PATH, "image_rgb_0.png")
scenegen.render_and_save(temp_render_path, temp_render_rgb_path)


def generate_imu_data(dx,dy, dz, dyaw, global_gen_x, global_gen_y, global_gen_z, global_gen_pitch, global_gen_roll, global_gen_yaw, imu_rate, global_time):
    t_array = []
    x_array = []
    y_array = []
    z_array = []
    roll_array = []
    pitch_array = []
    yaw_array = []

    t_array = [global_time+ i/imu_rate for i in range(100)]

    for i in range(102):
        dist_x = global_gen_x + dx * i/imu_rate
        dist_y = global_gen_y + dy * i/imu_rate
        dist_z = global_gen_z + dz * i/imu_rate
        x_array.append(dist_x)
        y_array.append(dist_y)
        z_array.append(dist_z)
    
    for i in range(101):
        yaw_angle = global_gen_yaw + dyaw * i/imu_rate
        roll_angle = global_gen_roll + dyaw * i/imu_rate
        pitch_angle = global_gen_pitch + dyaw * i/imu_rate
        if yaw_angle > 360:
            yaw_angle = yaw_angle - 360
        if yaw_angle < -360:
            yaw_angle = yaw_angle + 360
        if abs(roll_angle) > 30:
            # Get the sign of the angle
            roll_angle = 30 * (roll_angle/abs(roll_angle))
        if abs(pitch_angle) > 30:
            pitch_angle = 30 * (pitch_angle/abs(pitch_angle))
        roll_array.append(roll_angle)
        pitch_array.append(pitch_angle)        
        yaw_array.append(yaw_angle)

    # Output - Real Data - Pure linear model N*3
    accel_data = cal_linear_acc(x_array, y_array, z_array, imu_rate)

    # Output - Real Data - Pure angular model N*3
    gyro_data = cal_angular_vel(roll_array, pitch_array, yaw_array, imu_rate)

    # Synthetic data- Real - Adding Error and Vibration according to model
    fs = 100
    acc_err = accel_high_accuracy
    gyro_err = gyro_high_accuracy

    accel_env = '[0.03 0.001 0.01]-random'
    gyro_env = '[6 5 4]d-0.5Hz-sinusoidal'

    vib_accel_def = vib_from_env(accel_env, fs)
    
    vib_gyro_def = vib_from_env(gyro_env, fs)

    # Output - Synthetic Data - Adding Error and Vibration according to model
    real_acc = acc_gen(fs, accel_data, acc_err, vib_accel_def)
    real_gyro = gyro_gen(fs, gyro_data, gyro_err, vib_gyro_def)

    return t_array, real_acc, real_gyro

def get_relative_pose(pose1, pose2):
    # Get relative pose
    pose1 = np.array(pose1)
    pose2 = np.array(pose2)

    relative_pose = np.dot(np.linalg.inv(pose1), pose2)
    return relative_pose

def get_relative_pose_quaternion(pose1, pose2):
    # Convert euler angles to rotation matrix
    R1 = convert_R_to_rotmat(pose1[3:])
    R2 = convert_R_to_rotmat(pose2[3:])

    T1 = np.array(pose1[:3])
    T2 = np.array(pose2[:3])

    # Get relative pose - Relative Rotation and Relative Translation
    relative_rot = np.dot(R1.T, R2)
    relative_pose = np.zeros(7)
    relative_pose[:3] = np.dot(R1.T, T2 - T1)
    relative_pose[3:] = convert_euler_to_quaternion(convert_rotmat_to_R(relative_rot))

    return relative_pose


# timestamp tx ty tz qx qy qz qw
with open(imu_csv, mode='a') as file:
    writer = csv.writer(file)
    writer.writerow(["# timestamp","ax", "ay", "az", "gx", "gy", "gz"])

# timestamp tx ty tz qx qy qz qw
with open(relative_pose_csv, mode='a') as file:
    writer = csv.writer(file)
    writer.writerow(["# timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])

with open(trajectory_pose_csv, mode='a') as file:
    writer = csv.writer(file)
    writer.writerow(["# timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])


for i in range(trajectory_time):
    # Move camera to a new location
    # Get random location with addition of error 
    dx = random.uniform(-0.01, 0.2) 
    dy = random.uniform(-0.01, 0.2)
    #dz = random.uniform(-0.005, 0.005)
    dz = 0

    pose1 = [global_gen_x, global_gen_y, global_gen_z, global_gen_roll, global_gen_pitch, global_gen_yaw]

    dyaw = random.uniform(-10,10)
    droll = random.uniform(-5,5)
    dpitch = random.uniform(-5,5)


    # Generate IMU data
    t_array, real_acc, real_gyro = generate_imu_data(dx, dy, dz, dyaw, global_gen_x, global_gen_y, global_gen_z, global_gen_pitch, global_gen_roll, global_gen_yaw, imu_rate, global_time)

    global_gen_x += dx
    global_gen_y += dy
    global_gen_z += dz

    global_gen_yaw += dyaw
    global_gen_pitch += dpitch
    global_gen_roll += droll

    global_time += 1

    pose2 = [global_gen_x, global_gen_y, global_gen_z, global_gen_roll, global_gen_pitch, global_gen_yaw]

    with open(imu_csv, mode='a') as file:
        writer = csv.writer(file)
        for t, acc, gyro in zip(t_array, real_acc, real_gyro):
            writer.writerow([t, acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2]])
    
    # Relative Pose Calculation in Quaternion
    relative_pose = get_relative_pose_quaternion(pose1, pose2)

    with open(relative_pose_csv, mode='a') as file:
        # Write relative pose with respect to the first pose
        writer = csv.writer(file)
        writer.writerow([global_time , relative_pose[0], relative_pose[1], relative_pose[2], relative_pose[3], relative_pose[4], relative_pose[5], relative_pose[6]])

    with open(trajectory_pose_csv, mode='a') as file:
        # Write relative pose with respect to the first pose
        writer = csv.writer(file)
        # Convert pose2 to quaternion
        pose2_new = convert_euler_to_quaternion(pose2[3:])
        writer.writerow([global_time , pose2[0], pose2[1], pose2[2], pose2_new[0], pose2_new[1], pose2_new[2], pose2_new[3]])
    
    scenegen.move_camera((global_gen_x, global_gen_y, global_gen_z))
    scenegen.rotate_camera((0,0, math.radians(global_gen_yaw)))
    temp_render_path = os.path.join(RENDER_BASE_PATH, "image_{}.png".format(i+1))
    temp_render_rgb_path = os.path.join(RENDER_BASE_PATH, "image_rgb_{}.png".format(i+1))   
    scenegen.render_and_save(temp_render_path, temp_render_rgb_path)
    print(f"Generated Image {i+1}")


