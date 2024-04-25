# # Blender Data Generation Script

# # Generate IMU Data and Images for Visual Inertia Odometry
# # Author: Uday Girish Maradana

import bpy 
import random 
import csv 
import os
import shutil
import math
# Import pyquaternion using scipy
# from scipy.spatial.transform import Rotation as R
#from lib.helpers import *


# Create a large plane

class SceneGenerator:
    def __init__(self) -> None:
        self.description = "This class generates a scene with a floor plane and a camera"

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

    # Set up camera such that it looks at the floor
    def setup_camera(self):
        # Set up camera
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 5), rotation=(1.5708, 0, 0))
        bpy.context.object.data.type = 'ORTHO'
        bpy.context.object.data.ortho_scale = 6
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

    
    # Render settings 
    # Render both RGB and also in Material Preview mode
    def render_and_save(self, render_path, render_rgb_path):
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        bpy.context.scene.render.resolution_x = 480
        bpy.context.scene.render.resolution_y = 480
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        # Render in Material Preview mode
        bpy.context.space_data.shading.type = 'MATERIAL'
        bpy.context.scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

        # Render in RGB mode
        bpy.context.space_data.shading.type = 'RENDERED'
        bpy.context.scene.render.filepath = render_rgb_path
        bpy.ops.render.render(write_still=True)
        
    


scenegen = SceneGenerator()

scenegen.delete_all_objects()

temp_texture_paths = "/home/udaygirish/Projects/WPI/computer_vision/project4/p2/Visual-Inertial-Odometry/textures/blue_floor_tiles_01_diff_4k.jpg"

texture_paths = [temp_texture_paths]*10

locations_list = [(0, 0, 0), (0, 2, 0), (0, 4, 0), (0, 6, 0), (0, 8, 0), (0, 10, 0), (0, 12, 0), (0, 14, 0), (0, 16, 0), (0, 18, 0)]
scale_list = [(1,1,1)]*10



# scenegen.create_multiple_floors(texture_paths, locations_list, scale_list)
scenegen.create_grid_of_floors(texture_paths, grid_size=(5, 5), scale=(1, 1, 1))

scenegen.setup_camera()


# # Generate a Random trajectory for the camera and save images and IMU data

trajectory_time = 100000

base_render_dir = "data_t"
render_dir = "render_images"
render_rgb_dir = "render_rgb_images"
imu_csv = "imu_data.csv"



global_gen_x = 0
global_gen_y = 0
global_gen_z = 0

global_gen_roll = 0
global_gen_pitch = 0
global_gen_yaw = 0


# Here we limit the trajectory to not have roll and pitch
# We only have yaw

for i in range(trajectory_time):
    # Move camera to a new location
    # Get random location with addition of error 
    dx = random.uniform(0, 2)
    dy = random.uniform(0, 2)
    dz = random.uniform(0, 2)

    global_gen_x += dx
    global_gen_y += dy
    global_gen_z += dz
    
    scenegen.move_camera((random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(5, 10)))
    scenegen.rotate_camera((0,0, random.uniform(0, 3.14)))
    scenegen.render_and_save(f"/home/udaygirish/Projects/WPI/computer_vision/project4/p2/Visual-Inertial-Odometry/data_generator/images/image_{i}.png")
    print(f"Generated Image {i}")


