import bpy

bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_plane_add(
    enter_editmode=False, align="WORLD", location=(0, 0, 0), scale=(1, 1, 1)
)

bpy.context.space_data.context = "MATERIAL"
bpy.ops.material.new()

bpy.context.object.active_material.name = "Material.001"

bpy.ops.image.open(
    filepath="/home/udaygirish/Projects/WPI/computer_vision/project4/p2/Visual-Inertial-Odometry/textures/blue_floor_tiles_01_diff_4k.jpg",
    directory="/home/udaygirish/Projects/WPI/computer_vision/project4/p2/Visual-Inertial-Odometry/textures/",
    files=[
        {
            "name": "blue_floor_tiles_01_diff_4k.jpg",
            "name": "blue_floor_tiles_01_diff_4k.jpg",
        }
    ],
    relative_path=True,
    show_multiview=False,
)


bpy.ops.image.open(
    filepath="/home/udaygirish/Projects/WPI/computer_vision/project4/p2/Visual-Inertial-Odometry/textures/aerial_rocks_01_diff_4k.jpg",
    directory="/home/udaygirish/Projects/WPI/computer_vision/project4/p2/Visual-Inertial-Odometry/textures/",
    files=[
        {"name": "aerial_rocks_01_diff_4k.jpg", "name": "aerial_rocks_01_diff_4k.jpg"}
    ],
    show_multiview=False,
)


directory = (
    "/home/udaygirish/Projects/WPI/computer_vision/project4/p2/Visual-Inertial-Odometry/textures/",
)
files = [{"name": "aerial_rocks_01_diff_4k.jpg", "name": "aerial_rocks_01_diff_4k.jpg"}]
