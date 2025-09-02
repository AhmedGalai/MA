import bpy
import random
import math
import mathutils

from rich.progress import track
# -----------------------------
# Clean scene
# -----------------------------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# -----------------------------
# Create a simple room
# -----------------------------
def create_room(size=10, height=3):
    # Floor
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0,0,0))
    floor = bpy.context.object
    floor.name = "Floor"

    # Roof
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0,0,height))
    roof = bpy.context.object
    roof.name = "Roof"

    # Walls (4 sides)
    # Front wall
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, size/2, height/2))
    wall_front = bpy.context.object
    wall_front.rotation_euler[0] = math.radians(90)

    # Back wall
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, -size/2, height/2))
    wall_back = bpy.context.object
    wall_back.rotation_euler[0] = math.radians(90)

    # Left wall
    bpy.ops.mesh.primitive_plane_add(size=size, location=(-size/2, 0, height/2))
    wall_left = bpy.context.object
    wall_left.rotation_euler[1] = math.radians(90)

    # Right wall
    bpy.ops.mesh.primitive_plane_add(size=size, location=(size/2, 0, height/2))
    wall_right = bpy.context.object
    wall_right.rotation_euler[1] = math.radians(90)
create_room()

# -----------------------------
# Random objects with random materials
# -----------------------------
def add_random_object():
    obj_type = random.choice(["cube", "sphere", "cone"])
    if obj_type == "cube":
        bpy.ops.mesh.primitive_cube_add(location=(random.uniform(-3,3), random.uniform(-3,3), 1))
    elif obj_type == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(location=(random.uniform(-3,3), random.uniform(-3,3), 1))
    else:
        bpy.ops.mesh.primitive_cone_add(location=(random.uniform(-3,3), random.uniform(-3,3), 1))
    obj = bpy.context.object
    
    # Random material
    mat = bpy.data.materials.new(name="RandomMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]

    # Assign random base color
    bsdf.inputs['Base Color'].default_value = (random.random(), random.random(), random.random(), 1)
    
    # Random roughness
    bsdf.inputs['Roughness'].default_value = random.random()
    
    # Handle Blender version differences
    if "Specular" in bsdf.inputs.keys():
        bsdf.inputs['Specular'].default_value = random.random()
    elif "Specular IOR Level" in bsdf.inputs.keys():  # Blender 4.0+
        bsdf.inputs['Specular IOR Level'].default_value = random.random()

    obj.data.materials.append(mat)

for _ in range(5):
    add_random_object()

for _ in range(3):
    bpy.ops.object.light_add(
        type='POINT',
        location=(random.uniform(-3,3), random.uniform(-3,3), random.uniform(1.5,2.5))
    )
    light = bpy.context.object
    light.data.energy = random.uniform(500, 1500)

# Camera inside the room
bpy.ops.object.camera_add(location=(0, -2, 1.5), rotation=(math.radians(75), 0, 0))
camera = bpy.context.object
camera.data.lens = 35
camera.data.sensor_width = 36
camera.data.sensor_height = 24
bpy.context.scene.camera = camera

# Circle path inside the room
bpy.ops.curve.primitive_bezier_circle_add(radius=2.5, location=(0,0,1.5))
path = bpy.context.object
path.name = "CameraPath"
# Add follow path constraint
constraint = camera.constraints.new(type='FOLLOW_PATH')
constraint.target = path
constraint.use_fixed_location = True
constraint.use_curve_follow = True

# Animate along path
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 60  # 5 seconds at 24fps

# Animate camera following path
if hasattr(constraint, "offset_factor"):  # Blender 4.x
    constraint.offset_factor = 0.0
    constraint.keyframe_insert(data_path="offset_factor", frame=1)
    constraint.offset_factor = 1.0
    constraint.keyframe_insert(data_path="offset_factor", frame=120)
else:  # Blender ≤3.6
    constraint.offset = 0
    constraint.keyframe_insert(data_path="offset", frame=1)
    constraint.offset = 100
    constraint.keyframe_insert(data_path="offset", frame=120)

# Z rotation (yaw spin)
camera.rotation_mode = 'XYZ'
camera.keyframe_insert(data_path="rotation_euler", frame=1)
camera.rotation_euler[2] += math.radians(90)
camera.keyframe_insert(data_path="rotation_euler", frame=120)


# -----------------------------
# Render settings
# -----------------------------
scene = bpy.context.scene
scene.render.engine = 'CYCLES'  # or 'BLENDER_EEVEE'
scene.cycles.samples = 2       # lower for speed
scene.render.resolution_x = 640
scene.render.resolution_y = 480
scene.render.fps = 6            # we want 6 fps

scene.render.image_settings.file_format = 'PNG'

# Render 6 fps for 10 seconds = 60 frames
scene.frame_start = 1
scene.frame_end = 60

# -----------------------------
# Manual rendering with progress bar
# -----------------------------
from rich.progress import track
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

rendered_frames = []

for frame in track(range(scene.frame_start, scene.frame_end+1), description="Rendering frames..."):
    scene.frame_set(frame)
    filepath = f"//renders/frame_{frame:04d}.png"
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    rendered_frames.append(filepath)

# -----------------------------
# Show rendered frames as a quick video preview
# -----------------------------
fig, ax = plt.subplots()
ax.axis("off")

for img_path in rendered_frames[::10]:  # show every 10th frame
    img = mpimg.imread(bpy.path.abspath(img_path))
    ax.imshow(img)
    plt.pause(0.5)

plt.show()

