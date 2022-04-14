# This script builds on top of load_assets_mult.py
# Add support for loading an entire map from yaml config files
# Change direction on up axis from Y to Z (to be congruent with base/Vec_task)
# Refactor of cached tensors in several aspects

from mimetypes import init
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from helper import map_to_coord

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import yaml


# some comments


# intiailze gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="test file for loading summit_xl, multiple envs")

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.relaxation = 0.9
    sim_params.flex.dynamic_friction = 0.0
    sim_params.flex.static_friction = 0.0
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing GPU pipeline.")

# define simulation
sim = gym.create_sim(args.compute_device_id,
                     args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 0.1
plane_params.dynamic_friction = 0.1
plane_params.restitution = 0.
gym.add_ground(sim, plane_params)

###################### Create environment ######################

# Create some lists to keep track of all the assets
prim_names = ['robot', 'boxes', 'walls']  # note that walls is a dictionary
gym_assets = dict.fromkeys(prim_names)
gym_indices = dict.fromkeys(prim_names)

asset_root = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../assets")
room_cfg_root = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../cfg/rooms")
# summit_xl asset
asset_file = "summit_xl_description/robots/summit_xls_std_fixed_camera.urdf"
# wall config file
room_file = 'room_0.yaml'
with open(f'{room_cfg_root}/{room_file}', 'r') as f:
    room_config = yaml.load(f, Loader=yaml.loader.SafeLoader)

map_coords = map_to_coord(room_config)

# Load Summit
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.use_mesh_materials = False
asset_options.flip_visual_attachments = False
summit_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
gym_assets['robot'] = summit_asset

summit_sensor_contact_pose = gymapi.Transform(
    gymapi.Vec3(0., 0., 0.))
body_idx = gym.find_asset_rigid_body_index(summit_asset, 'summit_xl_base_link')
summit_sensor_front = gym.create_asset_force_sensor(
    summit_asset, body_idx, summit_sensor_contact_pose)

# Load box(es)
box_assets = []
box_width = 1
box_density = 20.
asset_options = gymapi.AssetOptions()
asset_options.density = box_density
asset_box = gym.create_box(sim, box_width, box_width, box_width, asset_options)
box_assets.append(asset_box)
gym_assets['boxes'] = box_assets


# Load wall(s)
wall_assets = {}  # a set with length as key
room_walls = room_config['walls']
wall_height = room_config['height']
wall_thickness = room_config['thickness']
wall_widths = set([wall['length'] for (name, wall) in room_walls.items()])
wall_asset_options = gymapi.AssetOptions()
wall_asset_options.fix_base_link = True
# wall_asset_options.density = 1000.
for width in wall_widths:
    asset_wall = gym.create_box(
        sim, wall_thickness, width, wall_height, wall_asset_options)
    wall_assets[width] = asset_wall
gym_assets['walls'] = wall_assets


# set up the env grid
num_envs = 10
num_per_row = 5
spacing = 12
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(25, 20.0, 30)
cam_target = gymapi.Vec3(5, -2.5, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []
box_handles = []
wall_handles = []
sensors = []

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    initial_pose = gymapi.Transform()
    initial_pose.p = gymapi.Vec3(-7.5, 2.5, 0.)
    initial_pose.r = gymapi.Quat(0., 0., -1., 1.)

    actor_handle = gym.create_actor(
        env, gym_assets['robot'], initial_pose, 'summit', i, 0)
    sensor = gym.get_actor_force_sensor(env, actor_handle, 0)
    sensors.append(sensor)
    actor_handles.append(actor_handle)

    box_handle = gym.create_actor(env, gym_assets['boxes'][0], gymapi.Transform(
        p=gymapi.Vec3(-7.5, -1, box_width/2)), 'box', i, 0)
    box_handles.append(box_handle)

    box_shape_props = gym.get_actor_rigid_shape_properties(env, box_handle)
    # change properties
    box_shape_props[0].compliance = 0.
    box_shape_props[0].friction = 0.1
    box_shape_props[0].rolling_friction = 0.
    box_shape_props[0].torsion_friction = 0.
    gym.set_actor_rigid_shape_properties(env, box_handle, box_shape_props)

    box_body_props = gym.get_actor_rigid_body_properties(env, box_handle)
    box_body_props[0].mass *= 1
    gym.set_actor_rigid_body_properties(env, box_handle, box_body_props)

    # Add wall
    for (name, wall) in room_walls.items():
        wall_name = name
        wall_thickness = wall_thickness
        wall_length = wall['length']
        wall_orientation = wall['orientation']
        wall_pos_x = wall['pos_x']
        wall_pos_y = wall['pos_y']

        pos = gymapi.Transform()
        pos.p = gymapi.Vec3(wall_pos_x, wall_pos_y, 1.0)

        if wall_orientation == 'horizontal':
            pos.r = gymapi.Quat(1.0, -1.0, 0.0, 0.0)

        wall_handle = gym.create_actor(
            env, gym_assets['walls'][wall_length], pos, wall_name, i, 0
        )
        wall_handles.append(wall_handle)

    # set default DOF positions
    # gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

    # Configure DOF
    props = gym.get_actor_dof_properties(env, actor_handle)
    # the dofs are in the following order:
    # back_left, back_right, front_left, front_right
    props["driveMode"].fill(gymapi.DOF_MODE_VEL)

    props["stiffness"].fill(0.0)
    props['damping'].fill(1000.0)
    gym.set_actor_dof_properties(env, actor_handle, props)

    # Set DOF drive targets
    front_left_wheel_handle0 = gym.find_actor_dof_handle(
        env, actor_handle, 'summit_xl_front_left_wheel_joint')
    front_right_wheel_handle0 = gym.find_actor_dof_handle(
        env, actor_handle, 'summit_xl_front_right_wheel_joint')
    back_left_wheel_handle0 = gym.find_actor_dof_handle(
        env, actor_handle, 'summit_xl_back_left_wheel_joint')
    back_right_wheel_handle0 = gym.find_actor_dof_handle(
        env, actor_handle, 'summit_xl_back_right_wheel_joint')

    # # Control DOF to make robot move forward
    velocity = 1
    gym.set_dof_target_velocity(env, back_left_wheel_handle0, velocity)
    gym.set_dof_target_velocity(env, back_right_wheel_handle0, velocity)
    gym.set_dof_target_velocity(env, front_left_wheel_handle0, velocity)
    gym.set_dof_target_velocity(env, front_right_wheel_handle0, velocity)

# Handles are just indices (but useful when there are multiple envs)
print(f'env handles: {envs}')
print(f'summit actor handle: {actor_handles}')
print(f'box handles: {box_handles}')
print(f'wall handles: {wall_handles}')

# Acquire global tensors
_actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
_dof_state_tensor = gym.acquire_dof_state_tensor(sim)
_force_sensor_tensor = gym.acquire_force_sensor_tensor(sim)


# To read these tensors:
actor_root_state_tensor = gymtorch.wrap_tensor(_actor_root_state_tensor)
dof_state_tensor = gymtorch.wrap_tensor(_dof_state_tensor)
dof_vel_tensor = dof_state_tensor[:, 1]
force_sensor_tensor = gymtorch.wrap_tensor(_force_sensor_tensor)

print(actor_root_state_tensor.size())
print(dof_state_tensor.size())

print(actor_root_state_tensor[actor_handles].size())

# Let's take a look at the robot in the first env
summit_state_tensor = actor_root_state_tensor[actor_handles[0]]
summit_pos_tensor = summit_state_tensor[0:3]
summit_vel_tensor = summit_state_tensor[7:10]

summit_front_sensor_tensor = force_sensor_tensor[0][0:3]
summit_front_torque_sensor_tensor = force_sensor_tensor[0][3:7]

print(summit_state_tensor)


# Simulate
t = 0
log_dt = 10

# create some lists to hold data to be ploted later
summit_pos_x, summit_pos_y = [], []
summit_vel, summit_wheel_vel = [], []
sensor_forces, sensor_torques = [], []

# check the effect of box weight and environment friction
while not gym.query_viewer_has_closed(viewer):

    if (t - 99) % log_dt == 0:
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_force_sensor_tensor(sim)
        vel_y = summit_vel_tensor[1]
        # print(f'wheel: {dof_vel_tensor[0]}')
        # print(f'pos_y: {summit_pos_tensor[1]}')
        # print(f'vel_y: {vel_y}')
        summit_pos_x.append(summit_pos_tensor[0].item())
        summit_pos_y.append(summit_pos_tensor[1].item())
        summit_vel.append(vel_y.abs().item())
        summit_wheel_vel.append(torch.mean(dof_vel_tensor[0]).item())

        sensor_data = summit_front_sensor_tensor
        sensor_forces.append(sensor_data.tolist())
        sensor_torques.append(summit_front_torque_sensor_tensor.tolist())

        # print([summit_pos_x[-1], summit_pos_y[-1]])

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    if t == 6000:
        break
    t += 1

print('Simulation Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

# plot position of summit to check that coord is correct
plt.plot(summit_pos_x, summit_pos_y)
plt.show()

quit()

for f in sensor_forces:
    print(f)

[sensor_fx, sensor_fy, sensor_fz] = list(zip(*sensor_forces))
[sensor_tx, sensor_ty, sensor_tz] = list(zip(*sensor_torques))


plt.subplot(1, 3, 1)
plt.title('force x')
plt.plot(sensor_tx)

plt.subplot(1, 3, 2)
plt.title('force y')
plt.plot(sensor_ty)

plt.subplot(1, 3, 3)
plt.title('force z')
plt.plot(sensor_tz)

# plt.subplot(1, 2, 1)
# plt.plot(summit_vel)

# plt.subplot(1, 2, 2)
# plt.plot(summit_wheel_vel)

plt.show()
