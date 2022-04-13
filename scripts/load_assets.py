# This script loads the summit_xl_steel robot, a box, and a fixed wall
# DOF of summit XL Steel is controlled w/ target vel, and summit pushes a box into a wall

# Note: run this file from within the /script folder

from mimetypes import init

from isaacgym import gymapi
from isaacgym import gymutil

# intiailze gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="test file for loading summit_xl")

# configure sim
sim_params = gymapi.SimParams()
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
plane_params.static_friction = 0.5
plane_params.dynamic_friction = 0.5
gym.add_ground(sim, gymapi.PlaneParams())

# set up the env grid
num_envs = 1
spacing = 0.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Load summit_xl asset
asset_root = "../assets"
asset_file = "summit_xl_description/robots/summit_xls_std_0.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = True
# asset_options.use_mesh_materials = True
asset_options.flip_visual_attachments = False


print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
summit_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Load box
box_size = 0.8
box_density = 20.
asset_options = gymapi.AssetOptions()
asset_options.density = box_density
asset_box = gym.create_box(sim, box_size, box_size, box_size, asset_options)

# Load wall
wall_asset_options = gymapi.AssetOptions()
wall_asset_options.fix_base_link = True
# wall_asset_options.density = 1000.
asset_wall = gym.create_box(
    sim, 0.1, 3., 5., wall_asset_options)

# Create environment
env0 = gym.create_env(sim, env_lower, env_upper, 0)

# Add summit
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0, 0, 0)
initial_pose.r = gymapi.Quat(-1.0, 0.0, 0.0, 1.0)

summit0 = gym.create_actor(env0, summit_asset, initial_pose, 'summit', 0, 0)

# Add box and set shape physics
box0 = gym.create_actor(env0, asset_box, gymapi.Transform(
    p=gymapi.Vec3(4., 0.5, 0.)), 'box', 0, 0)

# Set some physics properties
shape_props = gym.get_actor_rigid_shape_properties(
    env0, box0)
shape_props[0].friction = 1.
shape_props[0].rolling_friction = 0.
shape_props[0].torsion_friction = 0.
gym.set_actor_rigid_shape_properties(
    env0, box0, shape_props)

# Add wall
wall0 = gym.create_actor(env0, asset_wall, gymapi.Transform(
    p=gymapi.Vec3(6., 0., 0.)), 'wall', 0, 0)

# gym.set_rigid_linear_velocity(env0,
#                               gym.get_rigid_handle(
#                                   env0, 'box', gym.get_actor_rigid_body_names(
#                                       env0, box0)),
#                 collision group that actor will be part of. The actor will not collide with anything outside of the same collisionGroup              gymapi.Vec3(0., 0., 2.))


# Configure DOF
props = gym.get_actor_dof_properties(env0, summit0)
# the dofs are in the following order:
# back_left, back_right, camera, front_left, front_right, camera
props["driveMode"].fill(gymapi.DOF_MODE_VEL)
props["stiffness"].fill(0.0)
props['damping'].fill(1000.0)
gym.set_actor_dof_properties(env0, summit0, props)

# Set DOF drive targets
front_left_wheel_handle0 = gym.find_actor_dof_handle(
    env0, summit0, 'summit_xl_front_left_wheel_joint')
front_right_wheel_handle0 = gym.find_actor_dof_handle(
    env0, summit0, 'summit_xl_front_right_wheel_joint')
back_left_wheel_handle0 = gym.find_actor_dof_handle(
    env0, summit0, 'summit_xl_back_left_wheel_joint')
back_right_wheel_handle0 = gym.find_actor_dof_handle(
    env0, summit0, 'summit_xl_back_right_wheel_joint')

# Control DOF to make robot move forward
velocity = 2
gym.set_dof_target_velocity(env0, back_left_wheel_handle0, velocity)
gym.set_dof_target_velocity(env0, back_right_wheel_handle0, velocity)
gym.set_dof_target_velocity(env0, front_left_wheel_handle0, velocity)
gym.set_dof_target_velocity(env0, front_right_wheel_handle0, velocity)


summit_bodies = gym.get_asset_rigid_body_count(summit_asset)
summit_dofs = gym.get_asset_dof_count(summit_asset)

print(f'num summit bodies: {summit_bodies}, num summit dofs: {summit_dofs}')

# Simulate
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

print('Simulation Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
