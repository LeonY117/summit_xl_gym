# version 1 of summit xl steel class
# the code is adapted from ant.py from isaacgymenvs

from typing import Tuple
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *
from isaacgymenvs.utils.torch_jit_utils import *


import numpy as np
import yaml
import os
import torch

from .base.vec_task import VecTask  # pre-defined abstract class
from .helper import load_room_from_config


class Summit(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        """
        Initialises the class.

        Args:
           config: config dictionary for the environment.
           sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
           graphics_device_id: the device ID to render with.
           headless: Set to False to disable viewer rendering.
        """
        # ---------------- GLOBAL CONFIG ----------------
        self.cfg = cfg  # OmegaConf & Hydra Config

        # load room configuration (in future we can merge this with global config)
        room_cfg_root = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "../cfg/rooms")
        room_cfg_file = 'room_0.yaml'
        with open(f'{room_cfg_root}/{room_cfg_file}', 'r') as f:
            self.room_config = yaml.load(f, Loader=yaml.loader.SafeLoader)
        # load data from config file using helper function
        map_coords, goal_pos, goal_radius = load_room_from_config(
            self.room_config)
        self.goal_pos = to_torch(
            [goal_pos], device=self.device, dtype=torch.float).repeat((self.num_envs, 1))
        self.goal_radius = to_torch(
            goal_radius, device=self.device, dtype=torch.float)
        self.map_coords = to_torch(
            map_coords, device=self.device, dtype=torch.float).flatten().repeat((self.num_envs, 1))

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        # TODO: initiate numObservations dynamically from room_config
        self.cfg["env"]["numObservations"] = 65
        self.cfg["env"]["numActions"] = 4

        # to be configured later, randomize is set to false for now
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        self.power_scale = self.cfg["env"]["powerScale"]

        # reward parameters

        # cost parameters

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # plane parameters
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        super().__init__(config=self.cfg, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless)

        # set camera angle
        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

        # TODO: do we need other buffers (e.g. force)?

        # Get gym GPU buffers
        # retrieves buffer for actor rot states, with the shape (num_env, num_actors * 13)
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(
            self.sim)
        # retrieves buffer for dof state, with shape (num_env, num_dofs * 2)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        # populates tensors with latest data from the simulator
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # store the initial state, used for resetting the agent
        self.actor_root_state_tensor = gymtorch.wrap_tensor(
            _actor_root_state_tensor)
        # TODO: this needs to change: we have initial pos for both agent and boxes
        self.initial_root_states = self.actor_root_state_tensor.clone()

        # create some wrapper tensors for different slices
        # dof state tensors store pos and vel for each dof
        self.dof_state_tensor = gymtorch.wrap_tensor(_dof_state_tensor)
        self.dof_pos = self.dof_state_tensor.view(
            self.num_envs, self.summit_num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state_tensor.view(
            self.num_envs, self.summit_num_dofs, 2)[..., 1]

        self.initial_dof_pos = torch.zeros_like(
            self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        # snap pos constraints to initial pos (if initial dof pos with all zeros violate constraint)
        # self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
        #                                    torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(
            self.dof_vel, device=self.device, dtype=torch.float)

        # initialize some data used later on
        self.up_vec = to_torch(get_axis_params(
            1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))

        self.global_indices = torch.arange(
            self.num_envs * (10), dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.dt = self.cfg["sim"]["dt"]

        # create some summit tensors to be used later
        print(self.actor_root_state_tensor.size())

        self.summit_state_tensor = self.actor_root_state_tensor.view(
            self.num_envs, -1, 13)[:, 0, :13]

        # self.summit_state_tensor = self.actor_root_state_tensor.view(
        #     self.num_envs, -1, 13)[list(zip(*[torch.arange(self.num_envs), self.actor_handles]))]

        # print(self.summit_state_tensor.size())

        # self.actor_root_pos_tensor = self.actor_root_state_tensor[..., 0:3]
        # self.actor_root_rot_tensor = self.actor_root_state_tensor[..., 3:7]
        # self.actor_root_vel_tensor = self.actor_root_state_tensor[..., 7:10]
        # self.summit_pos_tensor = self.actor_root_pos_tensor[self.actor_handles]
        # quit()

        self.summit_pos_tensor = self.summit_state_tensor[:, 0:3]
        self.summit_rot_tensor = self.summit_state_tensor[:, 3:7]
        self.summit_vel_tensor = self.summit_state_tensor[:, 7:10]

        # calculate potentials (not hard coded & depends on init pos)
        dist_to_target = self.goal_pos - self.summit_pos_tensor
        dist_to_target[:, 2] = 0.0
        self.potentials = - torch.norm(dist_to_target, p=2, dim=-1) / self.dt
        self.prev_potentials = self.potentials.clone()

        self.box_state_tensor = self.actor_root_state_tensor[self.box_handles]

    def create_sim(self):
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id,
                                      self.physics_engine, self.sim_params)

        self._create_ground_plane()

        print(
            f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')

        self._create_envs(
            self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)  # given up axis is Z?
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # Create lists to keep track of all the assets
        # note that walls is a dictionary
        prim_names = ['robot', 'boxes', 'walls']
        self.gym_assets = dict.fromkeys(prim_names)
        self.gym_indices = dict.fromkeys(prim_names)

        # summit_xl asset
        asset_root = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '../assets')
        summit_asset_file = "summit_xl_description/robots/summit_xls_std_fixed_camera.urdf"

        # Load Summit
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.use_mesh_materials = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL

        summit_asset = self.gym.load_asset(
            self.sim, asset_root, summit_asset_file, asset_options)
        self.gym_assets['robot'] = summit_asset

        # get some info about summit:
        self.summit_num_dofs = self.gym.get_asset_dof_count(
            summit_asset)  # 4 DOFs for summit
        self.summit_num_bodies = self.gym.get_asset_rigid_body_count(
            summit_asset)

        # Load box(es)
        box_assets = []
        box_width = 1
        box_density = 20.
        asset_options = gymapi.AssetOptions()
        asset_options.density = box_density
        asset_box = self.gym.create_box(
            self.sim, box_width, box_width, box_width, asset_options)
        box_assets.append(asset_box)
        self.gym_assets['boxes'] = box_assets

        # Load wall(s)
        wall_assets = {}  # a set with length as key
        room_walls = self.room_config['walls']
        wall_height = self.room_config['wall_height']
        wall_thickness = self.room_config['wall_thickness']
        wall_widths = set([wall['length']
                          for (name, wall) in room_walls.items()])
        wall_asset_options = gymapi.AssetOptions()
        wall_asset_options.fix_base_link = True
        # wall_asset_options.density = 1000.
        for width in wall_widths:
            asset_wall = self.gym.create_box(
                self.sim, wall_thickness, width, wall_height, wall_asset_options)
            wall_assets[width] = asset_wall
        self.gym_assets['walls'] = wall_assets

        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # set summit dof properties
        summit_dof_props = self.gym.get_asset_dof_properties(summit_asset)
        summit_dof_props['stiffness'].fill(0.0)
        summit_dof_props['damping'].fill(1000.0)
        for i in range(self.summit_num_dofs):
            summit_dof_props['driveMode'][i] = gymapi.DOF_MODE_VEL
            # apply upper and lower limit here as well

        # cache indices of different actors for each env
        self.envs = []
        self.actor_handles = []
        self.box_handles = []
        self.wall_handles = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)

            # add actor
            initial_pose = gymapi.Transform()
            initial_pose.p = gymapi.Vec3(-7.5, 2.5, 0.)
            initial_pose.r = gymapi.Quat(0., 0., -1., 1.)

            actor_handle = self.gym.create_actor(
                env, self.gym_assets['robot'], initial_pose, 'summit', i, 0)
            self.gym.set_actor_dof_properties(
                env, actor_handle, summit_dof_props)
            self.actor_handles.append(actor_handle)

            # add box
            box_handle = self.gym.create_actor(env, self.gym_assets['boxes'][0], gymapi.Transform(
                p=gymapi.Vec3(-7.5, -1, box_width/2)), 'box', i, 0)
            self.box_handles.append(box_handle)

            # for each box:
            box_shape_props = self.gym.get_actor_rigid_shape_properties(
                env, box_handle)
            # change properties
            box_shape_props[0].compliance = 0.
            box_shape_props[0].friction = 0.1
            box_shape_props[0].rolling_friction = 0.
            box_shape_props[0].torsion_friction = 0.
            self.gym.set_actor_rigid_shape_properties(
                env, box_handle, box_shape_props)

            box_body_props = self.gym.get_actor_rigid_body_properties(
                env, box_handle)
            box_body_props[0].mass *= 10
            self.gym.set_actor_rigid_body_properties(
                env, box_handle, box_body_props)

            # Add walls
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

                wall_handle = self.gym.create_actor(
                    env, self.gym_assets['walls'][wall_length], pos, wall_name, i, 0
                )
                self.wall_handles.append(wall_handle)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_summit_reward(
            self.obs_buf, self.progress_buf, self.reset_buf, self.max_episode_length, self.potentials, self.prev_potentials, self.reached_target)

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # update internal states
        dist_to_target = self.goal_pos - self.summit_pos_tensor
        dist_to_target[:, 2] = 0.0

        test_vel = self.actor_root_state_tensor[self.actor_handles][:, 7:10]
        # print(f'summit pos: {self.summit_pos_tensor[0]}')
        # print(f'summit wheel vel: {self.dof_vel[0]}')
        # print(f'summit vel: {self.summit_vel_tensor[0]}')
        # print(f'summit vel_root: {test_vel[0]}')
        # print(f'dist: {dist_to_target[0]}')

        self.prev_potentials = self.potentials.clone()
        self.potentials = -torch.norm(dist_to_target, p=2, dim=-1) / self.dt
        # print(f'potentials: {self.potentials}')
        self.reached_target = torch.norm(
            dist_to_target, p=2, dim=-1) <= self.goal_radius

        # goal pos, expanded along envs
        goal_pos = self.goal_pos[:, 0:2]
        # summit pos, vel, rot, reducing to 2d
        summit_pos = self.summit_pos_tensor[:, 0:2]
        summit_vel = self.summit_vel_tensor[:, 0:2]
        summit_rot_roll, summit_rot_pitch, summit_rot_yaw = get_euler_xyz(
            self.summit_rot_tensor)
        # wall bounds
        wall_bounds = self.map_coords
        # box keypoints
        box_keypoints = torch.flatten(
            gen_keypoints(self.box_state_tensor[:, 0:7]), start_dim=1)

        self.obs_buf = torch.cat(
            (goal_pos, summit_pos, summit_vel, summit_rot_roll.unsqueeze(-1), summit_rot_pitch.unsqueeze(-1), summit_rot_yaw.unsqueeze(-1), wall_bounds, box_keypoints), dim=-1)

        # compute_summit_observations()

    def reset_idx(self, env_ids):

        # reset summit
        position_noise = torch_rand_float(-0.2, 0.2,
                                          (len(env_ids), self.summit_num_dofs), device=self.device)
        velocity_noise = torch_rand_float(-0.1, 0.1,
                                          (len(env_ids), self.summit_num_dofs), device=self.device)

        self.dof_pos[env_ids] = self.initial_dof_pos[env_ids] + position_noise
        self.dof_vel[env_ids] = velocity_noise

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        summit_indices = self.global_indices[env_ids, 0].flatten()

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(
                                                         self.initial_root_states),
                                                     gymtorch.unwrap_tensor(summit_indices), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self.dof_state_tensor),
                                              gymtorch.unwrap_tensor(summit_indices), len(env_ids_int32))

        dist_to_target = self.goal_pos[env_ids] - \
            self.initial_root_states[env_ids, 0:3]
        dist_to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = - \
            torch.norm(dist_to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        # if needed we can do a rescaling here
        target_velocities = self.actions * 4
        velocity_tensor = gymtorch.unwrap_tensor(target_velocities)
        self.gym.set_dof_velocity_target_tensor(
            self.sim, velocity_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            tempSize = self.dof_state_tensor.size()
            tempSize2 = env_ids.size()
            print(f'dof_shape: {tempSize}')
            print(f'env shape: {tempSize2}')
            tempSize = self.actor_root_state_tensor.size()
            print(f'state shape: {tempSize}')
            print(f'resetting {env_ids}')
            print(f'goal pos is at : {self.goal_pos[0]}')
            # quit()
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)


@ torch.jit.script
def compute_summit_reward(obs_buf, progress_buf, reset_buf, max_episode_length, potentials, prev_potentials, reached_target):
    # type: (Tensor, Tensor, Tensor, float, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    # euclidian distance to goal
    progress_reward = potentials - prev_potentials
    # print(progress_reward.size())

    # reached target reward
    reached_goal_reward = reached_target * 10

    # action cost
    # optional: additional penalty for pushing boxes

    # penalty for every time step
    time_cost = -torch.ones_like(potentials) * 0.5

    # optional: dof at limit
    total_reward = progress_reward + time_cost + reached_goal_reward
    # print(f'net reward: {total_reward[0]}')

    reset = torch.where(reached_target == 1,
                        torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length -
                        1, torch.ones_like(reset_buf), reset)

    return total_reward, reset


@ torch.jit.script
# refactor code to here if need expensive computations
def compute_summit_observations():
    pass


@ torch.jit.script
def gen_keypoints(pose: torch.Tensor, num_keypoints: int = 8, size: Tuple[float, float, float] = (0.065, 0.065, 0.065)):

    num_envs = pose.shape[0]

    keypoints_buf = torch.ones(
        num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)

    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = [(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],
        corner = torch.tensor(corner_loc, dtype=torch.float32,
                              device=pose.device) * keypoints_buf[:, i, :]
        keypoints_buf[:, i, :] = local_to_world_space(corner, pose)
    return keypoints_buf
