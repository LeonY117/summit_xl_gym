# version 4
# this class is designed for the square room setting where the objective is to push the box into a specified location

# to run this script in test mode:
# python3 train.py task=Summit_04 test=True checkpoint=runs/SquareRoom/nn/lastSummit.pth num_envs=16

from typing import Dict, Any, Tuple
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *
from isaacgymenvs.utils.torch_jit_utils import *

import numpy as np
import random
import yaml
import os
import torch
import wandb

from .base.vec_task import VecTask  # pre-defined abstract class
from .helper_1 import *

wandb.init(project="box_pushing_task", config={
           "room": "square_r0"}, entity="leonyao", mode="disabled")


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
        # ---------------- LOAD GLOBAL CONFIG ----------------
        self.cfg = cfg  # OmegaConf & Hydra Config

        # load room configuration (in future we can merge this with global config)
        room_cfg_root = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "../cfg/rooms")
        room_cfg_file = self.cfg['env']['room_config']
        with open(f'{room_cfg_root}/{room_cfg_file}', 'r') as f:
            self.room_config = yaml.load(f, Loader=yaml.loader.SafeLoader)

        if self.cfg['env']['useBoxes'] == True:
            self.max_num_boxes = len(self.room_config['box_cfg']['boxes'])  # 2
        else:
            self.max_num_boxes = 0

        self.num_walls = len(self.room_config['wall_cfg']['walls'])  # 8
        #               summit     walls             boxes        goal
        self.num_actors = 1 + self.num_walls + self.max_num_boxes + 1

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        # self.cfg["env"]["numObservations"] = 9 + \
        #     4 * self.num_walls + 24 * self.max_num_boxes  # 65
        numObservations = 12 + 24 * self.max_num_boxes
        print('NUMBER OF HISTORY')
        print(self.cfg['env']['numHistory'])
        self.cfg["env"]["numObservations"] = numObservations * \
            self.cfg["env"]["numHistory"]  # 65 * 3
        self.cfg["env"]["numActions"] = 2
        # self.temp_obs_buf = self.obs_buf = torch.zeros(
        #     (self.num_envs, numObservations), device=self.device, dtype=torch.float)

        # Load variables from main config file:

        # to be configured later, randomize is set to false for now
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        self.power_scale = self.cfg["env"]["powerScale"]

        # cost and reward weights dictionary
        self.weights = {}

        # reward parameters
        self.progress_weight = self.cfg['env']['progressWeight']
        self.dist_weight = self.cfg['env']['distWeight']
        self.reached_goal_weight = self.cfg['env']['reachedGoalWeight']
        self.box_progress_weight = self.cfg['env']['boxProgressWeight']
        self.box_dist_weight = self.cfg['env']['boxDistWeight']
        self.box_reached_goal_weight = self.cfg['env']['boxReachedGoalWeight']

        # cost parameters
        self.time_weight = self.cfg['env']['timeWeight']
        self.dof_offset_weight = self.cfg['env']['dofOffsetWeight']

        self.weights = {'progressWeight': self.progress_weight,
                        'distWeight': self.dist_weight,
                        'reachedGoalWeight': self.reached_goal_weight,
                        'timeWeight': self.time_weight,
                        'dofOffsetWeight': self.dof_offset_weight,
                        'boxProgressWeight': self.box_progress_weight,
                        'boxDistWeight': self.box_dist_weight,
                        'boxReachedGoalWeight': self.box_reached_goal_weight,
                        }

        # initial pos and rot, and randomization parameters
        self.randomize_box_pos = self.cfg['env']['randomize_box']['randomize_pos']
        self.randomize_box_rot = self.cfg['env']['randomize_box']['randomize_rot']
        self.randomize_summit_pos = self.cfg['env']['randomize_summit']['randomize_pos']
        self.randomize_summit_rot = self.cfg['env']['randomize_summit']['randomize_rot']
        self.randomize_goal_pos = self.cfg['env']['randomize_goal']['randomize_pos']
        self.randomize_box_properties = self.cfg['env']['randomize_box_properties']
        self.randomize_summit_properties = self.cfg['env']['randomize_summit_properties']

        if self.randomize_box_pos:
            # since boxes are placed down before summit, we can't fix summit pos if boxes are random
            self.randomize_summit_pos = True
        if self.randomize_box_pos:
            self.box_x_bound = self.cfg['env']['randomize_box']['x_bound']
            self.box_y_bound = self.cfg['env']['randomize_box']['y_bound']
        if self.randomize_summit_pos:
            self.summit_x_bound = self.cfg['env']['randomize_summit']['x_bound']
            self.summit_y_bound = self.cfg['env']['randomize_summit']['y_bound']
            self.summit_whole_map_start_prob = self.cfg['env']['randomize_summit']['whole_map_start_prob']
        if self.randomize_goal_pos:
            self.goal_x_bound = self.cfg['env']['randomize_goal']['x_bound']
            self.goal_y_bound = self.cfg['env']['randomize_goal']['y_bound']
        if self.randomize_box_properties:
            self.random_box_mass = self.cfg['env']['randomize_properties']['box_mass']
            self.random_box_friction = self.cfg['env']['randomize_properties']['box_friction']
            self.random_box_width = self.cfg['env']['randomize_properties']['box_width']

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # plane parameters
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        super().__init__(config=self.cfg, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless)

        wandb.config.update(
            {'Control Frequency': self.cfg['env']['controlFrequencyInv'],
             'Episode Length': self.max_episode_length,
             'Num Boxes': self.max_num_boxes,
             'Randomize Boxes': self.randomize_box_pos,
             'Randomize Summit prob': self.summit_whole_map_start_prob},
        )

        self.actionOptions = torch.tensor([[1, 1, 1, 1], [-1, -1, -1, -1],
                                           [1, -1, 1, -1], [-1, 1, -1, 1]], dtype=torch.float, device=self.device).repeat((self.num_envs, 1)).view(self.num_envs, 4, 4)
        self.max_episode_lengths = torch.ones(
            self.num_envs, dtype=torch.long, device=self.device) * self.max_episode_length
        self.max_episode_lengths += torch.randint_like(
            self.max_episode_lengths, low=-int(np.floor(0.2*self.max_episode_length)), high=0)

        # set camera angle
        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 40.0)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

        # ---------------- INITIATE GLOBAL TENSORS ----------------
        # (num_env, num_actors * 13)
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(
            self.sim)
        # (num_env, num_dofs * 2)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # (num_env, num_bodies * 3)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # create some wrapper tensors, these are not clones but pointers!
        self.dof_state_tensor = gymtorch.wrap_tensor(_dof_state_tensor)
        # store the initial state, used for resetting the agent
        self.actor_root_state_tensor = gymtorch.wrap_tensor(
            _actor_root_state_tensor).view(self.num_envs, self.num_actors, 13)

        self.dof_pos = self.dof_state_tensor.view(
            self.num_envs, self.summit_num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state_tensor.view(
            self.num_envs, self.summit_num_dofs, 2)[..., 1]
        self.target_velocities = torch.zeros_like(self.dof_vel)
        self.vel_offset = torch.zeros_like(self.rew_buf)

        # This is the initial state of summit & boxes
        self.initial_root_states = self.actor_root_state_tensor.clone()
        self.initial_summit_pos = self.initial_root_states[:, 0, 0:3]
        self.initial_summit_rot = self.initial_root_states[:, 0, 3:7]
        if self.max_num_boxes > 0:
            self.initial_boxes_pos = self.initial_root_states[:,
                                                              1+self.num_walls:self.num_actors-1, 0:3]
            self.initial_boxes_rot = self.initial_root_states[:,
                                                              1+self.num_walls:self.num_actors-1, 3:7]

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
            self.num_envs * (self.num_actors), dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.dt = self.cfg["sim"]["dt"]
        self.t = 0
        # self.log_frequency = self.cfg['env']['controlFrequencyInv']
        self.log_frequency = 1

        # create tensors for boxes and summit
        self.summit_state_tensor = self.actor_root_state_tensor[:, 0, :13]
        self.summit_pos_tensor = self.summit_state_tensor[:, 0:3]
        self.summit_rot_tensor = self.summit_state_tensor[:, 3:7]
        self.summit_vel_tensor = self.summit_state_tensor[:, 7:10]
        self.summit_ang_vel_tensor = self.summit_state_tensor[:, 10:13]
        if self.max_num_boxes > 0:
            self.box_state_tensor = self.actor_root_state_tensor[:,
                                                                 1+self.num_walls:self.num_actors-1, :13]

            self.boxes_pos_tensor = self.box_state_tensor[:, :, 0:3]
            self.boxes_rot_tensor = self.box_state_tensor[:, :, 3:7]
        else:
            self.box_state_tensor = self.summit_state_tensor.clone()
        self.goal_state_tensor = self.actor_root_state_tensor[:,
                                                              self.num_actors-1, :13]
        self.goal_pos_tensor = self.goal_state_tensor[:, 0:3]

        # calculate potentials (not hard coded & depends on init pos)
        self.dist_to_target = self.goal_pos_tensor - self.summit_pos_tensor
        self.dist_to_target[:, 2] = 0.0
        self.dist_to_target = torch.norm(self.dist_to_target, p=2, dim=-1)

        self.box_dist_to_target = self.goal_pos_tensor - \
            self.boxes_pos_tensor[:, 0, :]
        self.box_dist_to_target[:, 2] = 0.0
        self.box_dist_to_target = torch.norm(
            self.box_dist_to_target, p=2, dim=-1)

        self.potentials = - self.dist_to_target / self.dt
        self.prev_potentials = self.potentials.clone()

        self.box_potentials = -self.box_dist_to_target / self.dt
        self.prev_box_potentials = self.box_potentials.clone()

        self.reached_target = torch.zeros_like(self.potentials)
        self.box_reached_target = torch.zeros_like(self.potentials)

        # Create placeholder variables for wandb
        self.reward_log = 0
        self.dist_rew_log = 0
        self.progress_rew_log = 0
        self.potentials_log = 0
        self.num_reached_goal = 0
        self.episode_length_log = 0

        self.box_progress_rew_log = 0
        self.box_dist_rew_log = 0
        self.num_box_reached_goal = 0

        # we use a list here for efficient queue implementation
        self.running_agent_status = [0] * self.num_envs
        self.running_box_status = [0] * self.num_envs

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
        # ---------------- LOAD ASSETS ----------------
        # load data from self.room_config using helper function
        map_coords, goal_pos = load_room_from_config(
            self.room_config)
        self.box_width = self.room_config['box_cfg']['box_width']
        self.object_radius = (2 * self.box_width**2)**0.5
        self.summit_radius = 2**0.5
        # self.map_bounds = get_wall_bounds(2**0.5, map_coords)
        self.summit_map_bounds = get_wall_bounds(
            self.summit_radius, map_coords)
        self.map_coords = map_coords
        self.wall_coords = to_torch(
            map_coords, device=self.device, dtype=torch.float).flatten().repeat((self.num_envs, 1))
        self.goal_pos = to_torch(
            [goal_pos], device=self.device, dtype=torch.float).repeat((self.num_envs, 1))
        self.goal_radius = to_torch(
            self.room_config['goal_cfg']['goal_radius'], device=self.device, dtype=torch.float)
        self.summit_default_initial_pos = [
            self.room_config['summit_cfg']['pos_x'], self.room_config['summit_cfg']['pos_y']]
        self.room_width, self.room_height = self.room_config[
            'room_width'], self.room_config['room_height']
        self.max_dist = (self.room_width ** 2 + self.room_height ** 2) ** 0.5
        # global for now, dynamic would improve start state distribution

        # Create lists to keep track of all the assets
        # note that walls is a dictionary indexed by wall lengths
        prim_names = ['robot', 'boxes', 'walls', 'goal']
        self.gym_assets = dict.fromkeys(prim_names)
        self.gym_indices = dict.fromkeys(prim_names)

        # summit_xl asset
        asset_root = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), '../assets')
        summit_asset_file = "summit_xl_description/robots/summit_xls_std_fixed_camera.urdf"

        # Load summit asset
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

        # Load box asset
        if self.max_num_boxes > 0:
            box_assets = []
            box_density = 20.
            asset_options = gymapi.AssetOptions()
            asset_options.density = box_density
            asset_box = self.gym.create_box(
                self.sim, self.box_width, self.box_width, self.box_width, asset_options)
            box_assets.append(asset_box)
            self.gym_assets['boxes'] = box_assets

        # Load wall asset
        wall_assets = {}  # a set with length as key
        room_walls = self.room_config['wall_cfg']['walls']
        wall_height = self.room_config['wall_cfg']['wall_height']
        wall_thickness = self.room_config['wall_cfg']['wall_thickness']
        wall_widths = set([wall['length'] for wall in room_walls])
        wall_asset_options = gymapi.AssetOptions()
        wall_asset_options.fix_base_link = True
        # wall_asset_options.density = 1000.
        for width in wall_widths:
            asset_wall = self.gym.create_box(
                self.sim, wall_thickness, width, wall_height, wall_asset_options)
            wall_assets[width] = asset_wall
        self.gym_assets['walls'] = wall_assets

        # Load goal asset
        goal_height = 0.01
        goal_radius = self.room_config['goal_cfg']['goal_radius']
        goal_asset_options = gymapi.AssetOptions()
        goal_asset_options.fix_base_link = True
        asset_goal = self.gym.create_box(
            self.sim, goal_radius, goal_radius, goal_height, goal_asset_options)
        self.gym_assets['goal'] = asset_goal

        # Set summit dof properties
        summit_dof_props = self.gym.get_asset_dof_properties(summit_asset)
        summit_dof_props['stiffness'].fill(800.0)
        summit_dof_props['damping'].fill(300.0)
        for i in range(self.summit_num_dofs):
            summit_dof_props['driveMode'][i] = gymapi.DOF_MODE_VEL
            # TODO: apply upper and lower limit here

        # ---------------- CREATE ENVIRONMENTS ----------------

        lower = gymapi.Vec3(-0, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # cache indices of different actors for each env
        self.envs = []
        self.actor_handles = []
        self.box_handles = []
        self.wall_handles = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)

            # add summit actor
            self.summit_initial_quat = gymapi.Quat(
                0., 0., -1., 1.)  # rotates summit
            initial_pose = gymapi.Transform()
            initial_pose.p = gymapi.Vec3(
                self.summit_default_initial_pos[0], self.summit_default_initial_pos[1], 0.)
            initial_pose.r = self.summit_initial_quat

            actor_handle = self.gym.create_actor(
                env, self.gym_assets['robot'], initial_pose, 'summit', i, 0)
            self.gym.set_actor_dof_properties(
                env, actor_handle, summit_dof_props)

            actor_shape_props = self.gym.get_actor_rigid_shape_properties(
                env, actor_handle)
            for actor_shape_prop in actor_shape_props:
                actor_shape_prop.friction = 0.01
                actor_shape_prop.rolling_friction = 0.001
                actor_shape_prop.torsion_friction = 0.001
                # actor_shape_prop.compliance = 0
            self.gym.set_actor_rigid_shape_properties(
                env, actor_handle, actor_shape_props)

            self.actor_handles.append(actor_handle)

            # add wall actor
            for wall in room_walls:
                wall_name = wall['name']
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

            # add box actor
            for box_idx in range(self.max_num_boxes):
                box = self.room_config['box_cfg']['boxes'][box_idx]
                pos_x, pos_y = box['pos_x'], box['pos_y']
                box_handle = self.gym.create_actor(env, self.gym_assets['boxes'][0], gymapi.Transform(
                    p=gymapi.Vec3(pos_x, pos_y, self.box_width/2)), 'box', i, 0)
                self.box_handles.append(box_handle)

                box_shape_props = self.gym.get_actor_rigid_shape_properties(
                    env, box_handle)

                # TODO: randomize friction properties here
                box_shape_props[0].compliance = 0.
                box_shape_props[0].friction = 0.1
                box_shape_props[0].rolling_friction = 0.
                box_shape_props[0].torsion_friction = 0.
                self.gym.set_actor_rigid_shape_properties(
                    env, box_handle, box_shape_props)

                # TODO: randomize mass properties here
                box_body_props = self.gym.get_actor_rigid_body_properties(
                    env, box_handle)
                box_body_props[0].mass *= 1
                self.gym.set_actor_rigid_body_properties(
                    env, box_handle, box_body_props)

            # add goal actor
            goal_handle = self.gym.create_actor(
                env, self.gym_assets['goal'], gymapi.Transform(p=gymapi.Vec3(goal_pos[0], goal_pos[1], goal_height/2)), 'goal', i+1, 0)
            self.gym.set_rigid_body_color(
                env, goal_handle, 0, gymapi.MeshType.MESH_VISUAL, gymapi.Vec3(0, 255, 0))

    def compute_reward(self, actions):

        self.rew_buf[:], self.reset_buf[:], logs = compute_summit_reward(
            self.obs_buf, self.progress_buf, self.reset_buf, self.max_episode_lengths,
            self.potentials, self.prev_potentials, self.dist_to_target, self.reached_target,
            self.box_potentials, self.prev_box_potentials, self.box_dist_to_target,
            self.box_reached_target, self.max_dist, self.vel_offset, self.power_scale, self.weights)

        mean_rew = logs['meanRew']
        mean_progress_rew = logs['meanProgressRew']
        mean_dist_rew = logs['meanDistRew']
        num_reached_goal = logs['numReachedGoal']
        num_fail_reach_goal = logs['numFailReachedGoal']
        mean_box_progress_rew = logs['meanBoxProgressRew']
        mean_box_dist_rew = logs['meanBoxDistRew']
        num_box_reached_goal = logs['numBoxReachedGoal']
        num_box_fail_reached_goal = logs['numBoxFailReachedGoal']

        # update
        for _ in range(int(num_reached_goal)):
            self.running_agent_status.pop(0)
            self.running_agent_status.append(1)
        for _ in range(int(num_fail_reach_goal)):
            self.running_agent_status.pop(0)
            self.running_agent_status.append(0)

        for _ in range(int(num_box_reached_goal)):
            self.running_box_status.pop(0)
            self.running_box_status.append(1)
        for _ in range(int(num_box_fail_reached_goal)):
            self.running_box_status.pop(0)
            self.running_box_status.append(0)

        # log data for wandb
        self.reward_log = mean_rew
        self.progress_rew_log = mean_progress_rew
        self.dist_rew_log = mean_dist_rew
        self.num_reached_goal = num_reached_goal
        self.num_box_reached_goal = num_box_reached_goal
        self.box_dist_rew_log = mean_box_dist_rew
        self.box_progress_rew_log = mean_box_progress_rew

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)

        self.obs_buf[:], self.reached_target[:], self.potentials[:], self.prev_potentials[:], self.dist_to_target, self.box_reached_target[:], self.box_potentials[:], self.prev_box_potentials[:], self.box_dist_to_target[:] = compute_summit_observations(
            self.obs_buf[:], self.potentials, self.prev_potentials, self.num_envs, self.max_num_boxes, self.dt,
            self.goal_pos_tensor, self.goal_radius, self.wall_coords, self.summit_pos_tensor,
            self.summit_vel_tensor, self.summit_rot_tensor, self.summit_ang_vel_tensor,
            self.target_velocities, self.dof_vel, self.box_state_tensor, self.box_potentials, self.prev_box_potentials)

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        summit_indices = self.global_indices[env_ids, 0].flatten()
        box_indices = self.global_indices[env_ids,
                                          1 + self.num_walls:self.num_actors-1].flatten()

        goal_indices = self.global_indices[env_ids,
                                           self.num_actors-1].flatten()
        # reset summit and box positions
        for env_idx in env_ids_int32:
            box_idx = 0
            box_poses = [[-100, -100]] * self.max_num_boxes
            box_radii = [0] * self.max_num_boxes
            while box_idx < self.max_num_boxes:
                if self.randomize_box_rot:
                    theta = np.pi/2 * \
                        random.uniform(-1, 1)  # +- 180 deg
                    q = gymapi.Quat.from_axis_angle(gymapi.Vec3(
                        0., 0., 1.), theta)
                    self.boxes_rot_tensor[env_idx][box_idx][:] = to_torch(
                        [q.x, q.y, q.z, q.w], dtype=torch.float, device=self.device)
                elif not self.randomize_box_rot:
                    self.boxes_rot_tensor[env_idx][box_idx][:
                                                            ] = self.initial_boxes_rot[0, box_idx, :]
                    theta = 0
                # caculate minimum bounding box from box width and rotation:
                box_max_width = max(
                    abs(self.box_width * np.cos(theta)), abs(self.box_width * np.sin(theta)))
                # assume bounding box to be square to avoid complications
                box_radius = (2 * box_max_width ** 2) ** 0.5
                map_bound = get_wall_bounds(box_radius, self.map_coords)
                if self.randomize_box_pos:
                    pos = [random.uniform(self.box_x_bound[0], self.box_x_bound[1]) * (self.room_width/2 - box_radius),
                           random.uniform(self.box_y_bound[0], self.box_y_bound[1]) * (self.room_height/2 - box_radius)]
                    if not is_valid_pos(pos, map_bound, box_poses, box_radius * 2):
                        continue
                    box_poses[box_idx] = pos
                    self.boxes_pos_tensor[env_idx][box_idx][:] = to_torch(
                        [pos[0], pos[1], self.box_width/2], dtype=torch.float, device=self.device)
                elif not self.randomize_box_pos:
                    pos = self.initial_boxes_pos[0, box_idx, 0:2]
                    box_poses[box_idx] = pos
                    self.boxes_pos_tensor[env_idx][box_idx][:] = to_torch(
                        [pos[0], pos[1], self.box_width/2], dtype=torch.float, device=self.device)
                box_radii[box_idx] = box_radius
                box_idx += 1
            # RESET SUMMIT
            invalid_summit_pos = True
            while invalid_summit_pos:
                if self.randomize_summit_pos:
                    if random.uniform(0, 1) < self.summit_whole_map_start_prob:
                        pos = [random.uniform(-1, 1) * (self.room_width/2 - self.summit_radius),
                               random.uniform(-1, 1) * (self.room_height/2 - self.summit_radius)]
                    else:
                        pos = [random.uniform(self.summit_x_bound[0], self.summit_x_bound[1]) * (self.room_width/2 - self.summit_radius),
                               random.uniform(self.summit_y_bound[0], self.summit_y_bound[1]) * (self.room_height/2 - self.summit_radius)]
                    invalid_summit_pos = not is_valid_pos(pos, self.summit_map_bounds,
                                                          box_poses, self.summit_radius + max(box_radii))
                    if invalid_summit_pos:
                        continue
                    # found valid position * 2
                    self.summit_pos_tensor[env_idx][:] = to_torch(
                        [pos[0], pos[1], 0], dtype=torch.float, device=self.device)
                elif not self.randomize_summit_pos:
                    self.summit_pos_tensor[env_idx][:] = self.initial_summit_pos[0, :]
                    invalid_summit_pos = False
                # set random rotation
                random_theta = np.pi/2 * random.uniform(-1, 1)  # +- 180 deg
                q = gymapi.Quat.from_axis_angle(gymapi.Vec3(
                    0., 0., 1.), random_theta) * self.summit_initial_quat
                self.summit_rot_tensor[env_idx][0:4] = to_torch(
                    [q.x, q.y, q.z, q.w], dtype=torch.float, device=self.device)

        # reset summit dof
        summit_dof_velocity_noise = torch_rand_float(-0, 0,
                                                     (len(env_ids), self.summit_num_dofs), device=self.device)

        # reset goal position
        if self.randomize_goal_pos:
            goal_pos = torch_rand_float(-self.room_width/2, self.room_width/2,
                                        (len(env_ids), 3), device=self.device)
            goal_pos[:, 2] = 0.005
            self.goal_pos_tensor[env_ids] = goal_pos

        self.dof_pos[env_ids] = self.initial_dof_pos[env_ids]
        self.dof_vel[env_ids] = summit_dof_velocity_noise

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self.dof_state_tensor),
                                              gymtorch.unwrap_tensor(summit_indices), len(env_ids_int32))
        update_indices = torch.cat(
            (summit_indices, goal_indices, box_indices), dim=-1)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(
                                                         self.actor_root_state_tensor),
                                                     gymtorch.unwrap_tensor(update_indices), len(update_indices))

        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(
        #                                                  self.actor_root_state_tensor),
        #                                              gymtorch.unwrap_tensor(goal_indices), len(env_ids_int32))

        # if self.max_num_boxes > 0:
        #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                                  gymtorch.unwrap_tensor(
        #                                                      self.actor_root_state_tensor),
        #  gymtorch.unwrap_tensor(box_indices), len(env_ids_int32))

        # reset other internal variables
        dist_to_target = self.goal_pos_tensor - self.summit_pos_tensor
        dist_to_target[:, 2] = 0.0
        dist_to_target = torch.norm(dist_to_target, p=2, dim=-1)

        box_dist_to_target = self.goal_pos_tensor - \
            self.boxes_pos_tensor[:, 0, :]
        box_dist_to_target[:, 2] = 0.0
        box_dist_to_target = torch.norm(box_dist_to_target, p=2, dim=-1)

        self.potentials[env_ids] = - dist_to_target[env_ids] / self.dt
        self.prev_potentials[env_ids] = self.potentials[env_ids].clone()

        self.box_potentials[env_ids] = - box_dist_to_target[env_ids] / self.dt
        self.prev_box_potentials[env_ids] = self.box_potentials[env_ids].clone(
        )

        self.obs_buf[env_ids] = torch.zeros_like(self.obs_buf[env_ids])
        self.max_episode_lengths[env_ids] = torch.ones_like(
            self.max_episode_lengths[env_ids]) * self.max_episode_length \
            + torch.randint_like(self.max_episode_lengths[env_ids], low=-int(
                np.floor(0.2*self.max_episode_length)), high=0)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.reached_target[env_ids] = 0
        self.box_reached_target[env_ids] = 0

    def pre_physics_step(self, actions):
        # TAKE ACTION
        self.actions = actions.clone().to(self.device)
        # if needed we can do a rescaling here
        self.target_velocities[:, 0] = self.actions[:, 0]
        self.target_velocities[:, 1] = self.actions[:, 1]
        self.target_velocities[:, 2] = self.actions[:, 0]
        self.target_velocities[:, 3] = self.actions[:, 1]

        # back_left, back_right, front_left, front_right
        self.target_velocities *= self.power_scale
        # print(torch.mean(target_velocities, dim=0))'

        # # DISCRETE ACTION SPACE: FORWARD, BACKWARD, ROT CW, ROT CCW
        # # softmax layer
        # prob = torch.softmax(self.actions, 0)
        # indices = torch.max(
        #     prob, -1).indices.unsqueeze(-1).repeat(1, 4).unsqueeze(1)
        # # indices = torch.multinomial(prob, 1, False).repeat(1, 4).unsqueeze(1)
        # # print(self.actionOptions.shape)
        # target_velocities = torch.gather(
        #     self.actionOptions, 1, indices).squeeze()
        # # print(target_velocities.shape)
        # target_velocities *= self.power_scale

        velocity_tensor = gymtorch.unwrap_tensor(self.target_velocities)
        self.gym.set_dof_velocity_target_tensor(
            self.sim, velocity_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        # if (self.progress_buf[0] % 100 == 0):
        #     print(f'reward:{self.rew_buf[0]}')

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        self.t += 1
        # TODO: refactor this into compute_observations
        offset = torch.mean(torch.norm(
            self.target_velocities - self.dof_vel, p=2, dim=-1))
        if self.t % self.log_frequency == 0:
            wandb.log({
                'total reward': self.reward_log,
                'dist rew': self.dist_rew_log,
                'progress reward': self.progress_rew_log,
                'dist rew (box)': self.box_dist_rew_log,
                'progress reward (box)': self.box_progress_rew_log,
                'num reached goal': self.num_reached_goal,
                'num boxes reached goal': self.num_box_reached_goal,
                'episode length': torch.sum(self.progress_buf)/self.num_envs,
                'percentage completed': sum(self.running_agent_status)/len(self.running_agent_status),
                'percentage completed (box)': sum(self.running_box_status) / len(self.running_agent_status),
                'time steps': self.t,
                'vel_offset': offset,
            })


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


@ torch.jit.script
def compute_summit_reward(obs_buf, progress_buf, reset_buf, max_episode_lengths, potentials, prev_potentials,
                          dist_to_target, reached_target, box_potentials, prev_box_potentials, box_dist_to_target,
                          box_reached_target, max_dist, vel_offset, power_scale, weights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, float, Dict[str, float]) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]
    # euclidian distance to goal
    progress_reward = (potentials - prev_potentials) * \
        weights['progressWeight']
    dist_reward = ((max_dist - dist_to_target) /
                   max_dist) * weights['distWeight']

    box_progress_reward = (
        box_potentials - prev_box_potentials) * weights['boxProgressWeight']
    box_dist_reward = ((max_dist - box_dist_to_target) /
                       max_dist) * weights['boxDistWeight']

    # reached target reward
    reached_goal_reward = reached_target * weights['reachedGoalWeight']
    box_reached_goal_reward = box_reached_target * \
        weights['boxReachedGoalWeight']

    # action cost
    # optional: additional penalty for pushing boxes
    # optional: additional penalty for pushing walls

    # penalty for every time step
    time_cost = -torch.ones_like(potentials) * weights['timeWeight']

    # dof offset cost
    # dof_offset_cost = - vel_offset/power_scale *  weights['timeWeight']

    # optional: dof at limit
    total_reward = progress_reward + time_cost + \
        box_progress_reward + box_dist_reward + \
        reached_goal_reward + box_reached_goal_reward
    # print(f'net reward: {total_reward[0]}')

    # Reset agents who have reached the target
    # reset = torch.where(reached_target == 1,
    #                     torch.ones_like(reset_buf), reset_buf)

    num_reached_goal = torch.sum(reset_buf)

    reset = torch.where(box_reached_target == 1,
                        torch.ones_like(reset_buf), reset_buf)

    num_box_reached_goal = torch.sum(reset)

    # Reset agents who have exceeded maximum episode length
    reset = torch.where(progress_buf >= max_episode_lengths -
                        1, torch.ones_like(reset_buf), reset)

    num_box_fail_reached_goal = torch.sum(reset) - num_box_reached_goal

    num_fail_reach_goal = torch.sum(reset) - num_reached_goal

    # Get the mean value of rewards for w&b
    mean_rew = torch.mean(total_reward)
    mean_progress_rew = torch.mean(progress_reward)
    mean_dist_rew = torch.mean(dist_reward)
    mean_box_progress_rew = torch.mean(box_progress_reward)
    mean_box_dist_rew = torch.mean(box_dist_reward)

    logs = {
        'meanRew': mean_rew,
        'meanProgressRew': mean_progress_rew,
        'meanDistRew': mean_dist_rew,
        'meanBoxProgressRew': mean_box_progress_rew,
        'meanBoxDistRew': mean_box_dist_rew,
        'numReachedGoal': num_reached_goal,
        'numBoxReachedGoal': num_box_reached_goal,
        'numFailReachedGoal': num_fail_reach_goal,
        'numBoxFailReachedGoal': num_box_fail_reached_goal}

    return total_reward, reset, logs


@ torch.jit.script
# refactor code to here if need expensive computations
def compute_summit_observations(obs_buf, potentials, prev_potentials, num_envs, max_num_boxes, dt,
                                goal_pos, goal_radius, map_coords, summit_pos_tensor,
                                summit_vel_tensor, summit_rot_tensor, summit_ang_vel_tensor,
                                target_velocities, dof_vel, box_state_tensor, box_potentials, prev_box_potentials):
    # type: (Tensor, Tensor, Tensor, int, int, float, Tensor, float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    # update internal states
    dist_to_target = goal_pos - summit_pos_tensor
    dist_to_target[:, 2] = 0.0
    dist_to_target = torch.norm(dist_to_target, p=2, dim=-1)

    box_pos = box_state_tensor[:, 0, 0:3]
    box_dist_to_target = goal_pos - box_pos
    box_dist_to_target[:, 2] = 0.0
    box_dist_to_target = torch.norm(box_dist_to_target, p=2, dim=-1)

    prev_potentials = potentials.clone()
    potentials = -dist_to_target / dt

    prev_box_potentials = box_potentials.clone()
    box_potentials = -box_dist_to_target / dt

    reached_target = torch.where(dist_to_target <= goal_radius, torch.ones_like(
        dist_to_target), torch.zeros_like(dist_to_target))

    box_reached_target = torch.where(box_dist_to_target <= goal_radius, torch.ones_like(
        box_dist_to_target), torch.zeros_like(box_dist_to_target))

    vel_offset = torch.norm(target_velocities - dof_vel, p=2, dim=-1)

    goal_pos = goal_pos[:, 0:2]
    # summit pos, vel, rot, reducing to 2d
    summit_pos = summit_pos_tensor[:, 0:2]
    summit_vel = summit_vel_tensor[:, 0:2]
    summit_rot_roll, summit_rot_pitch, summit_rot_yaw = get_euler_xyz(
        summit_rot_tensor)
    summit_ang_vel = summit_ang_vel_tensor[:, 0:3]
    # wall bounds
    wall_bounds = map_coords

    # obs_buf for curr time step
    curr_obs_buf = torch.cat((goal_pos, summit_pos, summit_vel, summit_rot_roll.unsqueeze(
        -1),  summit_rot_pitch.unsqueeze(-1), summit_rot_yaw.unsqueeze(-1), summit_ang_vel), dim=-1)

    if max_num_boxes > 0:
        # box keypoints
        boxes_keypoints_buf = torch.ones(
            num_envs, max_num_boxes, 24, dtype=torch.float32, device=potentials.device)
        for i in range(max_num_boxes):
            box_keypoints = (torch.flatten(
                gen_keypoints(box_state_tensor[:, i, 0:7]), start_dim=1))
            boxes_keypoints_buf[:, i, :] = box_keypoints
        boxes_keypoints = torch.flatten(boxes_keypoints_buf, start_dim=1)

        curr_obs_buf = torch.cat((curr_obs_buf, boxes_keypoints), dim=-1)

    # add obs_buf to history:
    num_obs = curr_obs_buf.shape[1]
    num_history = obs_buf.shape[1]//num_obs
    for i in range(num_history-1):
        obs_buf[:, i*num_obs:(i+1)*num_obs] = obs_buf[:, (i+1)
                                                      * num_obs:(i+2)*num_obs]
    i = max(0, num_history - 2)
    obs_buf[:, (i+1)*num_obs:(i+2)*num_obs] = curr_obs_buf[:]

    # obs_buf[:, (i+2)*num_obs:] = wall_bounds

    return obs_buf, reached_target, potentials, prev_potentials, dist_to_target, box_reached_target, box_potentials, prev_box_potentials, box_dist_to_target
