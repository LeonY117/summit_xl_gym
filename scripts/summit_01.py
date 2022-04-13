# version 1 of summit xl steel class
# the code is adapted from ant.py from isaacgymenvs

from click import pass_context
import numpy as np
import yaml
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask  # pre-defined abstract class

# TODO: find better way to store map config
room_walls = [{
    'name': 'left_bound',
    'pos': (-10, 0),
    'length': 10,
    'orientation': 'vertical',
    'thickness': 0.1,
}, {
    'name': 'right_bound',
    'pos': (10, 0),
    'length': 10,
    'orientation': 'vertical',
    'thickness': 0.1,
}, {
    'name': 'top_bound',
    'pos': (0, 5),
    'length': 20,
    'orientation': 'horizontal',
    'thickness': 0.1,
}, {
    'name': 'bottom_bound',
    'pos': (0, -5),
    'length': 20,
    'orientation': 'horizontal',
    'thickness': 0.1,
}, {
    'name': 'wall_1',
    'pos': (-5, 2.5),
    'length': 5,
    'orientation': 'vertical',
    'thickness': 0.1
}, {
    'name': 'wall_2',
    'pos': (0, -1.5),
    'length': 7,
    'orientation': 'vertical',
    'thickness': 0.1
}, {
    'name': 'wall_3',
    'pos': (1.5, 2),
    'length': 3,
    'orientation': 'horizontal',
    'thickness': 0.1
}, {
    'name': 'wall_4',
    'pos': (8.5, 2),
    'length': 3,
    'orientation': 'horizontal',
    'thickness': 0.1
}
]


class summit(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        """
        Initialises the class.

        Args:
           config: config dictionary for the environment.
           sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
           graphics_device_id: the device ID to render with.
           headless: Set to False to disable viewer rendering.
        """
        # --SET CONFIG--
        self.cfg = cfg  # OmegaConf & Hydra Config

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        # TODO: figure out what these two represent exactly
        self.cfg["env"]["numObservations"] = 60
        self.cfg["env"]["numActions"] = 6

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
        _actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # retrieves buffer for dof state, with shape (num_env, num_dofs * 2)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        # populates tensors with latest data from the simulator
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # store the initial state, used for resetting the agent
        self.root_states = gymtorch.wrap_tensor(_actor_root_state)
        # TODO: this needs to change: we have initial pos for both agent and boxes
        self.initial_root_states = self.root_states.clone()

        # create some wrapper tensors for different slices
        # dof state tensors store pos and vel for each dof
        self.dof_state = gymtorch.wrap_tensor(_dof_state_tensor)
        self.dof_pos = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(
            self.num_envs, self.num_dof, 2)[..., 1]

        self.initial_dof_pos = torch.zeros_like(
            self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        # snap pos constraints to initial pos (if initial dof pos with all zeros violate constraint)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(
            self.dof_vel, device=self.device, dtype=torch.float)

        # initialize some data used later on
        self.up_vec = to_torch(get_axis_params(
            1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        
        self.dt = self.cfg["sim"]["dt"]

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
        self.num_dof = self.gym.get_asset_dof_count(
            summit_asset)  # 4 DOFs for summit
        self.num_bodies = self.gym.get_asset_rigid_body_count(summit_asset)
        body_names = [self.gym.get_asset_rigid_body_name(
            summit, i) for i in range(self.num_bodies)]

        # Load box(es)
        box_assets = []
        box_width = 0.8
        box_density = 20.
        asset_options = gymapi.AssetOptions()
        asset_options.density = box_density
        asset_box = self.gym.create_box(
            self.sim, box_width, box_width, box_width, asset_options)
        box_assets.append(asset_box)
        self.gym_assets['boxes'] = box_assets

        # Load wall(s)
        wall_assets = {}
        wall_height = 2.
        wall_thickness = 0.1
        wall_widths = set(wall['length'] for wall in room_walls)
        wall_asset_options = gymapi.AssetOptions()
        wall_asset_options.fix_base_link = True
        # wall_asset_options.density = 1000.
        for width in wall_widths:
            asset_wall = self.gym.create_box(
                self.sim, wall_thickness, wall_height, width, wall_asset_options)
            wall_assets[width] = asset_wall
        self.gym_assets['walls'] = wall_assets

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.envs = []
        self.actor_handles = []
        self.box_handles = []
        self.wall_handles = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env)

            # add actor
            initial_pose = gymapi.Transform()
            initial_pose.p = gymapi.Vec3(-7.5, 0, 2.5)
            initial_pose.r = gymapi.Quat(-1.0, 0.0, 0.0, 1.0)

            actor_handle = self.gym.create_actor(
                env, self.gym_assets['robot'], initial_pose, 'summit', i, 0)
            self.actor_handles.append(actor_handle)

            # add box
            box_handle = self.gym.create_actor(env, self.gym_assets['boxes'][0], gymapi.Transform(
                p=gymapi.Vec3(-5., box_width/2, -2.5)), 'box', i, 0)
            self.box_handles.append(box_handle)

            box_props = self.gym.get_actor_rigid_shape_properties(
                env, box_handle)
            # change properties
            box_props[0].compliance = 0.
            box_props[0].friction = 0.1
            box_props[0].rolling_friction = 0.
            box_props[0].torsion_friction = 0.
            self.gym.set_actor_rigid_shape_properties(
                env, box_handle, box_props)

            # Add walls
            for (i, wall) in enumerate(room_walls):
                wall_name = wall['name']
                wall_thickness = wall['thickness']
                wall_length = wall['length']
                wall_orientation = wall['orientation']
                wall_pos = wall['pos']

                pos = gymapi.Transform()
                pos.p = gymapi.Vec3(wall_pos[0], 1.0, wall_pos[1])
                if wall_orientation == 'horizontal':
                    pos.r = gymapi.Quat(-1.0, 0.0, 1.0, 0.0)

                wall_handle = self.gym.create_actor(
                    env, self.gym_assets['walls'][wall_length], pos, wall_name, i, 0
                )
                self.wall_handles.append(wall_handle)

    def compute_reward(self, actions):
        pass

    def compute_observations(self):
        pass
