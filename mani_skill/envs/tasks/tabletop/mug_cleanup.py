from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
import trimesh

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig

from mani_skill.utils.geometry.geometry import transform_points


@register_env("MugCleanup-v1", max_episode_steps=500)
class MugCleanupEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to open the drawer, pick up the mug and put it inside the drawer, and close the drawer.

    **Randomizations:**
    - mug initial position and orientation
    - drawer initial position and orientation

    **Success Conditions:**
    """

    # _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    min_open_frac = 0.75
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 0, 0.6], target=[-0.1, 0, -0.1])
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 3, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 0, 0.6], target=[-0.1, 0, -0.1])
        return [CameraConfig("render_camera", pose, 256, 256, np.pi / 3, 0.01, 100)]

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()


        model_id = '025_mug'
        self._objs: List[Actor] = []
        self.obj_heights = []

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{model_id}",
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        builder.set_scene_idxs([0])
        self._objs = []
        self._objs.append(builder.build(name=f"{model_id}-0"))
        self.remove_from_state_dict_registry(self._objs[-1])
        self.obj = Actor.merge(self._objs, name="ycb_object")
        self.add_to_state_dict_registry(self.obj)

        self._drawers: List[Articulation] = []
        handle_links: List[List[Link]] = []
        handle_links_meshes: List[List[trimesh.Trimesh]] = []

        mjcf_path = 'assets/drawer_long.xml'
        loader = self.scene.create_mjcf_loader()
        builders = loader.parse(str(mjcf_path))
        articulation_builders = builders["articulation_builders"]
        actor_builders = builders["actor_builders"]
        builder = articulation_builders[0]
        builder.initial_pose = sapien.Pose(p=[0.2, 0.2, 0.0])
        builder.fix_root_link = True  # fix the base of the drawer
        drawer = builder.build(name="drawer_articulation")

        # import ipdb; ipdb.set_trace()

        for link in drawer.links:
            # Clear collision filtering bits (group 2) so it can collide with robot
            for bit in range(32):
                link.set_collision_group_bit(2, bit, 0)

        self._drawers.append(drawer)
        handle_links.append([])
        handle_links_meshes.append([])
        

        for link, joint in zip(drawer.links, drawer.joints):
            if joint.type[0] in ['prismatic']:
                handle_links[-1].append(link)
                # save the first mesh in the link object that correspond with a handle
                handle_links_meshes[-1].append(
                    link.generate_mesh(
                        filter=lambda _, render_shape: "handle"
                        in render_shape.name,
                        mesh_name="handle",
                    )[0]
                )
        
        self.drawer = Articulation.merge(self._drawers, name="drawer")

        # print("drawer wrapper type:", type(drawer))
        # print("drawer wrapper dir has:", [k for k in ["entity","_entity","raw","_raw","articulation","_articulation"] if hasattr(drawer, k)])
        # print("merged drawer type:", type(self.drawer))
        # print("merged drawer dir has:", [k for k in ["entity","_entity","raw","_raw","articulation","_articulation"] if hasattr(self.drawer, k)])
        
        self.add_to_state_dict_registry(self.drawer)
        self.handle_link = Link.merge(
            [links[0] for i, links in enumerate(handle_links)],
            name="handle_link",
        )
        self.handle_link_meshes = handle_links_meshes
        self.handle_link_pos = common.to_tensor(
            np.array(
                [
                    meshes[0].bounding_box.center_mass
                    for i, meshes in enumerate(handle_links_meshes)
                ]
            ),
            device=self.device,
        )
        self.handle_link_goal = actors.build_sphere(
            self.scene,
            radius=0.02,
            color=[0, 1, 0, 1],
            name="handle_link_goal",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        )

    def _after_reconfigure(self, options: dict):
        self.object_zs = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            # this value is used to set object pose so the bottom is at z=0
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)

        # import ipdb; ipdb.set_trace()
        self.drawer_zs = []

        for drawer in self._drawers:
            collision_mesh = drawer.get_first_collision_mesh()
            self.drawer_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.drawer_zs = common.to_tensor(self.drawer_zs, device=self.device)

        # 0 when close, -1 when open
        target_qlimits = self.handle_link.joint.limits
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmax - (qmax - qmin) * self.min_open_frac

    def handle_link_positions(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            return transform_points(
                self.handle_link.pose.to_transformation_matrix().clone(),
                common.to_tensor(self.handle_link_pos, device=self.device),
            )
        return transform_points(
            self.handle_link.pose[env_idx].to_transformation_matrix().clone(),
            common.to_tensor(self.handle_link_pos[env_idx], device=self.device),
        )


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 0] = torch.rand((b,1)) * 0.2 - 0.1
            xyz[:, 1] = torch.rand((b,1)) * 0.3 - 0.3

            xyz[:, 2] = self.object_zs[env_idx]
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            drawer_xyz = torch.zeros((b, 3))
            drawer_xyz[:, 0] = -0.3 + (torch.rand((b,1)) * 0.4 - 0.2)
            drawer_xyz[:, 1] = 0.3 + (torch.rand((b,1)) * 0.2 - 0.1)
            drawer_xyz[:, 2] = self.drawer_zs[env_idx]
            drawer_qs = random_quaternions(b, lock_x=True, lock_y=True, bounds=(-np.pi/4, np.pi/4))
            self.drawer.set_pose(Pose.create_from_pq(p=drawer_xyz, q=drawer_qs))

            self.handle_link_goal.set_pose(
                Pose.create_from_pq(p=self.handle_link_positions(env_idx))
            )

            if self.robot_uids == "panda" or self.robot_uids == "panda_wristcam":
                # fmt: off
                qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
                )
                # fmt: on
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)

    def _after_control_step(self):
        # after each control step, we update the goal position of the handle link
        # for GPU sim we need to update the kinematics data to get latest pose information for up to date link poses
        # and fetch it, followed by an apply call to ensure the GPU sim is up to date
        if self.gpu_sim_enabled:
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        self.handle_link_goal.set_pose(
            Pose.create_from_pq(p=self.handle_link_positions())
        )
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()


    def evaluate(self):
        
        return {
            "success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        
        return 0

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8

    def get_prompt_content(self):
        
        if self.agent.controller.controllers['gripper'].qpos[0][0] < 0.02:
            gripper_state = 'closed'
        else:
            gripper_state = 'open'

        return {
            'task_desc': 'Open the drawer, pick up the mug and put it inside the drawer, and close the drawer.',
            'ground_truth': {
                'gripper_state': gripper_state,
            }
        }