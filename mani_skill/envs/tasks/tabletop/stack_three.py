from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig


@register_env("StackThree-v1", max_episode_steps=500)
class StackThreeEnv(BaseEnv):
    """
    **Task Description:**
    The goal is to pick up a red cube (A) and stack it on top of a green cube (B), then stack a blue cube (C) on top of the red cube (A).

    **Randomizations:**
    - Three cubes have their z-axis rotation randomized
    - Three cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

    **Success Conditions:**
    - the red cube is on top of the green cube (to within half of the cube size)
    - the red cube is static
    - the red cube is not being grasped by the robot (robot must let go of the cube)
    - same conditions for the blue cube on top of the red cube
    """

    # _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    cube_min_init_dist = 0.04
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
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

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
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        )
        self.cubeC = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 0, 1, 1],
            name="cubeC",
            initial_pose=sapien.Pose(p=[0, 1, 0.1]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + self.cube_min_init_dist
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)
            cubeC_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            xyz[:, :2] = cubeC_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeC.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        pos_C = self.cubeC.pose.p

        # 1. grasp cubeA
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        is_cubeA_lifted = pos_A[..., 2] > self.cube_half_size[..., 2] * 4
        stage_1_success = is_cubeA_grasped * is_cubeA_lifted

        # 2. stack cubeA on cubeB
        offset = pos_A - pos_B
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        # NOTE (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        # is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        stage_2_success = is_cubeA_on_cubeB * (~is_cubeA_grasped)


        # 3. grasp cubeC
        is_cubeC_grasped = self.agent.is_grasping(self.cubeC)
        is_cubeC_lifted = pos_C[..., 2] > self.cube_half_size[..., 2] * 6
        stage_3_success = is_cubeC_grasped * is_cubeC_lifted

        # 4. stack cubeC on cubeA
        offset = pos_C - pos_A
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeC_on_cubeA = torch.logical_and(xy_flag, z_flag)
        # is_cubeC_static = self.cubeC.is_static(lin_thresh=1e-1, ang_thresh=0.5)
        is_cubeC_grasped = self.agent.is_grasping(self.cubeC)
        stage_4_success = is_cubeC_on_cubeA * (~is_cubeC_grasped)




        success = torch.logical_and(
            stage_2_success, stage_4_success
        )


        return {
            "stage_1_success": stage_1_success,
            "stage_2_success": stage_2_success,
            "stage_3_success": stage_3_success,
            "stage_4_success": stage_4_success,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                cubeC_pose=self.cubeC.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeC_pos=self.cubeC.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
                cubeA_to_cubeC_pos=self.cubeC.pose.p - self.cubeA.pose.p,
                cubeB_to_cubeC_pos=self.cubeC.pose.p - self.cubeB.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

        # # grasp and place reward
        # cubeA_pos = self.cubeA.pose.p
        # cubeB_pos = self.cubeB.pose.p
        # goal_xyz = torch.hstack(
        #     [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        # )
        # cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        # place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)

        # reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

        # # ungrasp and static reward
        # gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
        #     self.device
        # )  # NOTE: hard-coded with panda
        # is_cubeA_grasped = info["is_cubeA_grasped"]
        # ungrasp_reward = (
        #     torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        # )
        # ungrasp_reward[~is_cubeA_grasped] = 1.0
        # v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        # av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        # static_reward = 1 - torch.tanh(v * 10 + av)
        # reward[info["is_cubeA_on_cubeB"]] = (
        #     6 + (ungrasp_reward + static_reward) / 2.0
        # )[info["is_cubeA_on_cubeB"]]

        # reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8

    def get_prompt_content(self):
        
        return {
            'task_desc': 'Stack the red cube on the green cube, and then stack the blue cube on the red cube to form a tower.',
            'ground_truth': {
                'red_cube_pos': self.cubeA.pose.p.cpu().numpy().tolist(),
                'red_cube_quat': self.cubeA.pose.q.cpu().numpy().tolist(),
                'green_cube_pos': self.cubeB.pose.p.cpu().numpy().tolist(),
                'green_cube_quat': self.cubeB.pose.q.cpu().numpy().tolist(),
                'blue_cube_pos': self.cubeC.pose.p.cpu().numpy().tolist(),
                'blue_cube_quat': self.cubeC.pose.q.cpu().numpy().tolist(),
                'cube_half_size': 0.02
            }
        }