from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose  
from mani_skill.utils.logging_utils import logger

@register_env("StackPyramid-v1", max_episode_steps=350)
class StackPyramidEnv(BaseEnv):
    """
    **Task Description:**
    - The goal is to pick up a red cube, place it next to the green cube, and stack the blue cube on top of the red and green cube without it falling off.

    **Randomizations:**
    - all cubes have their z-axis rotation randomized
    - all cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

    **Success Conditions:**
    - the blue cube is static
    - the blue cube is on top of both the red and green cube (to within half of the cube size)
    - none of the red, green, blue cubes are grasped by the robot (robot must let go of the cubes)

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackPyramid-v1_rt.mp4"

    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["none", "sparse"]
    
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # pose = sapien_utils.look_at(eye=[0.3, 0, 0.4], target=[-0.05, 0, 0.1])
        # return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]
        pose = sapien_utils.look_at(eye=[0.5, 0, 0.6], target=[-0.1, 0, -0.1])
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 3, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        # return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
        pose = sapien_utils.look_at(eye=[0.5, 0, 0.6], target=[-0.1, 0, -0.1])
        return [CameraConfig("render_camera", pose, 256, 256, np.pi / 3, 0.01, 100)]


    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene, half_size=0.02, color=[1, 0, 0, 1], name="cubeA", initial_pose=sapien.Pose(p=[0, 0, 0.2])
        )
        self.cubeB = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 1, 0, 1], name="cubeB", initial_pose=sapien.Pose(p=[1, 0, 0.2])
        )
        self.cubeC = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 0, 1, 1], name="cubeC", initial_pose=sapien.Pose(p=[-1, 0, 0.2])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 2] = 0.02
            xy = xyz[:, :2]
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b, device=self.device)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02]))
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)
            cubeC_xy = xy + sampler.sample(radius, 100, verbose=False)

            # Cube A
            xyz[:, :2] = cubeA_xy
            
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )

            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Cube B
            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
            
            # Cube C
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

        offset_AB = pos_A - pos_B
        offset_BC = pos_B - pos_C
        offset_AC = pos_A - pos_C

        def evaluate_cube_distance(offset, cube_a, cube_b, top_or_next):
            xy_flag = (torch.linalg.norm(offset[..., :2], axis=1) 
                       <= torch.linalg.norm(2*self.cube_half_size[:2]) 
                       + 0.01
                       )
            z_flag = torch.abs(offset[..., 2]) > 0.02
            if top_or_next == "top":
                is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
            elif top_or_next == "next_to":
                is_cubeA_on_cubeB = xy_flag
            else:
                return NotImplementedError(f"Expect top_or_next to be either 'top' or 'next_to', got {top_or_next}")
            
            is_cubeA_static = cube_a.is_static(lin_thresh=1e-2, ang_thresh=0.5)
            is_cubeA_grasped = self.agent.is_grasping(cube_a)

            success = is_cubeA_on_cubeB & is_cubeA_static & (~is_cubeA_grasped)            
            return success.bool()

        success_A_B = evaluate_cube_distance(offset_AB, self.cubeA, self.cubeB, "next_to")
        success_C_B = evaluate_cube_distance(offset_BC, self.cubeC, self.cubeB, "top")
        success_C_A = evaluate_cube_distance(offset_AC, self.cubeC, self.cubeA, "top")
        success = torch.logical_and(success_A_B, torch.logical_and(success_C_B, success_C_A))


        is_cubeC_lifted = pos_C[..., 2] > 0.08
        is_cubeC_grasped = self.agent.is_grasping(self.cubeC)
        stage_2_success = torch.logical_and(is_cubeC_lifted, is_cubeC_grasped)


        return {
            "stage_1_success": success_A_B,
            "stage_2_success": stage_2_success,
            "stage_3_success": success,
            "success": success,
        }
    
    def check_stage(self):
        info = self.evaluate()
        if info['success']:
            return 3
    
        if info['stage_1_success']:
            if info['stage_2_success']:
                return 2
            else:
                return 1
            
        return 0

    @property
    def stage_cnt(self):
        return 3

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
                cubeB_to_cubeC_pos=self.cubeC.pose.p - self.cubeB.pose.p,
                cubeA_to_cubeC_pos=self.cubeC.pose.p - self.cubeA.pose.p,
            )
        return obs
    

    def get_prompt_content(self):

        if self.agent.controller.controllers['gripper'].qpos[0][0] < 0.03:
            gripper_state = 'closed'
        else:
            gripper_state = 'open'

        instruction_for_stage_id = [
            'You need to push the red cube next to the green cube (the distance can be half of the cube_half_size to ensure the cubes are next to each other) at very first.',
            'The red and green cube are already next to each other. Now you need to grasp and lift up the blue cube.(at least z=0.1)',
            'The blue cube is already lifted up. Now you need to place the blue cube on top of the red and green cube (which should be in the middle, like (red_cube_pos + green_cube_pos)/2), and make sure the blue cube is static and not grasped by the robot.',
        ]

        return {
            'task_desc': 'Push the red cube next to the green cube, then stack the blue cube on top of the red and green cube.',
            'ground_truth': {
                'additional_info': 'In order to push the red cube, the solution is that move the gripper above the red cube, close the gripper(notice that NOT grasping the cube), then move downwards to the red cube (z = half of the cube_half_size) to make a force to squeeze the red cube, then move the gripper next to the green cube. In this situation, the red cube will be pushed next to the green cube.',
                'current_stage': instruction_for_stage_id[self.check_stage()],
                'gripper_state': gripper_state,
                'red_cube_pos': self.cubeA.pose.p.cpu().numpy().tolist(),
                'red_cube_quat': self.cubeA.pose.q.cpu().numpy().tolist(),
                'green_cube_pos': self.cubeB.pose.p.cpu().numpy().tolist(),
                'green_cube_quat': self.cubeB.pose.q.cpu().numpy().tolist(),
                'blue_cube_pos': self.cubeC.pose.p.cpu().numpy().tolist(),
                'blue_cube_quat': self.cubeC.pose.q.cpu().numpy().tolist(),
                'cube_half_size': 0.02
            }
        }