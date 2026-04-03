from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch
import trimesh

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
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
from mani_skill import MY_ASSET_DIR


@register_env("MugCleanup-v1", max_episode_steps=400)
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
    min_open_frac = 0.10
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, -0.1])
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at([0.6, -0.7, 0.6], [0.0, 0.0, 0.35])
        # return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, -0.1])
        return [CameraConfig("render_camera", pose, 256, 256, np.pi / 2, 0.01, 100)]

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=100, control_freq=10)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()


        model_id = '025_mug'
        # model_id = '061_foam_brick'
        # model_id = '062_dice'
        self._objs: List[Actor] = []
        self.obj_heights = []

        builder = actors.get_actor_builder(
            self.scene,
            id=f"ycb:{model_id}",
            scale=0.9
            # scale=4.0
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

        mjcf_path = MY_ASSET_DIR / 'assets/drawer_long.xml'
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
            color=[0, 0, 0, 0],
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

        # 0 when close, -1 when open to the max
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
            region = [[-0.2, -0.35], [0.03, -0.28]]
            sampler = randomization.samplers.UniformPlacementSampler(
                bounds=region,
                batch_size=b,
                device=self.device,
            )

            # xyz[:, 0] = torch.rand((b,1)) * 0.2 - 0.1
            # xyz[:, 1] = torch.rand((b,1)) * 0.2 - 0.3
            xyz[:, :2] = sampler.sample(radius=0.05, max_trials=100)

            xyz[:, 2] = self.object_zs[env_idx]
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            drawer_xyz = torch.zeros((b, 3))
            drawer_region = [[-0.15, 0.22], [0.05, 0.28]]
            drawer_sampler = randomization.samplers.UniformPlacementSampler(
                bounds=drawer_region,
                batch_size=b,
                device=self.device,
            )
            drawer_sampler.fixture_positions = xyz[:, :2].unsqueeze(0)
            drawer_sampler.fixtures_radii = torch.tensor([0.05],device=self.device)

            drawer_xyz[:, :2] = drawer_sampler.sample(radius=0.4, max_trials=100)
            # import ipdb; ipdb.set_trace()

            # drawer_xyz[:, 0] = -0.3 + (torch.rand((b,1)) * 0.2 - 0.1)
            # drawer_xyz[:, 1] = 0.28 + (torch.rand((b,1)) * 0.03)
            drawer_xyz[:, 2] = self.drawer_zs[env_idx]
            drawer_qs = random_quaternions(b, lock_x=True, lock_y=True, bounds=(-np.pi/8, np.pi/8))
            self.drawer.set_pose(Pose.create_from_pq(p=drawer_xyz, q=drawer_qs))

            # close all drawers when initializing
            qlimits = self.drawer.get_qlimits()  # [b, self.cabinet.max_dof, 2])
            self.drawer.set_qpos(qlimits[env_idx, :, 1])
            self.drawer.set_qvel(self.drawer.qpos[env_idx] * 0)

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
        open_enough = self.handle_link.joint.qpos <= self.target_qpos
        link_is_static = (
            torch.linalg.norm(self.handle_link.angular_velocity, axis=1) <= 1
        ) & (torch.linalg.norm(self.handle_link.linear_velocity, axis=1) <= 0.1) 

        stage_1_success = open_enough & link_is_static

        is_obj_grasped = self.agent.is_grasping(self.obj)
        is_obj_lifted = self.obj.pose.p[:, 2] >= 0.2
        
        # TODO: solve the case when num_envs > 1
        is_obj_in_drawer = torch.zeros_like(is_obj_grasped, dtype=torch.bool)
        all_contacts = self.scene.get_contacts()
        for c in all_contacts:
            b0, b1 = c.bodies
            if 'drawer' in b0.name and 'mug' in b1.name:
                is_obj_in_drawer = torch.ones_like(is_obj_grasped, dtype=torch.bool)
            if 'drawer' in b1.name and 'mug' in b0.name:
                is_obj_in_drawer = torch.ones_like(is_obj_grasped, dtype=torch.bool)
        
        stage_2_success = is_obj_in_drawer & (~is_obj_grasped) & (~is_obj_lifted)

        close_enough = self.handle_link.joint.qpos >= -0.02
        stage_3_success = close_enough & link_is_static
        
        return {
            "success": stage_2_success & stage_3_success,
            "stage_1_success": stage_1_success,
            "stage_2_success": stage_2_success,
            "stage_3_success": stage_3_success,
        }
        
    def check_stage(self):
        info = self.evaluate()
        if info['stage_2_success'] & info['stage_3_success']:
            return 3
        if info['stage_2_success'] & (~info['stage_3_success']):
            return 2
        if info['stage_1_success']:
            return 1
        return 0
    
    @property
    def stage_cnt(self):
        return 3

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_handle_pos=self.handle_link_positions() - self.agent.tcp.pose.p,
                target_link_qpos=self.handle_link.joint.qpos,
                target_handle_pos=self.handle_link_positions(),
            )

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

        FINGER_LENGTH = 0.025
        
        mesh = self.handle_link_meshes[0][0]
        obb_local = mesh.bounding_box_oriented
        T = obb_local.primitive.transform.copy()

        center_local = T[:3, 3]
        axes_local = T[:3, :3]
        extents = obb_local.primitive.extents
        U, _, Vt = np.linalg.svd(axes_local)
        axes_local = U @ Vt

        T_link = self.handle_link.pose.to_transformation_matrix()[0].cpu().numpy()

        center_world = T_link[:3, :3] @ center_local + T_link[:3, 3]
        axes_world = T_link[:3, :3] @ axes_local
        
        R = self.drawer.pose.to_transformation_matrix()[0, :3, :3]
        v = np.array([0,1,0])
        # apply q on v
        v_rotated = R @ v
        v_rotated = np.array(v_rotated)

        pull_dir = -v_rotated
        pull_dir /= np.linalg.norm(pull_dir)

        # import ipdb; ipdb.set_trace()


        radius = np.min(extents) / 2
        center_world = center_world + pull_dir * radius * 1.2

        T_final = np.eye(4)
        T_final[:3, :3] = axes_world
        T_final[:3, 3] = center_world

        obb = trimesh.primitives.Box(extents=extents, transform=T_final)
        handle_extent = extents.tolist()
        handle_transform = T_final.copy()


        # import ipdb; ipdb.set_trace()

        approaching = np.array([0, 0, -1])
        target_closing = self.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = self.agent.build_grasp_pose(approaching, closing, center)
        pin_pose = sapien.Pose([0, 0, -0.03]) * grasp_pose



        R = self.obj.pose.to_transformation_matrix()[0, :3, :3]
        rim_dir = R @ np.array([1,0,0])
        rim_dir = rim_dir / np.linalg.norm(rim_dir)
        rim_dir = np.array(rim_dir)

        obb_obj = get_actor_obb(self.obj)
        # rim 半径（用 mug 高度近似）
        rim_radius = obb_obj.extents[2] / 2 * 1.2
        T = obb_obj.primitive.transform.copy()
        extents = obb_obj.extents

        center = T[:3, 3]
        new_center = center + rim_dir * rim_radius
        

        T_final = np.eye(4)
        T_final[:3, :3] = T[:3, :3]
        T_final[:3, 3] = new_center

        obb = trimesh.primitives.Box(extents=extents, transform=T_final)
        mug_extent = extents.tolist()
        mug_transform = T_final.copy()

        approaching = np.array([0, 0, -1])
        target_closing = self.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = self.agent.build_grasp_pose(approaching, closing, center)
        grasp_mug_pose = sapien.Pose([0, 0, -0.01]) * grasp_pose

        place_mug_pos = pin_pose.p - pull_dir * 0.20


        instruction_for_stage_id = [
            'You need to open the drawer first.',
            'The drawer has already been opened. Now you can pick up the mug and put it inside the drawer.',
            'The mug has been put inside the drawer. Now you only need to close the drawer.',
            'The task is completed successfully.',
        ]

        info_for_stage_id = [
            'When you are completing the open drawer stage, the place_mug_when_drawer_open_pos is wrong and useless. You should **never** close the gripper because you can only use one side of the gripper to pin on the handle. Always remember the pull_direction is definitely correct, and maintain the gripper open when pull the drawer. Never use the grasp_mug_pos and grasp_mug_quat in this stage because they are wrong.',
            'When you are completing the pick and place mug stage, you can use the place_mug_when_drawer_open_pos to help you move to a proper position above the drawer to put the mug in. Remember, if the gripper is currently pined on the handle, you must move up first to a higher position (z>0.2) to avoid collisions with the drawer, then go to pick the mug.',
            'When you are completing the close drawer stage, you can use the now_handle_pos and now_handle_quat to pin on the handle, and you do not need to close the gripper because you can only use one side of the gripper to pin on the handle. Then close the drawer along the negative pull_direction with the scale of (pull_open_length + 0.02) to make sure the drawer is closed completely.',
            'The task is completed successfully.'
        ]


        return {
            'task_desc': 'Hook the handle with the gripper of a single side to open the drawer, pick up the mug and put it inside the drawer, and close the drawer.',
            'ground_truth': {
                'current_stage': instruction_for_stage_id[self.check_stage()],
                'current_stage_info': info_for_stage_id[self.check_stage()],
                'tcp_pos': self.agent.tcp.pose.p.tolist(),
                'tcp_quat': self.agent.tcp.pose.q.tolist(),
                'gripper_state': gripper_state,
                'now_handle_pos': pin_pose.p.tolist(),
                'now_handle_quat': pin_pose.q.tolist(),
                'pull_direction': pull_dir.tolist(),
                'grasp_mug_pos': grasp_mug_pose.p.tolist(),               
                'grasp_mug_quat': grasp_mug_pose.q.tolist(),
                'pull_open_length': 0.21,
                'place_mug_when_drawer_open_pos': place_mug_pos.tolist(),
                'additional_info': 'Always check first, if you want to move to some higher position, and the ground truth shows the gripper is in some lower position (z<0.2), always move up first to a higher position. When you want to approach some determined position, always reach to a little above first, then move down to the target position. When you finish reach one target, and want to move to another target, always move up first to avoid collision. For this task, use 0.15 meters above the target position as the approach position. When you grasp the mug, use 0.21 meters above the grasp pose to avoid collisions with the drawer. place_mug_when_drawer_open_pos only works when you are in the stage of putting mug into the drawer, So do not consider it in other stages.'
            }
        }
    
        # return {
        #     'task_desc': 'Hook the handle with the gripper of a single side to open the drawer, pick up the mug and put it inside the drawer, and close the drawer.',
        #     'ground_truth': {
        #         'current_stage': instruction_for_stage_id[self.check_stage()],
        #         'current_stage_info': info_for_stage_id[self.check_stage()],
        #         'tcp_pos': self.agent.tcp.pose.p.tolist(),
        #         'tcp_quat': self.agent.tcp.pose.q.tolist(),
        #         'gripper_state': gripper_state,
        #         # 'now_handle_pos': pin_pose.p.tolist(),
        #         # 'now_handle_quat': pin_pose.q.tolist(),
        #         'handle_extent': handle_extent,
        #         'handle_transform': handle_transform.tolist(),
        #         'approaching_direction': approaching.tolist(),
        #         'closing_direction': target_closing.tolist(),
        #         'depth': FINGER_LENGTH,

        #         'pull_direction': pull_dir.tolist(),
        #         # 'grasp_mug_pos': grasp_mug_pose.p.tolist(),               
        #         # 'grasp_mug_quat': grasp_mug_pose.q.tolist(),
        #         'mug_extent': mug_extent,
        #         'mug_transform': mug_transform.tolist(),
                
        #         'pull_open_length': 0.21,
        #         # 'place_mug_when_drawer_open_pos': place_mug_pos.tolist(),
        #         'place_mug_length': 0.20,
        #         'additional_info': 'Always check first, if you want to move to some higher position, and the ground truth shows the gripper is in some lower position (z<0.2), always move up first to a higher position. When you want to approach some determined position, always reach to a little above first, then move down to the target position. When you finish reach one target, and want to move to another target, always move up first to avoid collision. For this task, use 0.15 meters above the target position as the approach position. When you grasp the mug, use 0.21 meters above the grasp pose to avoid collisions with the drawer. place_mug_when_drawer_open_pos only works when you are in the stage of putting mug into the drawer, So do not consider it in other stages.'
        #     }
        # }

    def get_fix_prompt_content(self):

        FINGER_LENGTH = 0.025
        
        mesh = self.handle_link_meshes[0][0]
        obb_local = mesh.bounding_box_oriented
        T = obb_local.primitive.transform.copy()

        center_local = T[:3, 3]
        axes_local = T[:3, :3]
        extents = obb_local.primitive.extents
        U, _, Vt = np.linalg.svd(axes_local)
        axes_local = U @ Vt

        T_link = self.handle_link.pose.to_transformation_matrix()[0].cpu().numpy()

        center_world = T_link[:3, :3] @ center_local + T_link[:3, 3]
        axes_world = T_link[:3, :3] @ axes_local
        
        handle_center_pos = center_world


        R = self.drawer.pose.to_transformation_matrix()[0, :3, :3]
        v = np.array([0,1,0])
        # apply q on v
        v_rotated = R @ v
        v_rotated = np.array(v_rotated)

        pull_dir = -v_rotated
        pull_dir /= np.linalg.norm(pull_dir)

        # import ipdb; ipdb.set_trace()


        radius = np.min(extents) / 2
        center_world = center_world + pull_dir * radius * 1.2

        T_final = np.eye(4)
        T_final[:3, :3] = axes_world
        T_final[:3, 3] = center_world

        obb = trimesh.primitives.Box(extents=extents, transform=T_final)
        handle_extent = extents.tolist()
        handle_transform = T_final.copy()


        # import ipdb; ipdb.set_trace()

        approaching = np.array([0, 0, -1])
        target_closing = self.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = self.agent.build_grasp_pose(approaching, closing, center)
        pin_pose = sapien.Pose([0, 0, -0.03]) * grasp_pose



        R = self.obj.pose.to_transformation_matrix()[0, :3, :3]
        rim_dir = R @ np.array([1,0,0])
        rim_dir = rim_dir / np.linalg.norm(rim_dir)
        rim_dir = np.array(rim_dir)

        obb_obj = get_actor_obb(self.obj)
        # rim 半径（用 mug 高度近似）
        rim_radius = obb_obj.extents[2] / 2 * 1.2
        T = obb_obj.primitive.transform.copy()
        extents = obb_obj.extents

        center = T[:3, 3]
        new_center = center + rim_dir * rim_radius
        

        T_final = np.eye(4)
        T_final[:3, :3] = T[:3, :3]
        T_final[:3, 3] = new_center

        obb = trimesh.primitives.Box(extents=extents, transform=T_final)
        mug_extent = extents.tolist()
        mug_transform = T_final.copy()
        mug_center_pos = center

        approaching = np.array([0, 0, -1])
        target_closing = self.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = self.agent.build_grasp_pose(approaching, closing, center)
        grasp_mug_pose = sapien.Pose([0, 0, -0.01]) * grasp_pose

        place_mug_pos = pin_pose.p - pull_dir * 0.20


        instruction_for_stage_id = [
            'You need to open the drawer first.',
            'The drawer has already been opened. Now you can pick up the mug and put it inside the drawer.',
            'The mug has been put inside the drawer. Now you only need to close the drawer.',
            'The task is completed successfully.',
        ]

        # info_for_stage_id = [
        #     'When you are completing the open drawer stage, the place_mug_when_drawer_open_pos is wrong and useless. You should **never** close the gripper because you can only use one side of the gripper to pin on the handle. Always remember the pull_direction is definitely correct, and maintain the gripper open when pull the drawer. Never use the grasp_mug_pos and grasp_mug_quat in this stage because they are wrong.',
        #     'When you are completing the pick and place mug stage, you can use the place_mug_when_drawer_open_pos to help you move to a proper position above the drawer to put the mug in. Remember, if the gripper is currently pined on the handle, you must move up first to a higher position (z>0.2) to avoid collisions with the drawer, then go to pick the mug.',
        #     'When you are completing the close drawer stage, you can use the now_handle_pos and now_handle_quat to pin on the handle, and you do not need to close the gripper because you can only use one side of the gripper to pin on the handle. Then close the drawer along the negative pull_direction with the scale of (pull_open_length + 0.02) to make sure the drawer is closed completely.',
        #     'The task is completed successfully.'
        # ]

        # import ipdb; ipdb.set_trace()
        return {
            'task_desc': 'Hook the handle with the gripper of a single side to open the drawer, pick up the mug and put it inside the drawer, and close the drawer.',
            'ground_truth': {
                'current_stage': instruction_for_stage_id[self.check_stage()],
                'current_tiny_task': 'Pick up the mug.',
                # 'last_error': 'The gripper position xy is not aligned with the mug position xy.',
                # 'current_stage_info': info_for_stage_id[self.check_stage()],
                'tcp_pos': self.agent.tcp.pose.p.tolist(),
                'camera_pos': self.agent.robot.links_map['camera_link'].pose.p.tolist(),
                # 'tcp_quat': self.agent.tcp.pose.q.tolist(),
                # 'gripper_state': gripper_state,

                # 'now_handle_pos': pin_pose.p.tolist(),
                # 'now_handle_quat': pin_pose.q.tolist(),
                'handle_extent': handle_extent,
                'handle_center_pos': handle_center_pos.tolist(),
                # 'handle_transform': handle_transform.tolist(),


                'pull_direction': pull_dir.tolist(),

                # 'grasp_mug_pos': grasp_mug_pose.p.tolist(),               
                # 'grasp_mug_quat': grasp_mug_pose.q.tolist(),
                'mug_extent': mug_extent,
                'mug_center_pos': mug_center_pos.tolist(),


                # 'mug_transform': mug_transform.tolist(),

                
                'additional_info': 'When you grasp the mug, you can not directly use the mug_center_pos because you can not grasp the mug from its center. You should use the extent to compute a better grasping position, such as the top rim position of the mug.',
                # 'pull_open_length': 0.21,
                # 'place_mug_when_drawer_open_pos': place_mug_pos.tolist(),
                # 'additional_info': 'Always check first, if you want to move to some higher position, and the ground truth shows the gripper is in some lower position (z<0.2), always move up first to a higher position. When you want to approach some determined position, always reach to a little above first, then move down to the target position. When you finish reach one target, and want to move to another target, always move up first to avoid collision. For this task, use 0.15 meters above the target position as the approach position. When you grasp the mug, use 0.21 meters above the grasp pose to avoid collisions with the drawer. place_mug_when_drawer_open_pos only works when you are in the stage of putting mug into the drawer, So do not consider it in other stages.'
            }
        }