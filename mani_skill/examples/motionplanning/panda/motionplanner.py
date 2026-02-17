import mplib
import torch
import numpy as np
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.two_finger_gripper.motionplanner import TwoFingerGripperMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.ee_planner import NonConvexPlanner
from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.agents.utils import get_active_joint_indices
from mani_skill.agents.controllers.utils.qpos_pose_transfer import transfer_qpos_2_ee_pose
from mani_skill.agents.controllers.utils.delta_pose import controller_delta_pose_calculate
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix, matrix_to_euler_angles, matrix_to_quaternion

def get_numpy(data, device="cpu"):
    if isinstance(data, torch.Tensor):
        if device == "cpu":
            return data.numpy()
        else:
            return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError("parameter passed is not torch.tensor")



class PandaArmMotionPlanningSolver(TwoFingerGripperMotionPlanningSolver):
    OPEN = 1
    CLOSED = -1
    MOVE_GROUP = "panda_hand_tcp"

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose = None,  # TODO mplib doesn't support robot base being anywhere but 0
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
    ):
        super().__init__(env, debug, vis, base_pose, visualize_target_grasp_pose, print_env_info, joint_vel_limits, joint_acc_limits)

        # import ipdb; ipdb.set_trace()

        self.kinematics = Kinematics(
            urdf_path=self.env.agent.urdf_path,
            end_link_name=self.env.agent.ee_link_name,
            articulation=self.env.agent.robot,
            active_joint_indices=get_active_joint_indices(self.env.agent.robot, self.env.agent.arm_joint_names),
        )
    
    def setup_planner(self):    
        move_group = self.MOVE_GROUP if hasattr(self, "MOVE_GROUP") else "eef"
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = NonConvexPlanner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=move_group,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
        planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
        return planner

    def preprocess_qpos(self, abs_qpos):
        """
            transfer raw_qpos(q_pos) to action(delta_ee_pose)
            just for qpos in pd_ee_delta_pose control mode
        """
        
        # import ipdb; ipdb.set_trace()

        target_pose = transfer_qpos_2_ee_pose(self.env, self.kinematics, abs_qpos, world_frame=False)
        current_pose = self.env.agent.ee_pose_at_robot_base
            
        delta_xyz, delta_euler_angle = controller_delta_pose_calculate(
            self.env.agent.controller.configs["arm"].frame,
            self.env.agent.controller.configs["arm"].normalize_action,
            float(self.env.agent.controller.configs["arm"].pos_upper),
            float(self.env.agent.controller.configs["arm"].rot_upper),
            current_pose.to_transformation_matrix().squeeze(0),
            target_pose.to_transformation_matrix().squeeze(0),
            self.env.device
        )

        delta_action = torch.cat([delta_xyz, delta_euler_angle,
                                    torch.tensor([self.gripper_state]).to(self.env.unwrapped.device)])
        
        abs_action = torch.cat(
            [
                target_pose.p[0],
                matrix_to_euler_angles(quaternion_to_matrix(target_pose.q[0]),"XYZ"),
                torch.tensor([self.gripper_state]).to(self.env.unwrapped.device),
            ]
        )

        return (get_numpy(delta_action, self.env.unwrapped.device), 
                get_numpy(abs_action, self.env.unwrapped.device), 
                target_pose)

    def follow_path(self, result, refine_steps: int = 0): 
        n_step = result["position"].shape[0]

        # import ipdb; ipdb.set_trace()

        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            elif self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            elif self.control_mode == "pd_joint_delta_pos":
                action = self.qpos_action_to_pd_joint_delta_pos_action(action)
            elif self.control_mode == "pd_ee_delta_pose" :
                delta_action, abs_action, abs_action_pose = self.preprocess_qpos(qpos) # actually delta action
                action = delta_action
            elif self.control_mode == "pd_ee_pose":
                delta_action, abs_action, abs_action_pose = self.preprocess_qpos(qpos)
                action = abs_action
            else:
                raise ValueError(f"motion planning doesn't support control mode {self.control_mode}")
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if truncated:
                return None
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def open_gripper(self,t=6, gripper_state=None):
        if gripper_state is None:
            gripper_state = self.OPEN
        self.gripper_state = gripper_state
        qpos = get_numpy(self.robot.get_qpos()[0, :len(self.planner.joint_vel_limits)], device=self.env.unwrapped.device)
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            elif self.control_mode == "pd_joint_pos_vel":
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            elif self.control_mode == "pd_joint_delta_pos":
                action = self.qpos_action_to_pd_joint_delta_pos_action(np.hstack([qpos, self.gripper_state]))
            elif self.control_mode == "pd_ee_delta_pose":
                action = np.hstack([np.zeros((6,)), self.gripper_state])
            elif self.control_mode == "pd_ee_pose":
                action = torch.cat([self.env.agent.ee_pose_at_robot_base.p[0],
                                    matrix_to_euler_angles(quaternion_to_matrix(self.env.agent.ee_pose_at_robot_base.q[0]),"XYZ"),
                                    torch.tensor([self.gripper_state]).to(self.env.unwrapped.device)])
            else:
                raise ValueError(f"motion planning doesn't support control mode {self.control_mode}")
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t=6, gripper_state=None):
        # import ipdb; ipdb.set_trace()

        if gripper_state is None:
            gripper_state = self.CLOSED
        self.gripper_state = gripper_state
        qpos = get_numpy(self.robot.get_qpos()[0, :len(self.planner.joint_vel_limits)], device=self.env.unwrapped.device)
        for i in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            elif self.control_mode == "pd_joint_pos_vel":
                action = np.hstack([qpos, qpos * 0, self.gripper_state])
            elif self.control_mode == "pd_joint_delta_pos":
                action = self.qpos_action_to_pd_joint_delta_pos_action(np.hstack([qpos, self.gripper_state]))
            elif self.control_mode == "pd_ee_delta_pose":
                action = np.hstack([np.zeros((6,)), self.gripper_state])
            elif self.control_mode == "pd_ee_pose":
                action = torch.cat([self.env.agent.ee_pose_at_robot_base.p[0],
                                    matrix_to_euler_angles(quaternion_to_matrix(self.env.agent.ee_pose_at_robot_base.q[0]),"XYZ"),
                                    torch.tensor([self.gripper_state]).to(self.env.unwrapped.device)])
            else:
                raise ValueError(f"motion planning doesn't support control mode {self.control_mode}")
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

