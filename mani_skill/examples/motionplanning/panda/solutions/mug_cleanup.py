import argparse
import gymnasium as gym
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import StackThreeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.wrappers.record import RecordEpisode
import trimesh
from mani_skill.utils.geometry.geometry import transform_points

def to_rigid(T):
    R = T[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    T2 = np.eye(4)
    T2[:3, :3] = R
    T2[:3, 3] = T[:3, 3]
    return T2

def solve(env: StackThreeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    FINGER_LENGTH = 0.025
    env = env.unwrapped


    # grasp cubeA

    mesh = env.handle_link_meshes[0][0]
    obb_local = mesh.bounding_box_oriented
    T = obb_local.primitive.transform.copy()

    center_local = T[:3, 3]
    axes_local = T[:3, :3]
    extents = obb_local.primitive.extents
    U, _, Vt = np.linalg.svd(axes_local)
    axes_local = U @ Vt

    T_link = env.handle_link.pose.to_transformation_matrix()[0].cpu().numpy()

    center_world = T_link[:3, :3] @ center_local + T_link[:3, 3]
    axes_world = T_link[:3, :3] @ axes_local

    # axis_world = T_link[:3, 0]
    # axis_world /= np.linalg.norm(axis_world)
    # pull_dir = -axis_world
    joint = env.handle_link.joint
    joint_pose = joint.get_global_pose()
    T_joint = joint.get_global_pose().to_transformation_matrix().cpu().numpy()[0]
    axis_world = T_joint[:3, 0]
    axis_world /= np.linalg.norm(axis_world)

    pull_dir = -axis_world

    # import ipdb; ipdb.set_trace()



    radius = np.min(extents) / 2
    center_world = center_world + pull_dir * radius * 1.2

    T_final = np.eye(4)
    T_final[:3, :3] = axes_world
    T_final[:3, 3] = center_world

    obb = trimesh.primitives.Box(extents=extents, transform=T_final)


    # import ipdb; ipdb.set_trace()

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # Search a valid pose
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
        grasp_pose2 = grasp_pose * delta_pose
        res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
        if res == -1:
            continue
        grasp_pose = grasp_pose2
        break
    else:
        print("Fail to find a valid grasp pose")

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # pin the handle
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose * sapien.Pose([0, 0, 0.06]))


    now_pose = grasp_pose * sapien.Pose([0, 0, 0.06])
    # remove pull_dir's vertical component
    up_dir = np.array([0, 0, 1])
    pull_dir = pull_dir - np.dot(pull_dir, up_dir) * up_dir
    pull_dir /= np.linalg.norm(pull_dir)

    new_p = now_pose.p + pull_dir * 0.15
    new_q = now_pose.q  # 姿态保持不变
    new_pose = Pose.create_from_pq(new_p, new_q)

    res = planner.move_to_pose_with_screw(new_pose)











    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    # lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    # planner.move_to_pose_with_screw(lift_pose)

    # # -------------------------------------------------------------------------- #
    # # Stack
    # # -------------------------------------------------------------------------- #
    # goal_pose = env.cubeB.pose * sapien.Pose([0, 0, (env.cube_half_size[2] * 2).item()+0.01])
    # offset = (goal_pose.p - env.cubeA.pose.p).cpu().numpy()[0] # remember that all data in ManiSkill is batched and a torch tensor
    # align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    # planner.move_to_pose_with_screw(align_pose)

    # res = planner.open_gripper()

    # # --------------------------------------------------------------------------- #
    # # Lift
    # # --------------------------------------------------------------------------- #
    # lift_pose = sapien.Pose([0, 0, 0.1]) * align_pose
    # planner.move_to_pose_with_screw(lift_pose)


    # # grasp cubeC
    # obb = get_actor_obb(env.cubeC)

    # approaching = np.array([0, 0, -1])
    # target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    # grasp_info = compute_grasp_info_by_obb(
    #     obb,
    #     approaching=approaching,
    #     target_closing=target_closing,
    #     depth=FINGER_LENGTH,
    # )
    # closing, center = grasp_info["closing"], grasp_info["center"]
    # grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # # Search a valid pose
    # angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    # angles = np.repeat(angles, 2)
    # angles[1::2] *= -1
    # for angle in angles:
    #     delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
    #     grasp_pose2 = grasp_pose * delta_pose
    #     res = planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
    #     if res == -1:
    #         continue
    #     grasp_pose = grasp_pose2
    #     break
    # else:
    #     print("Fail to find a valid grasp pose")

    # # -------------------------------------------------------------------------- #
    # # Reach
    # # -------------------------------------------------------------------------- #

    # # first move at a higher level to avoid collision
    # reach_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
    # planner.move_to_pose_with_screw(reach_pose)

    # # -------------------------------------------------------------------------- #
    # # Grasp
    # # -------------------------------------------------------------------------- #
    # planner.move_to_pose_with_screw(grasp_pose)
    # planner.close_gripper()

    # # -------------------------------------------------------------------------- #
    # # Lift
    # # -------------------------------------------------------------------------- #
    # lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
    # planner.move_to_pose_with_screw(lift_pose)

    # # -------------------------------------------------------------------------- #
    # # Stack
    # # -------------------------------------------------------------------------- #
    # goal_pose = env.cubeA.pose * sapien.Pose([0, 0, (env.cube_half_size[2] * 2).item()+0.01])
    # offset = (goal_pose.p - env.cubeC.pose.p).cpu().numpy()[0] # remember that all data in ManiSkill is batched and a torch tensor
    # align_pose = sapien.Pose(lift_pose.p + offset, lift_pose.q)
    # planner.move_to_pose_with_screw(align_pose)

    # res = planner.open_gripper(t=10)




    planner.close()
    return res
