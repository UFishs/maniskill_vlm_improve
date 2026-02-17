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

def solve(env: StackThreeEnv, seed=None, debug=False, vis=False):
    # import ipdb; ipdb.set_trace()

    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
        "pd_ee_pose",
        "pd_ee_delta_pose",
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


    # Grasp the drawer handle
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
    
    R = env.drawer.pose.to_transformation_matrix()[0, :3, :3]
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
        return -1

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = sapien.Pose([0, 0, 0.05]) * grasp_pose
    res = planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # pin the handle
    # -------------------------------------------------------------------------- #
    pin_pose = sapien.Pose([0, 0, -0.03]) * grasp_pose
    
    res = planner.move_to_pose_with_screw(pin_pose)



    # -------------------------------------------------------------------------- #
    # Pull
    # -------------------------------------------------------------------------- #
    # remove pull_dir's vertical component
    up_dir = np.array([0, 0, 1])
    pull_dir = pull_dir - np.dot(pull_dir, up_dir) * up_dir
    pull_dir /= np.linalg.norm(pull_dir)

    new_p = pin_pose.p + pull_dir * 0.23
    new_q = pin_pose.q  # 姿态保持不变
    pull_pose = sapien.pysapien.Pose(new_p, new_q)


    res = planner.move_to_pose_with_screw(pull_pose)



    # -------------------------------------------------------------------------- #
    # Lift up after pull
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.10]) * pull_pose)



    # -------------------------------------------------------------------------- #
    # Grasp the mug
    # -------------------------------------------------------------------------- #

    # import ipdb; ipdb.set_trace()
    # R = env.obj.pose.to_transformation_matrix()[0, :3, :3]
    # rim_dir = R @ np.array([1,0,0])
    # rim_dir = rim_dir / np.linalg.norm(rim_dir)
    # rim_dir = np.array(rim_dir)

    obb_obj = get_actor_obb(env.obj)
    # rim 半径（用 mug 高度近似）
    # rim_radius = obb_obj.extents[2] / 2 * 1.2
    # T = obb_obj.primitive.transform.copy()
    # extents = obb_obj.extents

    # center = T[:3, 3]
    # new_center = center + rim_dir * rim_radius
    

    # T_final = np.eye(4)
    # T_final[:3, :3] = T[:3, :3]
    # T_final[:3, 3] = new_center

    # obb = trimesh.primitives.Box(extents=extents, transform=T_final)

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        obb_obj,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]


    # # 抓取法向（沿rim平面外侧）
    # # remove closing's vertical component
    # rim_dir = closing - np.dot(closing, up_dir) * up_dir
    # rim_dir = rim_dir / np.linalg.norm(rim_dir)

    # # rim 半径（用 mug 高度近似）
    # rim_radius = obb_obj.extents[2] / 2 * 1.5

    # center = center + rim_dir * rim_radius

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
        return -1

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = sapien.Pose([0, 0, 0.18]) * grasp_pose 
    res = planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    # import ipdb; ipdb.set_trace()

    res = planner.move_to_pose_with_screw(sapien.Pose([0, 0, -0.01]) * grasp_pose)
    res = planner.close_gripper()
    res = planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.21]) * grasp_pose)



    # -------------------------------------------------------------------------- #
    # Place the mug back to the drawer
    # -------------------------------------------------------------------------- #
    last_pose = sapien.Pose([0, 0, 0.18]) * grasp_pose
    new_p = pull_pose.p - pull_dir * 0.25
    new_q = grasp_pose.q  # 姿态保持不变
    place_pose = sapien.pysapien.Pose(new_p, new_q)
    res = planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.16]) * place_pose)
    res = planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.01]) * place_pose)
    res = planner.open_gripper()

    # -------------------------------------------------------------------------- #
    # Push the drawer to close
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.12]) * place_pose)




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

    radius = np.min(extents) / 2
    center_world = center_world + pull_dir * radius * 1.2

    T_final = np.eye(4)
    T_final[:3, :3] = axes_world
    T_final[:3, 3] = center_world

    obb = trimesh.primitives.Box(extents=extents, transform=T_final)

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
        return -1


    res = planner.move_to_pose_with_screw(sapien.Pose([0, 0, 0.15]) * grasp_pose)
    res = planner.move_to_pose_with_screw(sapien.Pose([0, 0, -0.03]) * grasp_pose)
    
    new_p = grasp_pose.p - pull_dir * 0.28
    new_q = grasp_pose.q
    new_pin_pose = sapien.pysapien.Pose(new_p, new_q)
    res = planner.move_to_pose_with_screw(new_pin_pose)


    planner.close()
    if res is None:
        return -1
    return res
