import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.agents.utils import get_active_joint_indices


def transfer_qpos_2_ee_pose(
    env: BaseEnv,
    kinematics: Kinematics,
    qpos,
    world_frame: bool = False,
):
    """Convert (arm) joint positions into end-effector Pose.

    Assumes `qpos` is ordered according to `env.agent.arm_joint_names`.
    Produces a full active-joint vector (ordered like env.agent.robot.active_joints),
    then calls the updated `kinematics.compute_fk`.
    """
    qpos = torch.as_tensor(qpos).squeeze()

    # Make it batched: (B, arm_dof)
    if qpos.ndim == 1:
        qpos = qpos.unsqueeze(0)
    assert qpos.ndim == 2, f"qpos must have shape (B, dof) or (dof,), got {qpos.shape}"

    device = env.agent.robot.device
    dtype = qpos.dtype

    # Build qpos in *active joint order* (length = number of active joints)
    num_active = len(env.agent.robot.active_joints)
    qpos_fk = torch.zeros((qpos.shape[0], num_active), dtype=dtype, device=device)

    # Fill only the controlled arm joints (indices are within active_joints order)
    arm_active_idx = get_active_joint_indices(env.agent.robot, env.agent.arm_joint_names)
    qpos_fk[:, arm_active_idx] = qpos.to(device=device, dtype=dtype)

    ee_pose = kinematics.compute_fk(qpos_fk)

    if world_frame:
        # Note: If ee_pose is already in world frame, remove this multiply.
        return ee_pose * env.agent.robot.root.pose
    return ee_pose
