import numpy as np
import torch
from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles

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

def to_numpy(data, device="cpu",):
    if isinstance(data, torch.Tensor):
        return get_numpy(data, device)
    elif isinstance(data, dict):
        return {key: to_numpy(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        try:
            return np.array([to_numpy(item, device) for item in data])
        except ValueError:
            return [to_numpy(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_numpy(item, device) for item in data)
    elif isinstance(data, bytes):
        return np.frombuffer(data, dtype=np.uint8)
    else:
        return data

def controller_delta_pose_calculate(delta_control_mode, 
                                    normalize_action, pos_scale, rot_scale,
                                    pose0, pose1, device):
    """
    Args:
        delta_control_mode: str, can be one of:
                'root_translation:root_aligned_body_rotation', # default for every agent
                'root_translation:body_aligned_body_rotation',
                'body_translation:root_aligned_body_rotation',
                'body_translation:body_aligned_body_rotation',
        normalize_action: bool, whether to normalize the delta action to [-1, 1]
        pos_scale: float, the scale to apply to the position delta (in meter)
        rot_scale: float, the scale to apply to the rotation delta (in radian)
        pose0: last pose (4x4 torch tensor). in the robot base frame.
        pose1: current pose (4x4 torch tensor). in the robot base frame.
    Returns:
        delta_xyz:  (torch.tensor, shape (3,)). in m
        delta_euler_angle:  (torch tensor, shape (3,)). in radian
    """
    # import ipdb; ipdb.set_trace()

    from scipy.spatial.transform import Rotation as R

    pose0 = pose0.to(device=device, dtype=torch.float32)
    pose1 = pose1.to(device=device, dtype=torch.float32)

    # root-frame translation delta (meters)
    delta_xyz = pose1[:3, 3] - pose0[:3, 3]

    R0 = pose0[:3, :3]
    R1 = pose1[:3, :3]

    if delta_control_mode=='root_translation:root_aligned_body_rotation':
        R_delta = R1 @ R0.transpose(0, 1) # root aligned rotation
    elif delta_control_mode=='root_translation:body_aligned_body_rotation':
        R_delta = R0.transpose(0, 1) @ R1 # rotation for body aligned rotation
    else:
        raise ValueError("""
                         Now we only support root_translation, but if you want to use
                         body_translation:body_aligned_body_rotation mode, you can use 
                         Homogeneous Transformation Matri(T).
                         The comments below are as demenstration.
                         T_last.inv() * T_now
                         """)
        # # For controller mode: body translation + body aligned rotation
        # delta_pose = (self.last_abs_action_pose.inv() * abs_action_pose)
        # delta_xyz = delta_pose.p[0]
        # delta_euler_angle = matrix_to_euler_angles(quaternion_to_matrix(delta_pose.q[0]),"XYZ")
    
    # Euler XYZ (radians) that the controller will feed into euler_angles_to_matrix(..., "XYZ")
    delta_euler = matrix_to_euler_angles(R_delta.unsqueeze(0), "XYZ").squeeze(0)

    # Norm if necessary
    if normalize_action:
        delta_xyz = delta_xyz / pos_scale
        delta_euler = delta_euler / rot_scale

        rot_norm = torch.linalg.norm(delta_euler)
        if rot_norm > 1:
            delta_euler = delta_euler / rot_norm
        
        delta_xyz = torch.clamp(delta_xyz, -1.0, 1.0)

    return (delta_xyz, delta_euler)