import numpy as np
from dataclasses import dataclass
import re
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
import sapien
import trimesh
from transforms3d.quaternions import quat2mat
from transforms3d.euler import euler2quat

@dataclass
class Primitive:
    type: str               # "move" | "open" | "close"
    target_pos: np.ndarray = None
    target_quat: np.ndarray = None

def parse_primitives(primitive_list):
    parsed = []
    # last_quat = None
    for p in primitive_list:
        if p.startswith("Move to"):
            nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", p)
            vals = np.array(nums, dtype=np.float32)

            assert len(vals) == 7, f"Invalid Move primitive: {p}"

            pos = vals[:3]
            quat = vals[3:]
            if np.allclose(quat, [-1, -1, -1, -1]):
                quat = None
                # if last_quat is not None:
                #     quat = last_quat
            # else:
            #     last_quat = quat

            parsed.append(Primitive(
                type="move",
                target_pos=pos,
                target_quat=quat
            ))

        elif "Close" in p:
            parsed.append(Primitive("close"))
        elif "Open" in p:
            parsed.append(Primitive("open"))
    return parsed

class PrimitiveExecutor:

    FINGER_LENGTH = 0.025
    def __init__(self, primitives, env, planner: PandaArmMotionPlanningSolver):
        self.primitives = primitives
        self.env = env
        self.planner = planner

    def cube_to_obb(self, p, q, size):
        """
        p: (3,) center position
        q: (x, y, z, w) quaternion
        size: edge length of the cube
        """
        R_mat = quat2mat(q)
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = p

        obb = trimesh.primitives.Box(
            extents=[size, size, size],
            transform=T
        )
        return obb

    def run(self):
        res = None
        for i in range(len(self.primitives)):
            primitive = self.primitives[i]
            # print(f"Executing primitive {i}, type: {primitive.type}")
            if primitive.type == "move":

                # create obb by p,q
                if primitive.target_quat is not None:
                    obb = self.cube_to_obb(primitive.target_pos, primitive.target_quat, 0.02)
                    approaching = np.array([0, 0, -1])
                    target_closing = self.env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
                
                    grasp_info = compute_grasp_info_by_obb(
                        obb,
                        approaching=approaching,
                        target_closing=target_closing,
                        depth=self.FINGER_LENGTH,
                    )
                    closing, center = grasp_info["closing"], grasp_info["center"]
                    grasp_pose = self.env.agent.build_grasp_pose(approaching, closing, center)


                    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
                    angles = np.repeat(angles, 2)
                    angles[1::2] *= -1
                    for angle in angles:
                        delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
                        grasp_pose2 = grasp_pose * delta_pose
                        res = self.planner.move_to_pose_with_screw(grasp_pose2, dry_run=True)
                        if res == -1:
                            continue
                        grasp_pose = grasp_pose2
                        break
                    else:
                        print("Fail to find a valid grasp pose")
                    
                    target_pose = grasp_pose
        

                else:

                    now_q = self.env.agent.tcp.pose.q.cpu().numpy()[0]
                    target_pose = sapien.Pose(p=primitive.target_pos, q=now_q)
                

                res = self.planner.move_to_pose_with_screw(target_pose)

            elif primitive.type == "open":
                res = self.planner.open_gripper(t=10)
            
            elif primitive.type == "close":
                res = self.planner.close_gripper(t=10)
            
            else:
                res = None
        
        return res
    
