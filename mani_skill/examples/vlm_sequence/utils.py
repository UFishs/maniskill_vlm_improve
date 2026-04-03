import numpy as np
from dataclasses import dataclass
import re

import torch
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
import sapien
import trimesh
from transforms3d.quaternions import quat2mat
from transforms3d.euler import euler2quat
from PIL import Image
import json

from mani_skill.examples.vlm_sequence.gemini_request_genai import request_fix_insert_in_action_chunk, request_vlm_sequence
from mani_skill.examples.vlm_sequence.prompts import vlm_sequence_prompt, fix_insert_action_prompt
import time
from copy import deepcopy

def request_action(env, obs, suffix=""):

    if 'sensor_data' in obs:
        images = [Image.fromarray(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy())]
    elif 'rgb' in obs:
        images = [Image.fromarray(obs['rgb'][0][-1][...,:3])]
    else:
        if len(obs['base_camera'].shape) == 4:
            image = obs['base_camera'][-1]
        elif len(obs['base_camera'].shape) == 5:
            image = obs['base_camera'][0][-1]
        else:
            image = obs['base_camera']
        image = np.array(image * 255, dtype=np.uint8).transpose(1, 2, 0)
        images = [Image.fromarray(image)]
    prompt_content = env.unwrapped.get_prompt_content()


    # # read from temp/
    # with open('temp/gemini_response.json', 'r') as f:
    #     json_response = json.load(f)


    while True:
        try:
            json_response = request_vlm_sequence(vlm_sequence_prompt, prompt_content, images)
        except Exception as e:
            print("Error: ", e)
            time.sleep(5)
            continue
        break

    # record all the info in temp/
    with open(f'temp/ground_truth{suffix}.json', 'w') as f:
        json.dump(prompt_content['ground_truth'], f, indent=4)
    with open(f'temp/gemini_response{suffix}.json', 'w') as f:
        json.dump(json_response, f, indent=4)
    # save images
    for i, img in enumerate(images):
        img.save(f'temp/frame_{i}{suffix}.png')

    primitive_list = json_response['primitives']
    parsed_primitives = parse_primitives(primitive_list)
    return parsed_primitives


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
    def __init__(self, primitives, env, planner: PandaArmMotionPlanningSolver, env_id, request_when_changing_stage=False):
        self.primitives = primitives
        self.env = env
        self.planner = planner
        self.env_id = env_id
        self.request_when_changing_stage = request_when_changing_stage

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

    def run(self, begin_obs=None):
        res = None
        current_stage = self.env.check_stage()
        # for i in range(len(self.primitives)):
        i = 0
        now_obs = begin_obs
        while True:
            primitive = self.primitives[i]
            # print(f"Executing primitive {i}, type: {primitive.type}")
            if primitive.type == "move":

                # create obb by p,q
                if self.env_id == 'StackThree-v1' or self.env_id == 'StackPyramid-v1':
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
                elif self.env_id == 'MugCleanup-v1':
                    if primitive.target_quat is not None:
                        target_pose = sapien.Pose(p=primitive.target_pos, q=primitive.target_quat)
                        # Search a valid pose
                        grasp_pose = deepcopy(target_pose)
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
                        
                        target_pose = deepcopy(grasp_pose)
                    else:
                        now_q = self.env.agent.tcp.pose.q.cpu().numpy()[0]
                        target_pose = sapien.Pose(p=primitive.target_pos, q=now_q)


                else:
                    raise NotImplementedError(self.env_id)
                

                res = self.planner.move_to_pose_with_screw(target_pose)


            elif primitive.type == "open":
                res = self.planner.open_gripper(t=6)
            
            elif primitive.type == "close":
                res = self.planner.close_gripper(t=6)
            
            else:
                res = None
            
            if res is not None and res != -1:
                now_obs, reward, terminated, truncated, info = res
                if truncated:
                    return None
            
            # import ipdb; ipdb.set_trace()
            now_stage = self.env.check_stage()  
            print(f"Finish primitive {i}, now_stage: {now_stage}")  
            i += 1
            if now_stage == self.env.stage_cnt:
                break

            if now_stage > current_stage:
                current_stage = now_stage
                print(f"Stage {current_stage} reached")
                if self.request_when_changing_stage:
                    # request new primitives
                    primitives = request_action(self.env, now_obs, suffix=f"_s{current_stage}")
                    self.primitives = primitives
                    i = 0
                
            if i >= len(self.primitives):
                break
            
        
        return res
    
def request_fix_action(env, obs, action_chunk, ee_env, return_on_world_frame=True):

    if 'sensor_data' in obs:
        image_base = Image.fromarray(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()) # (256, 256, 3)
        image_hand = Image.fromarray(obs['sensor_data']['hand_camera']['rgb'][0].cpu().numpy()) # (256, 256, 3)
    elif 'rgb' in obs:
        image_base = Image.fromarray(obs['rgb'][0][-1][...,:3]) # (256, 256, 3)
        image_hand = Image.fromarray(obs['rgb'][0][-1][...,:3]) # (256, 256, 3)

    else:
        if len(obs['base_camera'].shape) == 4:
            image_base = obs['base_camera'][-1]
            image_hand = obs['hand_camera'][-1]
        elif len(obs['base_camera'].shape) == 5:
            image_base = obs['base_camera'][0][-1]
            image_hand = obs['hand_camera'][0][-1]
        else:
            image_base = obs['base_camera']
            image_hand = obs['hand_camera']
        image_base = np.array(image_base * 255, dtype=np.uint8).transpose(1, 2, 0)
        image_hand = np.array(image_hand * 255, dtype=np.uint8).transpose(1, 2, 0)
        image_base = Image.fromarray(image_base)
        image_hand = Image.fromarray(image_hand)

    # img.save('step.png')
    images = [image_base, image_hand]
    prompt_content = env.unwrapped.get_fix_prompt_content()

    # use the root_link to get action_chunk in world frame
    key = list(ee_env.unwrapped.agent.controllers.keys())[0]
    to_base = ee_env.unwrapped.agent.controllers[key].controllers['arm'].root_link.pose.inv()
    action_chunk_world = []
    for ac in action_chunk:
        xyz = ac[:3]
        rpyg = ac[3:]

        xyz_world = xyz - to_base.p.squeeze().cpu().numpy()
        assert torch.allclose(to_base.q, torch.tensor([1.0,.0,.0,.0]))
        ac_world = np.concatenate([xyz_world, rpyg])
        action_chunk_world.append(ac_world)
    
    action_chunk_world = np.array(action_chunk_world)

    prompt_content['action_chunk'] = action_chunk_world.tolist()  # (16, 7)


    while True:
        try:
            json_response = request_fix_insert_in_action_chunk(fix_insert_action_prompt, prompt_content, images)
        except Exception as e:
            print("Error: ", e)
            time.sleep(5)
            continue
        break

    # record all the info in temp/
    with open(f'temp/fix_prompt_content.json', 'w') as f:
        json.dump(prompt_content, f, indent=4)
    with open(f'temp/fix_gemini_response.json', 'w') as f:
        json.dump(json_response, f, indent=4)
    # save images
    for i, img in enumerate(images):
        img.save(f'temp/fix_frame_{i}.png')

    fixed_action_chunk = json_response['action_chunk']  # list of [x, y, z, r, p, y, gripper]
    fixed_action_chunk = np.array(fixed_action_chunk)

    if return_on_world_frame:
        return fixed_action_chunk

    # convert back to robot base frame
    fixed_action_chunk_base = []
    for ac in fixed_action_chunk:
        xyz_world = ac[:3]
        rpyg = ac[3:]

        xyz = xyz_world + to_base.p.squeeze().cpu().numpy()
        assert torch.allclose(to_base.q, torch.tensor([1.0,.0,.0,.0]))
        ac_base = np.concatenate([xyz, rpyg])
        fixed_action_chunk_base.append(ac_base)

    fixed_action_chunk_base = np.array(fixed_action_chunk_base)
    # import ipdb; ipdb.set_trace()

    return fixed_action_chunk_base

