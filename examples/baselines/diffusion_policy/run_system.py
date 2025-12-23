from dataclasses import dataclass, field
from typing import List, Optional
import gymnasium as gym
import numpy as np
from collections import defaultdict
from diffusion_policy.make_env import make_eval_envs, make_system_envs
from diffusion_policy.evaluate import evaluate, finish_one_stage, finish_until_end
from mani_skill.examples.run_vlm_sequence import solve
from mani_skill.utils.wrappers import CPUGymWrapper
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from train_rgbd import Agent
import os
import torch
import tyro
from diffusers.training_utils import EMAModel
from PIL import Image

from mani_skill.examples.vlm_sequence.utils import PrimitiveExecutor, parse_primitives
from mani_skill.examples.vlm_sequence.gemini_request_genai import request_vlm_sequence, request_task_stage
from mani_skill.examples.vlm_sequence.prompts import vlm_sequence_prompt, check_stage_prompt
import time
import json

# TODO: timelimit wrapper
# may use own horizon cnt to control


primitive_list = ['stage_1', 'stage_2', 'stage_3', 'stage_4']
primitive_path = {
    'stage_1': '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/examples/baselines/diffusion_policy/runs/stage_1__1__1766336665/checkpoints/95000.pt',
    'stage_2': '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/examples/baselines/diffusion_policy/runs/stage_2__1__1766384578/checkpoints/90000.pt',
    'stage_3': '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/examples/baselines/diffusion_policy/runs/stage_3__1__1766384624/checkpoints/90000.pt',
    'stage_4': '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/examples/baselines/diffusion_policy/runs/stage_4__1__1766385177/checkpoints/90000.pt',
}




@dataclass
class Args:
    base_ckpt_path = '/cephfs/gyshare/ruizihang/maniskill_vlm_improve/examples/baselines/diffusion_policy/runs/base_traj__1__1766336638/checkpoints/95000.pt'
    seed: int = 1
    """seed of the experiment"""
    video_dir: Optional[str] = None
    env_id: str = "StackThree-v1"
    """the id of the environment"""
    # Environment/experiment specific arguments
    obs_mode: str = "rgb"
    """The observation mode to use for the environment, which dictates what visual inputs to pass to the model. Can be "rgb", "depth", or "rgb+depth"."""
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    num_eval_episodes: int = 2
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 1
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""
    
    obs_horizon: int = 2  # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8  # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = (
        16  # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    )
    diffusion_step_embed_dim: int = 256  # not very important
    unet_dims: List[int] = field(
        default_factory=lambda: [256, 512, 1024]
    )  # default setting is about ~4.5M params
    # diffusion_step_embed_dim: int = 64  # not very important
    # unet_dims: List[int] = field(
    #     default_factory=lambda: [64, 128, 256]
    # )  # default setting is about ~4.5M params
    n_groups: int = (
        8  # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are simila
    )


def load_ckpt(ckpt_path, agent, ema_agent):
    ckpt = torch.load(ckpt_path)
    agent.load_state_dict(ckpt["agent"])
    ema_agent.load_state_dict(ckpt["ema_agent"])

def load_all_primitive(envs, args, device):
    agents = {}
    ema_agents = {}
    for primitive in primitive_list:
        ckpt_path = primitive_path[primitive]
        agents[primitive] = Agent(envs, args).to(device)
        ema_agents[primitive] = Agent(envs, args).to(device)
        load_ckpt(ckpt_path, agents[primitive], ema_agents[primitive])
    return agents, ema_agents

def request_gemini_for_checking_stage(env, obs):

    if 'sensor_data' in obs:
        images = [Image.fromarray(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy())]
    else:
        images = [Image.fromarray(obs['rgb'][0][-1][...,:3])]

    task_desc = env.unwrapped.get_prompt_content()['task_desc']
    prompt_content = {
        'task_desc': task_desc
    }

    
    while True:
        try:
            json_response = request_task_stage(check_stage_prompt, prompt_content, images)
        except Exception as e:
            print("Error: ", e)
            time.sleep(5)
            continue
        break

    with open('temp/check_stage_gemini_response.json', 'w') as f:
        json.dump(json_response, f, indent=4)
    for i, img in enumerate(images):
        img.save(f'temp/check_stage_frame_{i}.png')

    # -1 for 0-based index
    stage_id = json_response['stage'] - 1
    return stage_id

def checking_stage_by_info(info):
    
    if not info['stage_2_success']:
        if info['stage_1_success']:
            return 1
        else:
            return 0

    if not info['stage_4_success']:
        if info['stage_3_success']:
            return 3
        else:
            return 2

    return 0



if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_eval_envs == 1, "Only support single env for now"


    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default")
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    other_kwargs = dict(obs_horizon=args.obs_horizon)
    envs = make_system_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=args.video_dir if args.video_dir is not None else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )
    temp_envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=args.video_dir if args.video_dir is not None else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )
    temp_envs.close()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agents, ema_agents = load_all_primitive(temp_envs, args, device)
    success_cnt = 0
    base_agent = Agent(temp_envs, args).to(device)
    base_ema_agent = Agent(temp_envs, args).to(device)
    load_ckpt(args.base_ckpt_path, base_agent, base_ema_agent)


    i = 0
    while True:
        if i >= args.num_eval_episodes:
            break

    # for i in range(args.num_eval_episodes):
        obs, info = envs.reset(seed=args.seed + i if args.seed is not None else None)

        print('run base policy!')
        base_success, obs, info = finish_until_end(
            agent = base_ema_agent,
            eval_envs=envs,
            last_obs=obs,
            device=device,
            sim_backend=args.sim_backend,
        )
        envs.reset_envs_elapsed_steps()

        if base_success:
            print('base policy success!!')
            i += 1
            continue
        
        print('run primitive policy!')
        # stage_id = request_gemini_for_checking_stage(envs, obs)
        print(info)
        stage_id = checking_stage_by_info(info)
        print(f'gemini response stage_id: {stage_id}')
        

        whole_success = True
        for stage_i in range(stage_id, len(primitive_list)):
            primitive = primitive_list[stage_i]
            current_stage = stage_i + 1
            stage_success, obs = finish_one_stage(
                agent=ema_agents[primitive],
                eval_envs=envs,
                last_obs=obs,
                device=device,
                sim_backend=args.sim_backend,
                current_stage=current_stage
            )
            if stage_success:
                pass
            else:
                whole_success = False
                break

        if whole_success:
            print('primitive policy success!!')
            # TODO: save
            i += 1
            continue
        envs.reset_envs_elapsed_steps()

        print('run vlm sequence!')
        code = solve(envs.unwrapped, obs, debug=False, vis=False)

        if code == "success":
            print('vlm sequence success!!')
            # TODO: save
            i += 1
            continue
        else:
            print('vlm sequence failed!!')
            i += 1
            continue


        
    

    envs.close()

    # with open(f"{args.video_dir}/success_rate.txt", "w") as f:
    #     f.write(f"Evaluated {args.num_eval_episodes} episodes")
    #     print(f"Evaluated {args.num_eval_episodes} episodes")

    #     f.write(f"Success rate: {success_cnt} / {args.num_eval_episodes}")
    #     print(f"Success rate: {success_cnt} / {args.num_eval_episodes}")
