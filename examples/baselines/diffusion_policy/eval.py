from dataclasses import dataclass, field
from typing import List, Optional
import gymnasium as gym
import numpy as np
from collections import defaultdict
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.evaluate import evaluate
from mani_skill.utils.wrappers import CPUGymWrapper
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from train_rgbd import Agent
import os
import torch
import tyro
from diffusers.training_utils import EMAModel


@dataclass
class Args:
    ckpt_path: Optional[str] = None
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
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
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


if __name__ == "__main__":
    args = tyro.cli(Args)



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
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=args.video_dir if args.video_dir is not None else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(envs, args).to(device)
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)

    load_ckpt(args.ckpt_path, agent, ema_agent)


    eval_metrics = evaluate(
        args.num_eval_episodes, ema_agent, envs, device, args.sim_backend, seed=args.seed
    )

    print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
    for k in eval_metrics.keys():
        eval_metrics[k] = np.mean(eval_metrics[k])
        print(f"{k}: {eval_metrics[k]:.4f}")
