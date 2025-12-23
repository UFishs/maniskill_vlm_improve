from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from mani_skill.utils import common

def evaluate(n: int, agent, eval_envs, device, sim_backend: str, progress_bar: bool = True, seed=None):
    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset(seed=seed)
        eps_count = 0
        while eps_count < n:
            obs = common.to_tensor(obs, device)

            # import ipdb; ipdb.set_trace()
            # from PIL import Image
            # Image.fromarray(obs['rgb'][0,-1,...].cpu().numpy()[...,:3]).save("base.png")
            # Image.fromarray(obs['rgb'][0,-1,...].cpu().numpy()[...,3:]).save("hand.png")



            action_seq = agent.get_action(obs)
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break

            if truncated.any():
                assert truncated.all() == truncated.any(), "all episodes should truncate at the same time for fair evaluation with other algorithms"
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                eps_count += eval_envs.num_envs
                if progress_bar:
                    pbar.update(eval_envs.num_envs)

                obs, info = eval_envs.reset(seed=seed+eps_count if seed is not None else None)
    agent.train()
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics


def finish_one_stage(agent, eval_envs, last_obs, device, sim_backend: str, current_stage, cnt_down = 10):
    agent.eval()
    
    with torch.no_grad():
        obs = last_obs
        stage_success = False
        current_cnt = -1
        while True:
            obs = common.to_tensor(obs, device)

            action_seq = agent.get_action(obs)
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                
                if current_cnt == -1 and info[f'stage_{current_stage}_success']:
                    current_cnt = cnt_down
                
                if current_cnt > 0:
                    current_cnt -= 1
                elif current_cnt == 0:
                    stage_success = info[f'stage_{current_stage}_success']
                    break

                if truncated.any():
                    stage_success = info[f'stage_{current_stage}_success']
                    break

            if truncated.any() or stage_success:
                break
            

    agent.train()
    return stage_success, obs
