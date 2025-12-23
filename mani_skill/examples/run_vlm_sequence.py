import argparse
from ast import parse
from typing import Annotated
import gymnasium as gym
import numpy as np
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
import sapien.utils.viewer
import h5py
import json
import mani_skill.trajectory.utils as trajectory_utils
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.record import RecordEpisode
import tyro
from dataclasses import dataclass, field

from mani_skill.examples.vlm_sequence.utils import PrimitiveExecutor, parse_primitives
from mani_skill.examples.vlm_sequence.gemini_request_genai import request_vlm_sequence
from mani_skill.examples.vlm_sequence.prompts import vlm_sequence_prompt

from PIL import Image
import time

import tqdm

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "StackThree-v1"
    obs_mode: str = "rgb"
    record_dir: str = "demos"
    control_mode: str = "pd_joint_delta_pos"
    """directory to record the demonstration data and optionally videos"""
    save_video: bool = False
    """whether to save the videos of the demonstrations after collecting them all"""
    sensor_configs: dict = field(
        default_factory=lambda: {
            "base_camera": {
                "width": 256,
                "height": 256,
            },
            "hand_camera": {
                "width": 256,
                "height": 256,
            },
        }
    )
    must_success: bool = False
    num_rollout: int = 10


def parse_args() -> Args:
    return tyro.cli(Args)

def main(args: Args):
    output_dir = f"{args.record_dir}/{args.env_id}/vlm_sequence/"
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        reward_mode="none",
        render_mode="rgb_array",
    )
    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
        source_type="vlm_sequence",
        source_desc="planning with vlm_sequence"
    )
    seed = 0
    obs, info = env.reset(seed=seed)
    pbar = tqdm.tqdm(total=args.num_rollout)


    attempts = 0
    successes = 0

    while True:
        code = solve(env, obs, debug=False, vis=False)
        attempts += 1

        if code == "success":
            successes += 1
            seed += 1
            obs, info = env.reset(seed=seed)
            pbar.update(1)

        else:
            seed += 1
            obs, info = env.reset(seed=seed, options=dict(save_trajectory=False))

        pbar.set_postfix(
            attempts=attempts,
            success_rate=f"{successes/attempts:.3f}"
        )
        print(f"Current success rate: {successes}/{attempts}")
        with open(f'{output_dir}/success_rate.txt', 'a') as f:
            f.write(f"{successes}/{attempts}\n")

        if args.must_success:
            if successes >= args.num_rollout:
                break
        else:
            if attempts >= args.num_rollout:
                break

    pbar.close()


    h5_file_path = env._h5_file.filename
    json_file_path = env._json_path
    env.close()
    del env
    print(f"Trajectories saved to {h5_file_path}")



    if args.save_video:
        print(f"Saving videos to {output_dir}")

        trajectory_data = h5py.File(h5_file_path)
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
        env = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            control_mode=args.control_mode,
            reward_mode="none",
            render_mode="rgb_array",
        )
        env = RecordEpisode(
            env,
            output_dir=output_dir,
            trajectory_name="trajectory",
            save_video=True,
            info_on_video=False,
            save_trajectory=False,
            video_fps=30
        )
        for episode in json_data["episodes"]:
            traj_id = f"traj_{episode['episode_id']}"
            data = trajectory_data[traj_id]
            env.reset(**episode["reset_kwargs"])
            env_states_list = trajectory_utils.dict_to_list_of_dicts(data["env_states"])

            env.base_env.set_state_dict(env_states_list[0])
            for action in np.array(data["actions"]):
                env.step(action)

        trajectory_data.close()
        env.close()
        del env

def request_action(env, obs):

    images = [Image.fromarray(obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy())]
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
    with open('temp/ground_truth.json', 'w') as f:
        json.dump(prompt_content['ground_truth'], f, indent=4)
    with open('temp/gemini_response.json', 'w') as f:
        json.dump(json_response, f, indent=4)
    # save images
    for i, img in enumerate(images):
        img.save(f'temp/frame_{i}.png')

    primitive_list = json_response['primitives']
    parsed_primitives = parse_primitives(primitive_list)
    return parsed_primitives


def solve(env: BaseEnv, obs, debug=False, vis=False):
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_delta_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    
    robot_has_gripper = True
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_acc_limits=0.5,
        joint_vel_limits=0.5,
    )

    primitives = request_action(env, obs)
    executor = PrimitiveExecutor(primitives, env, planner)

    res = executor.run()
    if res is not None and res != -1:
        now_obs, reward, terminated, truncated, info = res

        if info['success']:
            return "success"
        else:
            return "fail"

    else:
        return "fail"




if __name__ == "__main__":
    main(parse_args())
