import gymnasium as gym
import mani_skill.envs
import ipdb
from PIL import Image

env = gym.make(
    "MugCleanup-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="rgb", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    render_mode="rgb_array"
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0) # reset with a seed for determinism
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # import ipdb; ipdb.set_trace()
    img = env.render()[0].cpu().numpy()
    # img = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
    img = Image.fromarray(img)
    img.save("step.png")
    import ipdb; ipdb.set_trace()

    done = terminated or truncated
    env.render()  # a display is required to render
env.close()