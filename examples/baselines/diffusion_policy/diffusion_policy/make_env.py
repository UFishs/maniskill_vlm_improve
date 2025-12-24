from typing import Optional
import gymnasium as gym
import torch
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode, PrimitiveRecord
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from gymnasium import Env
from gymnasium.spaces import Space
from gymnasium.vector.utils import concatenate, create_empty_array, iterate
from gymnasium.vector.vector_env import VectorEnv


def make_eval_envs(
    env_id,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    other_kwargs: dict,
    video_dir: Optional[str] = None,
    wrappers: list[gym.Wrapper] = [],
):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    if sim_backend == "physx_cpu":

        def cpu_make_env(
            env_id, seed, video_dir=None, env_kwargs=dict(), other_kwargs=dict()
        ):
            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir:
                    env = RecordEpisode(
                        env,
                        output_dir=video_dir,
                        save_trajectory=False,
                        info_on_video=True,
                        source_type="diffusion_policy",
                        source_desc="diffusion_policy evaluation rollout",
                    )
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk

        vector_cls = (
            gym.vector.SyncVectorEnv
            if num_envs == 1
            else lambda x: gym.vector.AsyncVectorEnv(x, context="forkserver")
        )
        env = vector_cls(
            [
                cpu_make_env(
                    env_id,
                    seed + 20222022,
                    video_dir if seed == 0 else None,
                    env_kwargs,
                    other_kwargs,
                )
                for seed in range(num_envs)
            ]
        )
    else:
        env = gym.make(
            env_id,
            num_envs=num_envs,
            sim_backend=sim_backend,
            reconfiguration_freq=1,
            **env_kwargs
        )
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
        if video_dir:
            env = RecordEpisode(
                env,
                output_dir=video_dir,
                save_trajectory=False,
                save_video=True,
                source_type="diffusion_policy",
                source_desc="diffusion_policy evaluation rollout",
                max_steps_per_video=max_episode_steps,
            )
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env


class NoAutoresetSyncVectorEnv(VectorEnv):
    """Vectorized environment that serially runs multiple environments.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.vector.SyncVectorEnv([
        ...     lambda: gym.make("Pendulum-v1", g=9.81),
        ...     lambda: gym.make("Pendulum-v1", g=1.62)
        ... ])
        >>> env.reset(seed=42)
        (array([[-0.14995256,  0.9886932 , -0.12224312],
               [ 0.5760367 ,  0.8174238 , -0.91244936]], dtype=float32), {})
    """

    def __init__(
        self,
        env_fns: Iterable[Callable[[], Env]],
        observation_space: Space = None,
        action_space: Space = None,
        copy: bool = True,
    ):
        """Vectorized environment that serially runs multiple environments.

        Args:
            env_fns: iterable of callable functions that create the environments.
            observation_space: Observation space of a single environment. If ``None``,
                then the observation space of the first environment is taken.
            action_space: Action space of a single environment. If ``None``,
                then the action space of the first environment is taken.
            copy: If ``True``, then the :meth:`reset` and :meth:`step` methods return a copy of the observations.

        Raises:
            RuntimeError: If the observation space of some sub-environment does not match observation_space
                (or, by default, the observation space of the first sub-environment).
        """
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy
        self.metadata = self.envs[0].metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        super().__init__(
            num_envs=len(self.envs),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_spaces()
        self.observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncateds = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None

    @property
    def unwrapped(self):
        return self.envs[0]


    def seed(self, seed: Optional[Union[int, Sequence[int]]] = None):
        """Sets the seed in all sub-environments.

        Args:
            seed: The seed
        """
        super().seed(seed=seed)
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        for env, single_seed in zip(self.envs, seed):
            env.seed(single_seed)

    def reset_wait(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        """Waits for the calls triggered by :meth:`reset_async` to finish and returns the results.

        Args:
            seed: The reset environment seed
            options: Option information for the environment reset

        Returns:
            The reset observation of the environment and reset information
        """
        if seed is None:
            seed = [None for _ in range(self.num_envs)]
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        assert len(seed) == self.num_envs

        self._terminateds[:] = False
        self._truncateds[:] = False
        observations = []
        infos = {}
        for i, (env, single_seed) in enumerate(zip(self.envs, seed)):
            kwargs = {}
            if single_seed is not None:
                kwargs["seed"] = single_seed
            if options is not None:
                kwargs["options"] = options

            observation, info = env.reset(**kwargs)
            observations.append(observation)
            infos = self._add_info(infos, info, i)

        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )
        return (deepcopy(self.observations) if self.copy else self.observations), infos

    def step_async(self, actions):
        """Sets :attr:`_actions` for use by the :meth:`step_wait` by converting the ``actions`` to an iterable version."""
        self._actions = iterate(self.action_space, actions)

    def step_wait(self) -> Tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        observations, infos = [], {}
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            (
                observation,
                self._rewards[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action)

            # if self._terminateds[i] or self._truncateds[i]:
            #     old_observation, old_info = observation, info
            #     observation, info = env.reset()
            #     info["final_observation"] = old_observation
            #     info["final_info"] = old_info
            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )

    def call(self, name, *args, **kwargs) -> tuple:
        """Calls the method with name and applies args and kwargs.

        Args:
            name: The method name
            *args: The method args
            **kwargs: The method kwargs

        Returns:
            Tuple of results
        """
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def set_attr(self, name: str, values: Union[list, tuple, Any]):
        """Sets an attribute of the sub-environments.

        Args:
            name: The property name to change
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise, a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            setattr(env, name, value)

    def reset_envs_elapsed_steps(self):
        base_env = self.unwrapped.unwrapped
        base_env._elapsed_steps = (
            torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.int32)
        )


    def close_extras(self, **kwargs):
        """Close the environments."""
        [env.close() for env in self.envs]

    def _check_spaces(self) -> bool:
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                raise RuntimeError(
                    "Some environments have an observation space different from "
                    f"`{self.single_observation_space}`. In order to batch observations, "
                    "the observation spaces from all environments must be equal."
                )

            if not (env.action_space == self.single_action_space):
                raise RuntimeError(
                    "Some environments have an action space different from "
                    f"`{self.single_action_space}`. In order to batch actions, the "
                    "action spaces from all environments must be equal."
                )

        return True


def make_system_envs(
    env_id,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    other_kwargs: dict,
    video_dir: Optional[str] = None,
    wrappers: list[gym.Wrapper] = [],
):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    if sim_backend == "physx_cpu":
        assert num_envs == 1, "system env only support 1 env"
        def cpu_make_env(
            env_id, seed, video_dir=None, env_kwargs=dict(), other_kwargs=dict()
        ):
            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                env = PrimitiveRecord(env)
                if video_dir:
                    env = RecordEpisode(
                        env,
                        output_dir=video_dir,
                        save_trajectory=False,
                        info_on_video=True,
                        source_type="diffusion_policy",
                        source_desc="diffusion_policy evaluation rollout",
                    )
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk

        
        vector_cls = (
            NoAutoresetSyncVectorEnv
            if num_envs == 1
            else lambda x: gym.vector.AsyncVectorEnv(x, context="forkserver")
        )
        env = vector_cls(
            [
                cpu_make_env(
                    env_id,
                    seed + 20222022,
                    video_dir if seed == 0 else None,
                    env_kwargs,
                    other_kwargs,
                )
                for seed in range(num_envs)
            ]
        )
        # env = cpu_make_env(
        #     env_id,
        #     20222022,
        #     video_dir,
        #     env_kwargs,
        #     other_kwargs,
        # )()
        
    else:
        env = gym.make(
            env_id,
            num_envs=num_envs,
            sim_backend=sim_backend,
            reconfiguration_freq=1,
            **env_kwargs
        )
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
        if video_dir:
            env = RecordEpisode(
                env,
                output_dir=video_dir,
                save_trajectory=False,
                save_video=True,
                source_type="diffusion_policy",
                source_desc="diffusion_policy evaluation rollout",
                max_steps_per_video=max_episode_steps,
            )
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env


def make_record_env(args, env_kwargs, level_name):
    record_env = gym.make(
        args.env_id,
        **env_kwargs,
    )
    record_env = RecordEpisode(
        record_env,
        output_dir=f"{args.video_dir}/iter_{args.iter_idx}/{level_name}_record",
        trajectory_name=f"iter_{args.iter_idx}_{level_name}",
        save_video=False,
        info_on_video=False,
        source_type="iteration",
        source_desc=f"iteration under {level_name}"
    )
    return record_env