import gymnasium as gym

class PrimitiveRecord(gym.Wrapper):

    def __init__(self, env, base_record_env=None, primitive_record_envs=None):
        super().__init__(env)

        self.env = env
        self.base_record_env = base_record_env
        self.primitive_record_envs = primitive_record_envs
        self.primitive_list = ['stage_1', 'stage_2', 'stage_3', 'stage_4']
        self.current_stage = 0
        self.record = False
        self.last_stage = None
        self.last_cnt = -1
        self.required_cnt = 10
    
    def set_record_env(self, base_record_env, primitive_record_envs):
        self.base_record_env = base_record_env
        self.primitive_record_envs = primitive_record_envs

    def set_current_stage(self, stage):
        self.current_stage = stage
    
    def enable_record(self):
        self.record = True

    def disable_record(self):
        self.record = False

    def step(self, action):

        obs, rew, terminated, truncated, info = super().step(action)

        if self.record:
            assert self.base_record_env is not None and self.primitive_record_envs is not None

            self.base_record_env.step(action)
            if self.current_stage < len(self.primitive_list):
                self.primitive_record_envs[self.primitive_list[self.current_stage]].step(action)
            
            if self.last_stage is not None and self.last_cnt != -1:
                self.primitive_record_envs[self.last_stage].step(action)
                self.last_cnt -= 1
            
            if self.last_stage is not None and self.last_cnt == 0:
                self.last_stage = None
                self.last_cnt = -1

            if self.current_stage < len(self.primitive_list) and info[f'{self.primitive_list[self.current_stage]}_success']:

                self.last_stage = self.primitive_list[self.current_stage]
                self.last_cnt = self.required_cnt

                self.current_stage += 1
                if self.current_stage < len(self.primitive_list):
                    # print(f"change from {self.primitive_list[self.current_stage - 1]} to {self.primitive_list[self.current_stage]}")
                    self.primitive_record_envs[self.primitive_list[self.current_stage]].reset(
                        options={
                            'reset_to_env_states': {
                                'env_states': self.env.unwrapped.get_state_dict(),
                            }
                        }
                    )

        return obs, rew, terminated, truncated, info

    
    @property
    def base_env(self):
        return self.env.unwrapped