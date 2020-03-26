import gym
from envs.cartpole.model import build_model
from envs.base_setup import BaseSetup


class CartPoleSetup(BaseSetup):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.learning_rate = 0.005
        self.mem_size = int(1e4)
        
    def _build_model(self):
        return build_model(self.learning_rate, self.n_actions, self.input_shape)
    
    @property
    def base_dir(self):
        return '/home/shaun/Documents/ReinforcementLearning/Gym/DDQN/envs/cartpole'

