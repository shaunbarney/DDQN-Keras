import gym
from envs.cartpole.model import build_model
from envs.base_setup import BaseSetup


class CartPoleSetup(BaseSetup):
    def __init__(self, env: gym.Env):
        super().__init__()
        self.__env = env
        
    @property
    def env(self):
        return self.__env

    @env.setter
    def env(self, env):
        self.__env = env
    
    def _build_model(self):
        return build_model(self.learning_rate, self.n_actions, self.input_shape)
    
    @property
    def base_dir(self):
        return '/Users/shaunbarney/Documents/ReinforcementLearning/DDQN/envs/cartpole'

