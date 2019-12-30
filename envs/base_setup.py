import os
import gym
import tensorflow as tf

class BaseSetup:
    gamma = 0.99
    load_checkpoint = False
    epsilon = 0.0 if load_checkpoint else 1.0
    epsilon_dec = 0.999
    epsilon_min = 0.01
    mem_size = int(1e6)
    learning_rate = 0.001
    batch_size = 32
    replace = 100
    
    _q_values_fname = 'q_values.h5'
    _q_values_target_fname = 'q_target_values.h5'
    
    
    def __init__(self, env):
        self.__check_dirs_exist()
        self.__env = env
    
    @property
    def env(self) -> gym.Env:
        """Add Gym Environment Here"""
        raise NotImplementedError("Gym environment required")

    @property
    def env(self):
        return self.__env

    @env.setter
    def env(self, env):
        self.__env = env
    
    @property
    def input_shape(self) -> tuple:
        return self.env.observation_space.shape

    @property
    def n_actions(self) -> int:
        return self.env.action_space.n
    
    @property
    def action_space(self) -> list:
        return [x for x in range(self.n_actions)]
    
    def _build_model(self) -> tf.keras.Model:
        """Add build model function"""
        raise NotImplementedError("Keras build model function required")
    
    @property
    def q_values_model(self) -> tf.keras.Model:
        return self._build_model()

    @property
    def q_values_target_model(self) -> tf.keras.Model:
        return self._build_model()
    
    @property
    def base_dir(self) -> str:
        """Add base directory where models will be saved"""
        raise NotImplementedError("Base directory for models needs to be specified")
    
    @property
    def weights_dir(self):
        return os.path.join(self.base_dir, 'weights')
    
    @property
    def q_values_fname(self):
        return os.path.join(self.weights_dir, self._q_values_fname)
    
    @property
    def q_values_target_fname(self):
        return os.path.join(self.weights_dir, self._q_values_target_fname)

    def __check_dirs_exist(self):
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)