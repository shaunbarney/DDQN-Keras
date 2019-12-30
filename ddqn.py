import numpy as np
import tensorflow as tf
from envs.base_setup import BaseSetup

from replay_memory import ReplayMemory

class Agent:
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], BaseSetup):
            s = args[0]
            self.gamma = s.gamma
            self.epsilon = s.epsilon
            self.esp_dec = s.epsilon_dec
            self.eps_min = s.epsilon_min
            self.batch_size = s.batch_size
            self.replace = s.replace
            self.q_values_fname = s.q_values_fname
            self.q_values_target_fname = s.q_values_target_fname
            self.action_space = s.action_space
            
            self.memory = ReplayMemory(s.mem_size, s.input_shape)
            self.q_values = s.q_values_model
            self.q_values_target = s.q_values_target_model
        else:
            self.gamma = kwargs['gamma'] 
            self.epsilon = kwargs['epsilon']
            self.esp_dec = kwargs['eps_dec']
            self.eps_min = kwargs['eps_min']
            self.batch_size = kwargs['batch_size']
            self.replace = kwargs['replace']
            self.q_values_fname = kwargs['q_values_fname']
            self.q_values_target_fname = kwargs['q_values_target_fname']
            
            self.memory = ReplayMemory(kwargs['mem_size'], kwargs['input_shape'])
            self.q_values = kwargs['q_values_model']
            self.q_values_target = kwargs['q_values_target_model']
            self.action_space = [x for x in range(kwargs['n_actions'])]

        self.learn_step = 0

    def replace_target_network(self):
        if self.replace is not None and self.learn_step % self.replace == 0:
            self.q_values_target.set_weights(self.q_values.get_weights())
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, done, new_state)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array(np.atleast_2d(state), dtype=np.float32, copy=False)
            q_values = self.q_values.predict(state)
            action = np.argmax(q_values)

        return action
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.replace_target_network()

        state, action, reward, new_state, terminal = self.memory.sample_memory(self.batch_size)

        q_vals_target = self.q_values_target.predict(new_state)
        q_vals = self.q_values.predict(state)
        
        q_vals_y = q_vals[:]
        
        idxs = [x for x in range(self.batch_size)]
        q_vals_y[idxs, action] = reward + self.gamma * np.max(q_vals_target, axis=1) * terminal
        
        self.q_values.train_on_batch(state, q_vals_y)

        self.epsilon = self.epsilon * self.esp_dec if self.epsilon > self.eps_min else self.eps_min
        self.learn_step += 1
    
    def save_models(self):
        print("Saving models.")
        self.q_values.save(self.q_values_fname)
        self.q_values_target.save(self.q_values_target_fname)

    def load_models(self):
        print("Loading models.")
        self.q_values = tf.keras.model.load_models(self.q_values_fname)
        self.q_values_target = tf.keras.model.load_models(self.q_values_target)
        

        