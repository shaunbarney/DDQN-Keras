import numpy as np


class ReplayMemory:
    def __init__(self, mem_size, input_shape):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.states = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.states_ = np.zeros(
            (self.mem_size, *input_shape), dtype=np.float32)
        self.actions = np.zeros(self.mem_size, dtype=np.int32)
        self.rewards = np.zeros(self.mem_size, dtype=np.int32)
        self.terminals = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, done, state_):
        index = self.mem_cntr % self.mem_size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.terminals[index] = int(1 - done)
        self.states_[index] = state_
        self.mem_cntr += 1

    def sample_memory(self, batch_size=32):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch_idxs = np.random.choice(max_mem, batch_size, replace=False)

        states = self.states[batch_idxs]
        actions = self.actions[batch_idxs]
        rewards = self.rewards[batch_idxs]
        terminals = self.terminals[batch_idxs]
        states_ = self.states_[batch_idxs]

        return states, actions, rewards, states_, terminals
