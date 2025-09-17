# replay_buffer.py
import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, state_dim, n_agents, n_actions, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def store(self, obs, state, actions, reward, next_obs, next_state, done):
        idx = self.ptr
        self.obs[idx] = obs
        self.state[idx] = state
        self.actions[idx] = actions
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.next_state[idx] = next_state
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.tensor(self.obs[idxs], device=self.device),
            torch.tensor(self.state[idxs], device=self.device),
            torch.tensor(self.actions[idxs], device=self.device),
            torch.tensor(self.rewards[idxs], device=self.device),
            torch.tensor(self.next_obs[idxs], device=self.device),
            torch.tensor(self.next_state[idxs], device=self.device),
            torch.tensor(self.dones[idxs], device=self.device),
        )

    def __len__(self):
        return self.size
