import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from replay_buffer import ReplayBuffer
from algorithms.qmix_net import AgentQNetwork, MixingNetwork


class QMIXTrainer:
    def __init__(self,env, n_agents, state_dim, obs_dim, n_actions,
                 hidden_dim=64, lr=0.0005, gamma=0.99,
                 buffer_size=5000, batch_size=32, device="cpu"):

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.replay_buffer = ReplayBuffer(
    capacity=buffer_size,
    obs_dim=obs_dim,
    state_dim=state_dim,
    n_agents=n_agents,
    n_actions=n_actions,
    device=self.device
)

        # Individual agent networks (shared parameters for all agents)
        self.agent_q = AgentQNetwork(obs_dim, hidden_dim, n_actions).to(self.device)
        self.target_agent_q = AgentQNetwork(obs_dim, hidden_dim, n_actions).to(self.device)
        self.target_agent_q.load_state_dict(self.agent_q.state_dict())

        # Mixing network
        self.mixing_net = MixingNetwork(n_agents, state_dim, hidden_dim).to(self.device)
        self.target_mixing_net = MixingNetwork(n_agents, state_dim, hidden_dim).to(self.device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(list(self.agent_q.parameters()) +
                                    list(self.mixing_net.parameters()), lr=lr)

        # Replay buffer
        self.buffer = deque(maxlen=buffer_size)

        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995  # slower decay


    def select_action(self, obs):
        """epsilon-greedy action selection for all agents"""
        actions = []
        for agent_obs in obs:
            if np.random.rand() < self.epsilon:
                actions.append(np.random.randint(self.n_actions))
            else:
                obs_tensor = torch.FloatTensor(agent_obs).unsqueeze(0).to(self.device)
                q_values = self.agent_q(obs_tensor)
                actions.append(torch.argmax(q_values).item())
        return actions

    def store_transition(self, state, obs, actions, reward, next_state, next_obs, done):
        self.buffer.append((state, obs, actions, reward, next_state, next_obs, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        # Sample minibatch
        minibatch = random.sample(self.buffer, self.batch_size)
        states, obs, actions, rewards, next_states, next_obs, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        obs = torch.FloatTensor(obs).to(self.device)          # (batch, n_agents, obs_dim)
        actions = torch.LongTensor(actions).to(self.device)  # (batch, n_agents)
        rewards = torch.FloatTensor(rewards).to(self.device) # (batch,)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        batch_size = len(minibatch)

        # Compute current Q-values
        agent_qs = []
        for i in range(self.n_agents):
            q_values = self.agent_q(obs[:, i, :])
            q_selected = q_values.gather(1, actions[:, i].unsqueeze(1)).squeeze(1)
            agent_qs.append(q_selected)
        agent_qs = torch.stack(agent_qs, dim=1)  # (batch, n_agents)

        q_total = self.mixing_net(agent_qs, states)

        # Compute target Q-values
        next_agent_qs = []
        for i in range(self.n_agents):
            q_values = self.target_agent_q(next_obs[:, i, :])
            q_selected = q_values.max(1)[0]
            next_agent_qs.append(q_selected)
        next_agent_qs = torch.stack(next_agent_qs, dim=1)

        target_q_total = self.target_mixing_net(next_agent_qs, next_states)
        targets = rewards + self.gamma * (1 - dones) * target_q_total.squeeze(1)

        # Loss
        loss = nn.MSELoss()(q_total.squeeze(1), targets.detach())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.agent_q.parameters()) +
                                       list(self.mixing_net.parameters()), 10)
        self.optimizer.step()

        # Epsilon decay
       
    def update_target_networks(self):
        self.target_agent_q.load_state_dict(self.agent_q.state_dict())
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())
