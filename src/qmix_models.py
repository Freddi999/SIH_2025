# qmix_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentNetwork(nn.Module):
    """Per-agent Q-network"""
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class QMixer(nn.Module):
    """Mixes individual Qs into a joint Q"""
    def __init__(self, n_agents, state_dim, embed_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        self.hyper_w_1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_w_final = nn.Linear(state_dim, embed_dim)
        self.hyper_b_1 = nn.Linear(state_dim, embed_dim)
        self.V = nn.Sequential(nn.Linear(state_dim, embed_dim),
                               nn.ReLU(),
                               nn.Linear(embed_dim, 1))

    def forward(self, agent_qs, states):
        # agent_qs: (batch, n_agents)
        # states:   (batch, state_dim)
        bs = agent_qs.size(0)

        w1 = torch.abs(self.hyper_w_1(states))  # (batch, n_agents*embed_dim)
        b1 = self.hyper_b_1(states)             # (batch, embed_dim)
        w1 = w1.view(bs, self.n_agents, self.embed_dim)
        b1 = b1.view(bs, 1, self.embed_dim)

        hidden = torch.bmm(agent_qs.unsqueeze(1), w1).squeeze(1) + b1
        hidden = F.elu(hidden)

        w_final = torch.abs(self.hyper_w_final(states))  # (batch, embed_dim)
        w_final = w_final.view(bs, self.embed_dim, 1)
        v = self.V(states)  # (batch, 1)

        y = torch.bmm(hidden.unsqueeze(1), w_final).squeeze(1) + v
        return y  # (batch, 1)


class RegressorNetwork(nn.Module):
    """Extra regressor to predict waiting time (optional enhancement)"""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
