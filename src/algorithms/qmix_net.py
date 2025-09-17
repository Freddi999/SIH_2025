import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions):
        super(AgentQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=32):
        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

        # Hypernetworks to generate weights and biases
        self.hyper_w_1 = nn.Linear(state_dim, hidden_dim * n_agents)
        self.hyper_b_1 = nn.Linear(state_dim, hidden_dim)

        self.hyper_w_final = nn.Linear(state_dim, hidden_dim)
        self.hyper_b_final = nn.Linear(state_dim, 1)

        self.V = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        agent_qs: (batch, n_agents)
        states: (batch, state_dim)
        """
        bs = agent_qs.size(0)

        # First layer
        w1 = torch.abs(self.hyper_w_1(states))  # enforce monotonicity
        b1 = self.hyper_b_1(states)
        w1 = w1.view(bs, self.n_agents, -1)
        b1 = b1.view(bs, 1, -1)

        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # Second layer
        w_final = torch.abs(self.hyper_w_final(states)).view(bs, -1, 1)
        b_final = self.hyper_b_final(states).view(bs, 1, 1)

        y = torch.bmm(hidden, w_final) + b_final
        v = self.V(states)
        q_total = y.view(bs, 1) + v

        return q_total
