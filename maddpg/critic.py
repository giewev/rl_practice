import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, full_state_dim, full_action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(full_state_dim + full_action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, full_state, full_action):
        x = torch.cat([full_state, full_action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
