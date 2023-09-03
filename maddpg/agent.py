import torch
import torch.optim as optim
import torch.nn as nn

from actor import Actor
from critic import Critic


class Agent:
    def __init__(self, state_dim, action_dim, full_state_dim, full_action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(full_state_dim, full_action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(full_state_dim, full_action_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

    # Update function now takes full_states and full_actions for the critic
    def update(
        self,
        full_states,
        full_actions,
        agent_idx,
        rewards,
        full_next_states,
        dones,
        gamma=0.99,
        tau=0.01,
    ):
        # Critic update
        with torch.no_grad():
            target_next_actions = [
                self.target_actor(full_next_states[:, i, :])
                for i in range(full_next_states.shape[1])
            ]
            target_next_actions = torch.cat(target_next_actions, dim=1)
            target_next_q_values = self.target_critic(
                full_next_states.view(full_next_states.shape[0], -1),
                target_next_actions,
            )
            target_q_values = rewards + (1 - dones) * gamma * target_next_q_values

        current_q_values = self.critic(
            full_states.view(full_states.shape[0], -1),
            full_actions.view(full_actions.shape[0], -1),
        )
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        predicted_actions = [
            self.actor(full_states[:, i, :])
            if i == agent_idx
            else self.actor(full_states[:, i, :]).detach()
            for i in range(full_states.shape[1])
        ]
        predicted_actions = torch.cat(predicted_actions, dim=1)
        actor_loss = -self.critic(
            full_states.view(full_states.shape[0], -1), predicted_actions
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for param, target_param in zip(
            self.actor.parameters(), self.target_actor.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
