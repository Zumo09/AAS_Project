import logging
import itertools
from typing import Dict
from argparse import Namespace

import torch
from torch import nn
from torch import optim
from torch import Tensor

from . import models
from .noise import OrnsteinUhlenbeckProcess

from rlc.agents import utils
from rlc.replay_buffer import BatchExperience


logger = logging.getLogger(__name__)


class DDPG(nn.Module):
    def __init__(self, config: Namespace, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.soft_update: float = config.soft_update
        self.discount_factor: float = config.discount_factor

        if config.observation_type == "image":
            actor = models.ActorCNN
            critic = models.CriticCNN
        else:
            actor = models.ActorMLP
            critic = models.CriticMLP

        # Build policy and value functions
        self.actor = actor(config)
        self.critic = critic(config)

        # Build policy and value target functions
        self.actor_t = actor(config)
        self.critic_t = critic(config)

        # Build optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.actor_lr, weight_decay=config.actor_wd,
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.critic_wd,
        )

        self.mse = nn.MSELoss()

        # Build temporally correlated noise
        self.noise = OrnsteinUhlenbeckProcess(
            action_dimension=config.action_size,
            theta=config.noise_theta,
            sigma=config.noise_sigma,
            mu=config.noise_mu,
        )

        self.float()
        self.to(device)

        # Load weights
        if config.checkpoint is None:
            # Copy local weights to targets
            self.actor_t.load_state_dict(self.actor.state_dict())
            self.critic_t.load_state_dict(self.critic.state_dict())
        else:
            self.load(config.checkpoint)

        # Freeze target networks
        for p in self.actor_t.parameters():
            p.requires_grad = False
        for p in self.critic_t.parameters():
            p.requires_grad = False

    def reset(self) -> None:
        """Get ready for a new episode"""
        self.noise.reset()

    @torch.no_grad()
    def act(self, state: Tensor, *, training: bool = False) -> Tensor:
        """Get the action in the current state"""
        state = state.unsqueeze(0).to(self.device).float()
        self.eval()
        actions: Tensor = self.actor(state)
        actions = actions.squeeze(0).cpu()
        if training:
            # Add noise
            actions += self.noise.step()
            actions = torch.clamp(actions, -1, 1)

        return actions

    def learn(self, batch: BatchExperience) -> Dict[str, float]:
        """Train the model for one step. Return dict of metrics to log"""
        self.train()
        states = batch.states.to(self.device).float()
        actions = batch.actions.to(self.device).float()
        rewards = batch.rewards.to(self.device).float()
        next_states = batch.next_states.to(self.device).float()
        dones = batch.dones.to(self.device).bool()

        loss_q = self._update_critic(states, actions, rewards, dones, next_states)
        loss_a = self._update_actor(states)

        utils.soft_update(self.actor_t, self.actor, self.soft_update)
        utils.soft_update(self.critic_t, self.critic, self.soft_update)

        return {
            "critic_loss": float(loss_q.item()),
            "actor_loss": float(loss_a.item()),
        }

    def _update_actor(self, states: Tensor) -> Tensor:
        # Freeze critic
        for p in self.critic.parameters():
            p.requires_grad = False

        self.actor_optimizer.zero_grad()
        local_act: Tensor = self.actor(states)
        pseudo: Tensor = self.critic(states, local_act)
        loss_a = -pseudo.mean()
        loss_a.backward()
        self.actor_optimizer.step()

        # Unfreeze Critic
        for p in self.critic.parameters():
            p.requires_grad = True
        return loss_a

    def _update_critic(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        next_states: Tensor,
    ) -> Tensor:
        self.critic_optimizer.zero_grad()
        # set target y = r_i + gamma * C'(s_i+1, A'(s_i+1))
        with torch.no_grad():
            target_act: Tensor = self.actor_t(next_states)
            target_q: Tensor = self.critic_t(next_states, target_act)
            target_q = rewards + ~(dones) * self.discount_factor * target_q

        local_values: Tensor = self.critic(states, actions)
        loss_q: Tensor = self.mse(local_values, target_q)
        loss_q.backward()
        self.critic_optimizer.step()
        return loss_q

    def save(self, base_path: str, identifier: str) -> None:
        """Save the models"""
        utils.save_model(self, f"{base_path}/ddpg_agent_{identifier}.pth")

    def load(self, checkpoint: str) -> None:
        """Load a pretrained model"""
        utils.load_model(self, checkpoint, self.device)


class DDPGDC(nn.Module):
    def __init__(self, config: Namespace, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.soft_update: float = config.soft_update
        self.discount_factor: float = config.discount_factor

        if config.observation_type == "image":
            actor = models.ActorCNN
            critic = models.CriticCNN
        else:
            actor = models.ActorMLP
            critic = models.CriticMLP

        # Build policy and value functions
        self.actor = actor(config)
        self.crit1 = critic(config)
        self.crit2 = critic(config)

        # Build target policy and value functions
        self.actor_t = actor(config)
        self.crit1_t = critic(config)
        self.crit2_t = critic(config)

        # Build optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.actor_lr, weight_decay=config.actor_wd,
        )

        self.critics_params = itertools.chain(
            self.crit1.parameters(), self.crit2.parameters()
        )
        self.critic_optimizer = optim.Adam(
            self.critics_params, lr=config.critic_lr, weight_decay=config.critic_wd,
        )

        self.mse = nn.MSELoss()

        # Build temporally correlated noise
        self.noise = OrnsteinUhlenbeckProcess(
            action_dimension=config.action_size,
            theta=config.noise_theta,
            sigma=config.noise_sigma,
            mu=config.noise_mu,
        )

        self.float()
        self.to(device)

        # Load weights
        if config.checkpoint is None:
            # Copy local weights to targets
            self.crit1_t.load_state_dict(self.crit1.state_dict())
            self.crit2_t.load_state_dict(self.crit2.state_dict())
        else:
            self.load(config.checkpoint)

        # Freeze target networks
        for p in self.actor_t.parameters():
            p.requires_grad = False
        for p in self.crit1_t.parameters():
            p.requires_grad = False
        for p in self.crit2_t.parameters():
            p.requires_grad = False

    def reset(self) -> None:
        """Get ready for a new episode"""
        self.noise.reset()

    @torch.no_grad()
    def act(self, state: Tensor, *, training: bool = False) -> Tensor:
        """Get the action in the current state"""
        state = state.unsqueeze(0).to(self.device).float()
        self.eval()
        actions: Tensor = self.actor(state)
        actions = actions.squeeze(0).cpu()
        if training:
            # Add noise
            actions += self.noise.step()
            actions = torch.clamp(actions, -1, 1)

        return actions

    def learn(self, batch: BatchExperience) -> Dict[str, float]:
        """Train the model for one step. Return dict of metrics to log"""
        self.train()
        states = batch.states.to(self.device).float()
        actions = batch.actions.to(self.device).float()
        rewards = batch.rewards.to(self.device).float()
        next_states = batch.next_states.to(self.device).float()
        dones = batch.dones.to(self.device).bool()

        loss_q = self._update_critic(states, actions, rewards, dones, next_states)
        loss_a = self._update_actor(states)

        utils.soft_update(self.actor_t, self.actor, self.soft_update)
        utils.soft_update(self.crit1_t, self.crit1, self.soft_update)
        utils.soft_update(self.crit2_t, self.crit2, self.soft_update)

        return {
            "critic_loss": float(loss_q.item()),
            "actor_loss": float(loss_a.item()),
        }

    def _update_actor(self, states: Tensor) -> Tensor:
        # Freeze critic
        for p in self.critics_params:
            p.requires_grad = False

        self.actor_optimizer.zero_grad()
        local_act: Tensor = self.actor(states)
        pseudo1: Tensor = self.crit1(states, local_act)
        pseudo2: Tensor = self.crit2(states, local_act)
        pseudo = torch.min(pseudo1, pseudo2)
        loss_a = -pseudo.mean()
        loss_a.backward()
        self.actor_optimizer.step()

        # Unfreeze Critic
        for p in self.critics_params:
            p.requires_grad = True
        return loss_a

    def _update_critic(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        next_states: Tensor,
    ) -> Tensor:
        self.critic_optimizer.zero_grad()
        # set target y = r_i + gamma * min C'(s_i+1, A'(s_i+1))
        with torch.no_grad():
            target_act: Tensor = self.actor_t(next_states)

            target_q1 = self.crit1_t(next_states, target_act)
            target_q2 = self.crit2_t(next_states, target_act)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + ~(dones) * self.discount_factor * target_q

        local_values_1: Tensor = self.crit1(states, actions)
        local_values_2: Tensor = self.crit2(states, actions)

        loss_q1: Tensor = self.mse(local_values_1, target_q)
        loss_q2: Tensor = self.mse(local_values_2, target_q)
        loss_q = loss_q1 + loss_q2

        loss_q.backward()
        self.critic_optimizer.step()
        return loss_q

    def save(self, base_path: str, identifier: str) -> None:
        """Save the models"""
        utils.save_model(self, f"{base_path}/ddpg_agent_{identifier}.pth")

    def load(self, checkpoint: str) -> None:
        """Load a pretrained model"""
        utils.load_model(self, checkpoint, self.device)
