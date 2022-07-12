from typing import Tuple
from argparse import Namespace

import torch

from rlc.agents import common


class _Actor(torch.nn.Module):
    def __init__(self, config: Namespace) -> None:
        super().__init__()
        self.feats = common.MLP(1, [], 1)
        self.mean = torch.nn.Linear(config.actor_feat_size, config.action_size)
        self.log_std = torch.nn.Linear(config.actor_feat_size, config.action_size)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.feats(state)
        mu = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -10, 10)
        std = torch.exp(log_std)
        return mu, std


class ActorCNN(_Actor):
    def __init__(self, config: Namespace) -> None:
        super().__init__(config)
        self.feats = common.FeatureExtractor(
            config.observation_channels,
            config.actor_conv_hiddens,
            config.actor_feat_size,
            last_activation=torch.nn.ReLU,
        )


class ActorMLP(_Actor):
    def __init__(self, config: Namespace) -> None:
        super().__init__(config)
        self.feats = common.MLP(
            config.observation_size,
            config.actor_hiddens,
            config.actor_feat_size,
            last_activation=torch.nn.ReLU,
        )


class _Critic(torch.nn.Module):
    def __init__(self, config: Namespace) -> None:
        super().__init__()
        self.feats = common.MLP(1, [], 1)
        s = config.critic_feat_size + config.action_size
        self.mlp = common.MLP(s, config.critic_mlp_hiddens, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        s = self.feats(state)
        x = torch.cat([s, action], dim=-1)
        x = self.mlp(x)
        return x


class CriticCNN(_Critic):
    def __init__(self, config: Namespace) -> None:
        super().__init__(config)
        self.feats = common.FeatureExtractor(
            config.observation_channels,
            config.critic_feat_hiddens,
            config.critic_feat_size,
            last_activation=torch.nn.Tanh,
        )


class CriticMLP(_Critic):
    def __init__(self, config: Namespace) -> None:
        super().__init__(config)
        self.feats = common.MLP(
            config.observation_size,
            config.critic_feat_hiddens,
            config.critic_feat_size,
            last_activation=torch.nn.Tanh,
        )
