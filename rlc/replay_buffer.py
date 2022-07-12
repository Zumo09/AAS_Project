from typing import Deque
from collections import deque
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

import torch


_Exp = namedtuple("Experience", "state action reward done next_state")


@dataclass
class BatchExperience:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_states: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, sample_size: int, ready_size: int) -> None:
        self.buffer: Deque[_Exp] = deque(maxlen=capacity)
        self.sample_size = sample_size
        self.ready_size = ready_size
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self) -> bool:
        """Buffer ready to sample a batch"""
        return len(self.buffer) > self.ready_size

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        new_state: torch.Tensor,
    ) -> None:
        self.buffer.append(_Exp(state, action, reward, done, new_state))

    def sample(self) -> BatchExperience:
        idxs = self.rng.choice(len(self.buffer), self.sample_size, replace=False)
        states, actions, rewards, dones, next_states = [], [], [], [], []
        for i in idxs:
            exp = self.buffer[i]
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.done)
            next_states.append(exp.next_state)

        return BatchExperience(
            states=torch.stack(states),
            actions=torch.stack(actions),
            rewards=torch.tensor(rewards).unsqueeze(1),
            dones=torch.tensor(dones).unsqueeze(1),
            next_states=torch.stack(next_states),
        )
