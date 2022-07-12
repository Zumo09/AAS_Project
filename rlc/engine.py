import logging
from typing import Dict
from typing import Tuple
from typing import Optional
from typing import Protocol

import gym
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter

from .replay_buffer import ReplayBuffer
from .replay_buffer import BatchExperience


class Agent(Protocol):
    def reset(self) -> None:
        """Get ready for a new episode"""

    def act(self, state: Tensor, *, training: bool = False) -> Tensor:
        """Get the action in the current state"""
        raise NotImplementedError()

    def learn(self, batch: BatchExperience) -> Dict[str, float]:
        """Train the model for one step. Return dict of metrics to log"""
        raise NotImplementedError()

    def save(self, base_path: str, identifier: str) -> None:
        """Save the models"""

    def load(self, checkpoint: str) -> None:
        """Load a pretrained model"""


logger = logging.getLogger(__name__)


def test_episode(
    env: gym.Env, agent: Agent, *, render: bool = False, max_episode_steps: int = 1000,
) -> float:
    obs = env.reset()
    agent.reset()
    state = torch.tensor(obs)
    episode_reward = 0
    done = False

    if render:
        env.render()
    with tqdm(leave=False, desc="Running Episode", total=max_episode_steps) as pbar:
        step = 0
        while not done:
            action = agent.act(state, training=False)
            obs, reward, done, _ = env.step(tuple(float(a) for a in action))
            episode_reward += reward
            new_state = torch.tensor(obs)
            if render:
                env.render()
            state = new_state

            step += 1
            if step > max_episode_steps:
                done = True
            pbar.update()

    return episode_reward


def train_episode(
    env: gym.Env,
    agent: Agent,
    buffer: ReplayBuffer,
    *,
    update_every: int,
    render: bool = False,
    max_episode_steps: int = 1000,
    writer: Optional[SummaryWriter] = None,
    train_step: int = 0,
) -> Tuple[float, int]:
    obs = env.reset()
    agent.reset()
    state = torch.tensor(obs)
    episode_reward = 0
    done = False

    if render:
        env.render()
    with tqdm(leave=False, desc="Running Episode", total=max_episode_steps) as pbar:
        step = 0
        while not done:
            if not buffer.is_ready():
                # Start randomly
                action = torch.tensor(env.action_space.sample())
            else:
                action = agent.act(state, training=True)

            obs, reward, done, _ = env.step(tuple(float(a) for a in action))

            episode_reward += reward
            new_state = torch.tensor(obs)

            buffer.add(state, action, reward, done, new_state)
            if step % update_every == 0 and buffer.is_ready():
                logs = agent.learn(buffer.sample())
                if writer is not None:
                    for tag, value in logs.items():
                        writer.add_scalar(tag, value, train_step)
                    train_step += 1

            if render:
                env.render()
            state = new_state

            step += 1
            if step > max_episode_steps:
                done = True
            pbar.update()

    return episode_reward, train_step


def train(
    env: gym.Env,
    agent: Agent,
    buffer: ReplayBuffer,
    n_episodes: int,
    *,
    model_base_path: str,
    n_test_episodes: int = 1,
    checkpoint_interval: int = 10,
    update_every: int = 4,
    render: bool = False,
    max_episode_steps: int = 1000,
    writer: Optional[SummaryWriter] = None,
) -> None:
    train_step = 0
    best_test_reward = 0
    running_mean = 0
    alpha = 0.2
    for episode in range(1, n_episodes + 1):
        reward, train_step = train_episode(
            env,
            agent,
            buffer,
            update_every=update_every,
            render=False,
            max_episode_steps=max_episode_steps,
            writer=writer,
            train_step=train_step,
        )
        running_mean = alpha * reward + (1 - alpha) * running_mean
        if episode % 10 == 0:
            logger.info(
                f"Episode {episode}: {reward:7.2f} (mean: {running_mean:7.2f}) [steps {train_step}]"
            )

        if writer is not None:
            writer.add_scalar("Train_Reward", reward, episode)

        if episode % checkpoint_interval == 0:
            agent.save(model_base_path, f"{episode:06d}")

            test_reward = test(
                env,
                agent,
                n_test_episodes,
                render=render,
                max_episode_steps=max_episode_steps,
            )
            logger.info(f"Test {episode}: {test_reward:.2f}")

            if writer is not None:
                writer.add_scalar("Test_Reward", test_reward, episode)

            if test_reward >= best_test_reward:
                best_test_reward = test_reward
                agent.save(model_base_path, "best")


def test(
    env: gym.Env,
    agent: Agent,
    n_episodes: int,
    *,
    render: bool = False,
    max_episode_steps: int = 1000,
) -> float:
    reward = 0
    for episode in range(n_episodes):
        ep_r = test_episode(
            env, agent, render=render, max_episode_steps=max_episode_steps
        )
        logger.info(f"Episode {episode}: {ep_r:7.2f}")
        reward += ep_r

    reward /= n_episodes
    return reward
