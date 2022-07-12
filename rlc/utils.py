import os
import sys
import json
import logging
from typing import Optional, Sequence, Tuple
from argparse import Namespace
from argparse import ArgumentParser

import gym
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation


def get_args(args: Optional[Sequence[str]] = None) -> Namespace:
    implemented = [
        name.split(".")[0] for name in os.listdir("config") if "engine" not in name
    ]
    parser = ArgumentParser()
    parser.add_argument("agent", help="algorithm to run", choices=implemented, type=str)
    parser.add_argument(
        "-r", "--render", help="render the environment", action="store_true"
    )
    parsed = parser.parse_args(args)

    vargs = vars(parsed)

    with open("config/engine.json", "r") as f:
        vargs.update(json.load(f))

    with open(f"config/{parsed.agent}.json", "r") as f:
        vargs.update(json.load(f))

    return Namespace(**vargs)


def setup_logging(config: Namespace) -> Tuple[str, str]:
    if not os.path.isdir("runs"):
        os.mkdir("runs")

    agent_run = os.path.join("runs", config.agent)
    if not os.path.isdir(agent_run):
        os.mkdir(agent_run)
    else:
        raise ValueError(
            f"{config.agent} has been already trained. Rename the config file to run another training"
        )

    model_path = os.path.join(agent_run, "models")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    logfile = os.path.join(agent_run, "train.log")

    logging.basicConfig(
        format="[%(asctime)s][%(name)s][%(levelname)s]\t%(message)s",
        datefmt="%m-%d %H:%M",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logfile),],
    )

    logging.info(config)

    return agent_run, model_path


def get_env(config: Namespace) -> gym.Env:
    env = gym.make(config.env_name, **config.env_kwargs)
    if config.observation_type == "image":
        env = GrayScaleObservation(env)
        env = FrameStack(env, config.frame_stack)
    return env
