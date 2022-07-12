from argparse import Namespace

from torch import device

from .sac.agent import SAC
from .ddpg.agent import DDPG
from .ddpg.agent import DDPGDC


def build(config: Namespace, device: device):
    if "ddpg_dc" in config.agent:
        agent = DDPGDC(config, device)
    elif "ddpg" in config.agent:
        agent = DDPG(config, device)
    elif "sac" in config.agent:
        agent = SAC(config, device)
    else:
        raise ValueError(f"Agent {config.agent} not implemented")
    return agent
