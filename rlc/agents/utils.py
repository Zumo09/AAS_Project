import torch
from torch.nn import Module


@torch.no_grad()
def soft_update(target: Module, local: Module, tau: float) -> None:
    for tp, lp in zip(target.parameters(), local.parameters()):
        tp.data.mul_(1 - tau)
        tp.data.add_(tau * lp.data)


def load_model(base_model: Module, load_path: str, device: torch.device) -> Module:
    state_dict = torch.load(load_path, map_location=device)
    base_model.load_state_dict(state_dict)
    return base_model


def save_model(model: Module, save_path: str) -> None:
    torch.save(model.state_dict(), save_path)
