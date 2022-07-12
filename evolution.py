import os
import torch

from rlc import utils
from rlc import agents
from rlc import engine


def main():
    config = utils.get_args()
    env = utils.get_env(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = agents.build(config, device)

    best = ""
    best_score = 0
    model_path = os.path.join("runs", config.agent, "models")
    for model in sorted(os.listdir(model_path)):
        checkpoint = os.path.join(model_path, model)
        agent.load(checkpoint)
        reward = engine.test_episode(
            env,
            agent,
            render=config.render,
            max_episode_steps=config.max_episode_steps,
        )
        print(f"{model} reward: {reward:.2f}")
        if reward >= best_score:
            best = model
            best_score = reward

    print("=" * 50)
    print(f"BEST {best} reward: {best_score:.2f}")


if __name__ == "__main__":
    main()
