import torch

from rlc import utils
from rlc import agents
from rlc import engine


def main():
    config = utils.get_args()
    env = utils.get_env(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = agents.build(config, device)

    reward = engine.test(
        env,
        agent,
        config.num_test_episodes,
        render=config.render,
        max_episode_steps=config.max_episode_steps,
    )
    print(f"Mean reward: {reward:.2f}")


if __name__ == "__main__":
    main()
