import torch
from torch.utils.tensorboard.writer import SummaryWriter

from rlc import utils
from rlc import agents
from rlc import engine
from rlc.replay_buffer import ReplayBuffer


def main():
    config = utils.get_args()
    env = utils.get_env(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = agents.build(config, device)

    log_dir, model_path = utils.setup_logging(config)

    buffer = ReplayBuffer(
        capacity=int(config.buffer_size),
        sample_size=int(config.sample_size),
        ready_size=int(config.ready_size),
    )

    with SummaryWriter(log_dir) as writer:
        engine.train(
            env,
            agent,
            buffer,
            config.num_episodes,
            model_base_path=model_path,
            n_test_episodes=config.num_test_episodes,
            checkpoint_interval=config.checkpoint_interval,
            update_every=config.update_every,
            render=config.render,
            max_episode_steps=config.max_episode_steps,
            writer=writer,
        )


if __name__ == "__main__":
    main()
