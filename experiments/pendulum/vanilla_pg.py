"""
    Vanilla Policy Gradient solving MountainCarContinuous
"""

import os
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from utils.misc_utils import set_seed
from deep_algos.vanilla_pg import VanillaPolicyGradient, ValueNetwork, PolicyNetwork

if __name__ == '__main__':
    set_seed()
    # Initialize the environment and device
    env = gym.make("MountainCarContinuous-v0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the policy network, value network (baseline), and Vanilla Policy Gradient agent
    policy = PolicyNetwork(input_dim=env.observation_space.shape[0], output_dim=env.action_space.shape[0])
    baseline = ValueNetwork(input_dim=env.observation_space.shape[0])
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')

    agent = VanillaPolicyGradient(
        env=env,
        gamma=0.99,
        lr_policy=3e-4,
        lr_baseline=3e-4,
        policy=policy,
        baseline=baseline,
        device=device,
        save_interval=10_000,  # Save every 10,000 timesteps
        checkpoint_dir=checkpoint_dir,
        batch_size=32
    )

    # Train the agent
    total_timesteps = 1_000_000
    returns = agent.train(total_timesteps)

    # Plot the returns over episodes and save the plot
    plt.figure(figsize=(10, 6))
    plt.plot(returns)
    plt.xlabel('Episode')
    plt.ylabel('Total Return')
    plt.title('Vanilla Policy Gradient - MountainCarContinuous')
    plt.grid(True)
    log_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(log_dir, 'vanilla_pg_mountaincarcontinuous.png'))
    plt.show()

    # Create a new environment instance for testing with rendering enabled
    test_env = gym.make("MountainCarContinuous-v0", render_mode='human')

    # Test and render the agent with different checkpoints
    num_checkpoints = len(range(10_000, total_timesteps + 1, 10_000))
    for i in range(1, num_checkpoints + 1):
        checkpoint_episode = i * 10_000
        checkpoint_path = os.path.join(checkpoint_dir, f'policy_checkpoint_{checkpoint_episode}.pth')
        agent.load_checkpoint(checkpoint_path)
        agent.test_agent(test_env)
