"""
    Use REINFORCE-Continous to solve Mountain Car Continous
"""

import os
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from utils.misc_utils import set_seed
from traditional_algos.reinforce import REINFORCEContinuous, PolicyNetwork


if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))
    # Initialize the environment and device
    env = gym.make("MountainCarContinuous-v0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the policy network and REINFORCE agent
    policy = PolicyNetwork(input_dim=env.observation_space.shape[0], output_dim=env.action_space.shape[0])
    agent = REINFORCEContinuous(env=env, gamma=0.99, alpha=1e-3, policy=policy, device=device)

    # Train the agent
    num_episodes = 1000
    returns = agent.train(num_episodes)

    # Plot the returns over episodes and save the plot
    plt.figure(figsize=(10, 6))
    plt.plot(returns)
    plt.xlabel('Episode')
    plt.ylabel('Total Return')
    plt.title('REINFORCE Continuous - Mountain Car')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'reinforce_continuous_mountain_car.png'))
    plt.show()

