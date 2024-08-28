"""
    Use REINFORCE to solve Corridor GridWorld
"""

# NOTE: large alpha may let the process unstable.

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.misc_utils import set_seed
from envs.grid_world import CorridorGridWorld
from traditional_algos.reinforce import REINFORCE, LinearApproximator

class REINFORCEWithLogging(REINFORCE):
    def train(self, num_episodes):
        """Train the agent and log the returns and weights for each episode."""
        self.reset()
        episode_returns = []
        weights_history = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False

            dlog_pis = []
            rewards = []

            # Record the weights before each episode
            weights_history.append(self.policy.w.copy())

            while not done:
                action, dlog_pi = self.policy.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)

                dlog_pis.append(dlog_pi)
                rewards.append(reward)

                state = next_state

            returns = self.compute_returns(rewards)
            self.update(dlog_pis, returns)

            undiscounted_return = sum(rewards)
            episode_returns.append(undiscounted_return)
            
        return episode_returns, weights_history

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))
    ##### 0. Load environment and agent #####
    env = CorridorGridWorld()

    policy = LinearApproximator()
    agent = REINFORCEWithLogging(env=env, gamma=0.99, policy=policy)

    ##### 1. Train #####
    num_episodes = 1000
    num_runs = 100
    all_runs_returns = []
    all_runs_weights = []

    for _ in tqdm(range(num_runs), desc="Total Runs"):
        episode_returns, weights_history = agent.train(num_episodes=num_episodes)
        all_runs_returns.append(episode_returns)
        all_runs_weights.append(weights_history)

    # Compute the average returns across all runs
    average_returns = np.mean(all_runs_returns, axis=0)
    average_weights = np.mean(all_runs_weights, axis=0)

    # Plot the average returns over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(average_returns)
    plt.xlabel('Episode')
    plt.ylabel('Average Total Return')
    plt.title(f'REINFORCE\nAverage Total Return over {num_runs} Runs')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'reinforce_average_total_return.png'))  # Save the plot
    plt.show()

    # Plot the weights over episodes
    plt.figure(figsize=(10, 6))
    for i in range(average_weights.shape[1]):
        plt.plot(average_weights[:, i], label=f'Weight {i + 1}')
    plt.xlabel('Episode')
    plt.ylabel('Weight Value')
    plt.title('REINFORCE\nWeights Evolution over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'reinforce_weights_evolution.png'))  # Save the plot
    plt.show()
