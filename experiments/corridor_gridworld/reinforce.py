"""
    Use REINFORCE to solve Corridor GridWorld
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.misc_utils import set_seed
from envs.grid_world import CorridorGridWorld
from traditional_algos.reinforce import REINFORCE, LinearApproximator
class REINFORCEWithLog(REINFORCE):
    def train(self, num_episodes):
        """Train the agent and log the returns for each episode."""
        episode_returns = []
        # for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False

            log_probs = []
            rewards = []

            while not done:
                action, log_prob = self.policy.select_action()
                next_state, reward, done, truncated, _ = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)

                state = next_state

            returns = self.compute_returns(rewards)
            self.policy.update(log_probs, returns, self.alpha)

            total_return = sum(rewards)
            episode_returns.append(total_return)

            # if episode % 100 == 0:
            #     print(f"Episode {episode}, Total Return: {total_return}")

        return episode_returns

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))

    ##### 0. Load environment and agent #####
    env = CorridorGridWorld()

    policy = LinearApproximator()
    agent = REINFORCEWithLog(env=env, policy=policy)

    ##### 1. Train #####
    num_episodes = 1000
    num_runs = 100
    all_runs_returns = []

    for _ in tqdm(range(num_runs), desc="Total Runs"):
        episode_returns = agent.train(num_episodes=num_episodes)
        all_runs_returns.append(episode_returns)

    # Compute the average returns across all runs
    average_returns = np.mean(all_runs_returns, axis=0)

    # Plot the average returns over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(average_returns)
    plt.xlabel('Episode')
    plt.ylabel('Average Total Return')
    plt.title(f'Average Total Return over {num_runs} Runs')
    plt.grid(True)
    plt.show()