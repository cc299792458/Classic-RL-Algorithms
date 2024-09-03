"""
    Use REINFORCE with Baseline to solve Corridor GridWorld
"""

# TODO: Why it stucks sometimes? Especially when setting a large alpha

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.misc_utils import set_seed
from envs.grid_world import CorridorGridWorld
from traditional_algos.policy_gradient.reinforce import REINFORCEWithBaseline, LinearApproximator, ValueEstimation

class REINFORCEWithBaselineLogging(REINFORCEWithBaseline):
    def train(self, num_episodes):
        """Train the agent and log the returns and weights for each episode."""
        self.reset()
        episode_returns = []
        policy_weights_history = []
        baseline_weights_history = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False

            dlog_pis = []
            rewards = []
            state_vectors = []

            # Record the weights before each episode
            policy_weights_history.append(self.policy.w.copy())
            baseline_weights_history.append(self.baseline.w.copy())

            while not done:
                action, dlog_pi = self.policy.select_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)

                dlog_pis.append(dlog_pi)
                rewards.append(reward)
                state_vectors.append(self._to_vector(state))

                state = next_state

            returns = self.compute_returns(rewards)
            self.update(dlog_pis, returns, state_vectors)

            undiscounted_return = sum(rewards)
            episode_returns.append(undiscounted_return)
            
        return episode_returns, policy_weights_history, baseline_weights_history

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))

    ##### 0. Load environment and agent #####
    env = CorridorGridWorld()

    policy = LinearApproximator()
    baseline = ValueEstimation()
    agent = REINFORCEWithBaselineLogging(env=env, gamma=0.99, policy=policy, baseline=baseline)

    ##### 1. Train #####
    num_episodes = 1000
    num_runs = 100
    all_runs_returns = []
    all_runs_policy_weights = []
    all_runs_baseline_weights = []

    for _ in tqdm(range(num_runs), desc="Total Runs"):
        episode_returns, policy_weights_history, baseline_weights_history = agent.train(num_episodes=num_episodes)
        all_runs_returns.append(episode_returns)
        all_runs_policy_weights.append(policy_weights_history)
        all_runs_baseline_weights.append(baseline_weights_history)

    # Compute the average returns across all runs
    average_returns = np.mean(all_runs_returns, axis=0)
    average_policy_weights = np.mean(all_runs_policy_weights, axis=0)
    average_baseline_weights = np.mean(all_runs_baseline_weights, axis=0)

    # Plot the average returns over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(average_returns)
    plt.xlabel('Episode')
    plt.ylabel('Average Total Return')
    plt.title(f'REINFORCE with Baseline\nAverage Total Return over {num_runs} Runs')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'reinforce_with_baseline_average_total_return.png'))  # Save the plot
    plt.show()

    # Plot the policy weights over episodes
    plt.figure(figsize=(10, 6))
    for i in range(average_policy_weights.shape[1]):
        plt.plot(average_policy_weights[:, i], label=f'Policy Weight {i + 1}')
    plt.xlabel('Episode')
    plt.ylabel('Weight Value')
    plt.title('REINFORCE with Baseline\nPolicy Weights Evolution over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'reinforce_with_baseline_policy_weights_evolution.png'))  # Save the plot
    plt.show()

    # Plot the baseline weights over episodes
    plt.figure(figsize=(10, 6))
    for i in range(average_baseline_weights.shape[1]):
        plt.plot(average_baseline_weights[:, i], label=f'Baseline Weight {i + 1}')
    plt.xlabel('Episode')
    plt.ylabel('Weight Value')
    plt.title('REINFORCE with Baseline\nBaseline Weights Evolution over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'reinforce_with_baseline_baseline_weights_evolution.png'))  # Save the plot
    plt.show()
