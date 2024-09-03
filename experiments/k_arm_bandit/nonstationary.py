"""
    Non-stationary K Arm Bandit solved by Sample-Average-Update Policy and Constant-Stepsize-Policy

    This is the implementation of the Exercise 2.5 of the RL Book.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from envs import NonStationaryBandit
from utils.misc_utils import set_seed
from traditional_algos.epsilon_greedy import EpsilonGreedy

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the current script
    ##### 0. Load environment, policy and initialize parameters #####
    max_time_steps = 20_000
    env = NonStationaryBandit(k=10, max_time_steps=max_time_steps, walk_std=0.01)
    sample_avg_policy = EpsilonGreedy(env=env)
    const_step_policy = EpsilonGreedy(env=env, step_size=0.1)

    n_runs = 1000

    ##### 1. Try 200 times with sample-average policy and constant stepsize policy #####
    sample_avg_rewards = np.zeros(env.max_time_steps)
    sample_avg_optimal_action_selection = np.zeros(env.max_time_steps)

    const_step_rewards = np.zeros(env.max_time_steps)
    const_step_optimal_action_selection = np.zeros(env.max_time_steps)

    for _ in tqdm(range(n_runs), desc="Running experiments"):
        rewards = []
        optimal_action_selections = []
        
        # Sample-average policy
        done = False
        while not done:
            action = sample_avg_policy.select_action()
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rewards.append(reward)
            optimal_action_selections.append(1 if action in env.optimal_actions else 0)

            sample_avg_policy.update_q_values(action, reward)

        # Accumulate rewards and optimal action selections
        sample_avg_rewards += np.array(rewards)
        sample_avg_optimal_action_selection += np.array(optimal_action_selections)

        # Reset the environment and policy after each run
        env.reset()
        sample_avg_policy.reset()

        rewards = []
        optimal_action_selections = []
        
        # Constant-stepsize policy
        done = False
        while not done:
            action = const_step_policy.select_action()
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            rewards.append(reward)
            optimal_action_selections.append(1 if action in env.optimal_actions else 0)

            const_step_policy.update_q_values(action, reward)

        # Accumulate rewards and optimal action selections
        const_step_rewards += np.array(rewards)
        const_step_optimal_action_selection += np.array(optimal_action_selections)

        # Reset the environment and policy after each run
        env.reset()
        const_step_policy.reset()

    # Compute the average reward across all runs
    sample_avg_rewards /= n_runs
    sample_avg_optimal_action_selection /= n_runs

    const_step_rewards /= n_runs
    const_step_optimal_action_selection /= n_runs

    ##### 2. Plot the results #####
    # Plot the average reward per timestep for sample-average and constant step-size methods
    plt.figure(figsize=(10, 6))
    plt.plot(sample_avg_rewards, label='Sample-Average Method')
    plt.plot(const_step_rewards, label='Constant Step-Size Method')
    
    plt.title('Average Reward per Timestep')
    plt.xlabel('Timestep')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    # Save the reward figure
    plt.savefig(os.path.join(log_dir, 'nonstationary_bandit_rewards_comparison.png'))
    plt.show()

    # Plot the proportion of optimal action selections per timestep for sample-average and constant step-size methods
    plt.figure(figsize=(10, 6))
    plt.plot(sample_avg_optimal_action_selection, label='Sample-Average Method')
    plt.plot(const_step_optimal_action_selection, label='Constant Step-Size Method')
    
    plt.title('Proportion of Optimal Action Selection per Timestep')
    plt.xlabel('Timestep')
    plt.ylabel('Proportion of Optimal Action Selection')
    plt.legend()
    plt.grid(True)
    # Save the optimal action selection figure
    plt.savefig(os.path.join(log_dir, 'nonstationary_bandit_optimal_action_comparison.png'))
    plt.show()