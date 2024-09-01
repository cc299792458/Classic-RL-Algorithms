import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from envs import KArmedBandit
from utils.misc_utils import set_seed
from traditional_algos.epsilon_greedy import EpsilonGreedy
    
if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))
    ##### 0. Load environment, policy and initialize parameters #####
    env = KArmedBandit()
    agent = EpsilonGreedy(env=env)

    n_runs = 2000
    epsilons = [0.0, 0.001, 0.01, 0.1, 0.2, 0.5]

    ##### 1. Try 2000 times with different epsilons #####
    all_avg_rewards = []
    all_avg_optimal_action_selection = []

    for epsilon in epsilons:
        avg_rewards_per_run = np.zeros(env.max_time_steps)  # To accumulate rewards for averaging
        avg_optimal_action_selection = np.zeros(env.max_time_steps)

        agent.set_epsilon(epsilon=epsilon)
        # Add a tqdm progress bar for the 2000 runs
        for _ in tqdm(range(n_runs), desc=f'Epsilon {epsilon}'):
            rewards = []
            optimal_action_selections = []

            done = False
            while not done:
                action = agent.select_action()
                _, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                rewards.append(reward)
                optimal_action_selections.append(1 if action in env.optimal_actions else 0)  # Check if action is one of the optimal actions

                agent.update_q_values(action, reward)

            # Accumulate rewards and optimal action selection for this run
            avg_rewards_per_run += np.array(rewards)
            avg_optimal_action_selection += np.array(optimal_action_selections)

            # Reset the environment and agent after each run
            env.reset()
            agent.reset()

        # Compute the average reward and optimal action selection across all runs
        avg_rewards_per_run /= n_runs
        avg_optimal_action_selection /= n_runs
        all_avg_rewards.append(avg_rewards_per_run)
        all_avg_optimal_action_selection.append(avg_optimal_action_selection)


    ##### 2. Plot the results #####
    # Plot the average reward per timestep for different epsilon values
    plt.figure(figsize=(10, 6))
    for idx, epsilon in enumerate(epsilons):
        plt.plot(all_avg_rewards[idx], label=f'epsilon={epsilon}')
    
    plt.title('Average Reward per Timestep for Different Epsilon Values')
    plt.xlabel('Timestep')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    # Save the reward figure
    plt.savefig(os.path.join(log_dir, 'k_arm_bandit_epsilon_greedy_rewards.png'))
    plt.show()

    # Plot the proportion of optimal action selections per timestep for different epsilon values
    plt.figure(figsize=(10, 6))
    for idx, epsilon in enumerate(epsilons):
        plt.plot(all_avg_optimal_action_selection[idx], label=f'epsilon={epsilon}')
    
    plt.title('Proportion of Optimal Action Selection per Timestep for Different Epsilon Values')
    plt.xlabel('Timestep')
    plt.ylabel('Proportion of Optimal Action Selection')
    plt.legend()
    plt.grid(True)
    # Save the optimal action selection figure
    plt.savefig(os.path.join(log_dir, 'k_arm_bandit_epsilon_greedy_optimal_action_selection.png'))
    plt.show()