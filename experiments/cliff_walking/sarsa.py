"""
    Use Sarsa to solve Cliff Walking
"""

import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.misc_utils import set_seed, moving_average_with_padding
from traditional_algos.td_learning.sarsa import Sarsa

class SarsaWithLogging(Sarsa):
    def estimation_and_control(self, num_episode):
        self.reset()
        rewards_per_episode = []  # Initialize the list to store total rewards per episode
        with tqdm(total=num_episode) as pbar:
            for _ in range(num_episode):
                state, info = self.env.reset()
                done = False
                total_reward = 0  # Initialize the total reward for the current episode
                action = np.random.choice(np.arange(self.num_action), p=self.policy[state])
                while not done:
                    next_state, reward, terminated, truncated, info = self.env.step(action=action)
                    done = terminated or truncated

                    # Update Q function
                    next_action = self.update_q_function(state, action, reward, next_state)

                    # Update policy based on new Q values
                    self.improve_policy(state) 

                    # Move to the next state
                    state = next_state
                    action = next_action

                    # Accumulate the reward
                    total_reward += reward

                # Store the total reward for this episode
                rewards_per_episode.append(total_reward)

                pbar.update(1)
        
        return rewards_per_episode

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))
    ##### 0. Load environment and initiate policy #####
    env = gym.make('CliffWalking-v0')
    agent = SarsaWithLogging(env=env)

    ##### 1. Use sarsa to solve cliff walking #####
    alphas = [(i+1)/10 for i in range(10)]
    num_episode = 1000
    window_size = 50  # Define a window size for the moving average

    plt.figure(figsize=(10, 6))

    for alpha in alphas:
        agent.reset()
        agent.set_alpha(alpha=alpha)
        rewards_per_episode = agent.estimation_and_control(num_episode=num_episode)

        # Compute the moving average of the rewards
        smoothed_rewards = moving_average_with_padding(rewards_per_episode, window_size=window_size)

        # Plot the smoothed rewards with a label indicating the alpha value
        plt.plot(range(num_episode), smoothed_rewards, label=f'Alpha = {alpha:.1f}')

    # Plot the sum of rewards per episode
    plt.xlabel('Episode')
    plt.ylabel('Sum of Rewards during Episode')
    plt.title('Sum of Rewards vs. Episode for Cliff Walking - Sarsa')
    plt.grid(True)
    plt.legend()  # Add a legend to indicate alpha values

    rewards_vs_episode_plot_path = os.path.join(log_dir, 'sarsa_rewards_vs_episode.png')
    plt.savefig(rewards_vs_episode_plot_path)

    plt.show()