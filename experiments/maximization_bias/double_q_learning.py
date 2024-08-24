"""
    Use Double Q Learning to solve Maximization Bias
"""

import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from tqdm import tqdm
from envs import MaximizationBias
from traditional_algos.td_learning.q_learning import DoubleQLearning
from utils.misc_utils import set_seed, moving_average_with_padding

class DoubleQLearningWithLogging(DoubleQLearning):
    def estimation_and_control(self, num_episode):
        self.reset()
        left_action_count = 0  # To track how often left action is taken
        left_action_ratio = []
        with tqdm(total=num_episode) as pbar:
            for index in range(num_episode):
                state, info = self.env.reset()
                done = False
                
                while not done:
                    action = np.random.choice(np.arange(self.num_action), p=self.policy[state])
                    if state == 0 and action == 0:
                        left_action_count += 1

                    next_state, reward, terminated, truncated, info = self.env.step(action=action)
                    done = terminated or truncated

                    # Update Q function using Double Q-learning
                    if np.random.rand() < 0.5:
                        best_next_action = np.argmax(self.Q1[next_state])
                        td_error = reward + self.gamma * self.Q2[next_state][best_next_action] - self.Q1[state][action]
                        self.Q1[state][action] += self.alpha * td_error
                    else:
                        best_next_action = np.argmax(self.Q2[next_state])
                        td_error = reward + self.gamma * self.Q1[next_state][best_next_action] - self.Q2[state][action]
                        self.Q2[state][action] += self.alpha * td_error

                    # Update policy based on new Q values
                    self.improve_policy(state)

                    # Move to the next state
                    state = next_state

                # Calculate and store the left action ratio after each episode
                left_action_ratio.append(left_action_count / (index + 1))

                pbar.update(1)
        
        return left_action_ratio

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))
    ##### 0. Load environment and initiate policy #####
    env = MaximizationBias()
    agent = DoubleQLearningWithLogging(env=env)

    ##### 1. Use Double Q-learning to solve the Maximization Bias problem #####
    num_episode = 1000
    window_size = 10

    agent.reset()
    left_action_ratio = agent.estimation_and_control(num_episode=num_episode)

    smoothed_left_action_ratio = moving_average_with_padding(data=left_action_ratio, window_size=window_size)

    # Plot the percentage of left actions taken over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_episode), smoothed_left_action_ratio, label="Double Q-learning")
    plt.xlabel('Episode')
    plt.ylabel('% Left Action in State A')
    plt.title('Double Q-learning on Maximization Bias Problem')
    plt.grid(True)

    plot_path = os.path.join(log_dir, 'double_q_learning_maximization_bias.png')
    plt.savefig(plot_path)
    plt.show()
