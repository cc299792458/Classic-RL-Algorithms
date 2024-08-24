"""
    Use Q Learning to solve Maximization Bias
"""

import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from tqdm import tqdm
from envs import MaximizationBias
from traditional_algos.td_learning.q_learning import QLearning
from utils.misc_utils import set_seed, moving_average_with_padding

# TODO: Why converge slower compared to the book's example?

class MaximizationBiasQLearning(QLearning):
    def reset(self):
        """
        Override reset method to handle different action space sizes in different states.
        """
        self.Q = {}
        self.policy = {}

        # State A (0) has 2 actions, State B (1) has 10 actions
        for state in range(self.env.observation_space.n):
            num_actions = 2 if state == 0 else 10
            self.Q[state] = np.zeros(num_actions)
            self.policy[state] = np.ones(num_actions) / num_actions

    def estimation_and_control(self, num_episode):
        self.reset()
        left_action_count = 0  # To track how often left action is taken
        left_action_ratio = []
        
        for index in range(num_episode):
            state, info = self.env.reset()
            done = False
            
            while not done:
                num_actions = 2 if state == 0 else 10  # Adjust the number of actions based on state
                action = np.random.choice(np.arange(num_actions), p=self.policy[state])
                if state == 0 and action == 0:
                    left_action_count += 1

                next_state, reward, terminated, truncated, info = self.env.step(action=action)
                done = terminated or truncated

                # Update Q function
                self.update_q_function(state, action, reward, next_state)

                # Update policy based on new Q values
                self.improve_policy(state)

                # Move to the next state
                state = next_state

            # Calculate and store the left action ratio after each episode
            left_action_ratio.append(left_action_count / (index + 1))
        
        return left_action_ratio
    
    def improve_policy(self, state):
        tolerance = 1e-8
        max_q_value = np.max(self.Q[state])
        best_actions = np.argwhere(np.abs(self.Q[state] - max_q_value) <= tolerance).flatten()

        # Update the policy to give equal probability to these best actions
        self.policy[state] = np.zeros_like(self.policy[state])
        self.policy[state][best_actions] = 1.0 / len(best_actions)
        
        # Implement ε-greedy exploration
        if self.epsilon > 0:
            self.policy[state] = (1 - self.epsilon) * self.policy[state] + (self.epsilon / len(self.policy[state]))

class QLearningWithLogging(MaximizationBiasQLearning):
    def estimation_and_control(self, num_episode):
        self.reset()
        left_action_count = 0  # To track how often left action is taken
        left_action_ratio = []
        for index in range(num_episode):
            state, info = self.env.reset()
            done = False
            
            while not done:
                num_actions = 2 if state == 0 else 10  # Adjust the number of actions based on state
                action = np.random.choice(np.arange(num_actions), p=self.policy[state])
                if state == 0 and action == 0:
                    left_action_count += 1

                next_state, reward, terminated, truncated, info = self.env.step(action=action)
                done = terminated or truncated

                # Update Q function
                self.update_q_function(state, action, reward, next_state)

                # Update policy based on new Q values
                self.improve_policy(state)

                # Move to the next state
                state = next_state

            # Calculate and store the left action ratio after each episode
            left_action_ratio.append(left_action_count / (index + 1))
        
        return left_action_ratio

def run_multiple_experiments(agent, num_episodes_per_run, num_runs):
    all_runs_left_action_ratios = []

    for _ in tqdm(range(num_runs), desc="Running experiments"):
        left_action_ratio = agent.estimation_and_control(num_episodes_per_run)
        all_runs_left_action_ratios.append(left_action_ratio)

    # Convert list of lists to numpy array for easier manipulation
    all_runs_left_action_ratios = np.array(all_runs_left_action_ratios)

    # Compute the average across the runs
    avg_left_action_ratio = np.mean(all_runs_left_action_ratios, axis=0)

    return avg_left_action_ratio

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))
    ##### 0. Load environment and initiate policy #####
    env = MaximizationBias()
    agent = QLearningWithLogging(env=env)

    ##### 1. Use Q-learning to solve the Maximization Bias problem #####
    num_episode = 5_000
    num_runs = 100
    window_size = 1

    agent.reset()
    avg_left_action_ratio = run_multiple_experiments(agent, num_episodes_per_run=num_episode, num_runs=num_runs)

    smoothed_left_action_ratio = moving_average_with_padding(data=avg_left_action_ratio, window_size=window_size)

    # Plot the percentage of left actions taken over episodes
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_episode), smoothed_left_action_ratio, label="Q-learning (Averaged over 100 runs)")
    plt.xlabel('Episode')
    plt.ylabel('% Left Action in State A')
    plt.title('Q-learning on Maximization Bias Problem (Averaged over 100 runs)')
    plt.grid(True)

    plot_path = os.path.join(log_dir, 'q_learning_maximization_bias_avg_100runs.png')
    plt.savefig(plot_path)
    plt.show()
