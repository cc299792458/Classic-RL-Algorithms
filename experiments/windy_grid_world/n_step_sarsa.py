"""
    Use N Step Sarsa to solve Wendy Grid World
"""

# Setting n=1000 won't diverge, seems like it's stabler than sarsa lambda with setting a big lambda like 1

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.misc_utils import set_seed
from traditional_algos.td_learning.sarsa import NStepSarsa
from envs.grid_world import WindyGridWorld, animate_trajectory

class NStepSarsaWithLogging(NStepSarsa):
    def estimation_and_control(self, num_episode):
        self.reset()
        episode_lengths = []  # List to record the length of each episode
        for _ in range(num_episode):
            state, info = self.env.reset()
            done = False
            action = np.random.choice(np.arange(self.num_action), p=self.policy[state])
            
            # Initialize the lists to store the episode trajectory
            states = [state]
            actions = [action]
            rewards = []
            
            t = 0  # Time step
            T = np.inf  # Time when the episode ends
            
            while True:
                if t < T:
                    # Execute the action and get the next state, reward, and done flags
                    next_state, reward, terminated, truncated, info = self.env.step(action=actions[-1])
                    rewards.append(reward)
                    done = terminated or truncated

                    if done:
                        T = t + 1  # Set the end time of the episode
                    else:
                        states.append(next_state)
                        next_action = np.random.choice(np.arange(self.num_action), p=self.policy[next_state])
                        actions.append(next_action)
                
                tau = t - self.n + 1  # Time whose estimate is being updated
                
                if tau >= 0:
                    # Update the Q-function using n-step return
                    self.update_q_function(states, actions, rewards, tau, T)

                if tau == T - 1:
                    break  # Break out of the loop when the episode ends

                t += 1
                if t < T:
                    action = actions[-1]  # Continue with the next action

            episode_lengths.append(T)  # Record the length of the current episode

        return episode_lengths

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))
    ##### 0. Load environment #####
    env = WindyGridWorld(max_episode_length=False)
    
    # Instantiate the NStepSarsaWithLogging class with different values of n
    n_values = [i+1 for i in range(10)]
    n_values.append(20)
    num_episode = 10
    num_runs = 50

    plt.figure(figsize=(10, 6))

    for n in n_values:
        all_runs_episode_lengths = []
        for _ in tqdm(range(num_runs), desc=f'n={n}', leave=False):
            agent = NStepSarsaWithLogging(env=env, n=n)
            episode_lengths = agent.estimation_and_control(num_episode=num_episode)
            all_runs_episode_lengths.append(episode_lengths)
        
        # Compute the average episode lengths across runs
        avg_episode_lengths = np.mean(all_runs_episode_lengths, axis=0)
        
        # Plot the average episode lengths for this n value
        plt.plot(range(num_episode), avg_episode_lengths, label=f"n={n}")

    plt.xlabel('Episode')
    plt.ylabel('Average Timesteps')
    plt.title(f'Comparison of Learning Speed with Different n values (Averaged over {num_runs} runs)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'n_step_sarsa_comparison_avg.png'))
    plt.show()
