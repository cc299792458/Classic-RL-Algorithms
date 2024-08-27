"""
    Use Sarsa Lambda to solve Wendy Grid World
"""

# TODO: Why setting gamma=1 and lambda=1 will cause divergence of Q? Seems like the variance is too big.

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.misc_utils import set_seed
from traditional_algos.td_learning.sarsa import SarsaLambda
from envs.grid_world import WindyGridWorld

class SarsaLambdaWithLogging(SarsaLambda):
    def estimation_and_control(self, num_episode):
        self.reset()
        episode_lengths = []  # List to record the length of each episode
        for _ in range(num_episode):
            state, info = self.env.reset()
            done = False
            action = np.random.choice(np.arange(self.num_action), p=self.policy[state])
            
            t = 0  # Time step

            while not done:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_action = self.update_q_function(state, action, reward, next_state)
                self.improve_policy(state)
                state = next_state
                action = next_action

                t += 1  # Increment the time step

            episode_lengths.append(t)  # Record the length of the current episode

        return episode_lengths

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))
    ##### 0. Load environment #####
    env = WindyGridWorld(max_episode_length=False)
    
    # Instantiate the SarsaLambdaWithLogging class with different values of λ
    lambda_values = [0.0, 0.1, 0.2, 0.5, 0.8, 0.9, 0.92, 0.95, 0.97, 0.99, 1.0]  # λ values from 0.0 to 1.0
    
    num_episode = 10
    num_runs = 50

    plt.figure(figsize=(10, 6))

    for lambda_ in lambda_values:
        all_runs_episode_lengths = []
        for _ in tqdm(range(num_runs), desc=f'λ={lambda_:.2f}', leave=False):
            agent = SarsaLambdaWithLogging(env=env, gamma=0.9, lambda_=lambda_)
            episode_lengths = agent.estimation_and_control(num_episode=num_episode)
            all_runs_episode_lengths.append(episode_lengths)
        
        # Compute the average episode lengths across runs
        avg_episode_lengths = np.mean(all_runs_episode_lengths, axis=0)
        
        # Plot the average episode lengths for this λ value
        plt.plot(range(num_episode), avg_episode_lengths, label=f"λ={lambda_:.1f}")

    plt.xlabel('Episode')
    plt.ylabel('Average Timesteps')
    plt.title(f'Comparison of Learning Speed with Different λ values (Averaged over {num_runs} runs)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'sarsa_lambda_comparison_avg.png'))
    plt.show()