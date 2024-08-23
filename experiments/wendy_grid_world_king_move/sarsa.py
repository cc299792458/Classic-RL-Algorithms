"""
    Use Sarsa to solve Wendy Grid World King Move
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.misc_utils import set_seed
from traditional_algos.td_learning.sarsa import Sarsa
from envs.grid_world import WindyGridWorldKingsMoves, animate_trajectory

class SarsaWithLogging(Sarsa):
    def estimation_and_control(self, num_episode):
        self.reset()
        episode_lengths = []
        timesteps = 0
        with tqdm(total=num_episode) as pbar:
            for _ in range(num_episode):
                state, _ = self.env.reset()
                done = False
                # timesteps = 0
                while not done:
                    action = np.random.choice(np.arange(self.num_action), p=self.policy[state])
                    next_state, reward, terminated, truncated, info = self.env.step(action=action)
                    done = terminated or truncated
                    
                    # Update Q function
                    next_action = np.random.choice(np.arange(self.num_action), p=self.policy[next_state])
                    td_error = (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
                    self.Q[state][action] += self.alpha * td_error
                    
                    # Update policy
                    self.improve_policy(state=state)
                    
                    state = next_state
                    timesteps += 1

                episode_lengths.append(timesteps)

                pbar.update(1)

        return episode_lengths

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.dirname(os.path.abspath(__file__))
    ##### 0. Load environment and initiate policy #####
    env = WindyGridWorldKingsMoves(max_episode_length=False)
    agent = SarsaWithLogging(env=env)

    ##### 1. Use td learning's estimation and control to solve wendy grid world king move
    num_episode = 5_000
    agent.reset()
    episode_lengths = agent.estimation_and_control(num_episode=num_episode)

    # Plot the Episode vs Timesteps graph
    plt.figure(figsize=(10, 6))
    plt.plot(episode_lengths, range(num_episode))
    plt.xlabel('Timesteps')
    plt.ylabel('Episode')
    plt.title('Timesteps vs. Episode to Solve Windy Grid World')
    plt.grid(True)

    timesteps_vs_episode_plot_path = os.path.join(log_dir, 'timesteps_vs_episode.png')
    plt.savefig(timesteps_vs_episode_plot_path)
    
    plt.show()

    # Generate and plot the trajectory under the optimal policy
    agent.set_epsilon(epsilon=0.0)
    
    state, _ = env.reset()
    trajectory = []
    done = False

    while not done:
        # Choose the best action according to the learned policy
        action = np.random.choice(np.arange(agent.num_action), p=agent.policy[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        trajectory.append((state, action))
        state = next_state
        done = terminated or truncated

    # Animate the trajectory
    animate_trajectory(env, trajectory, grid_size=(env.height, env.width), goal_state=env.goal_state)