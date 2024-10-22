"""
    Use Monte Carlo Method to Solve a 4 x 4 GridWorld
"""

import time

from utils.misc_utils import set_seed
from envs.grid_world.grid_world import GridWorld
from traditional_algos.monte_carlo import MonteCarlo

if __name__ == '__main__':
    set_seed()
    ##### Step 0: Build GridWorld environment and initiate policy #####
    env = GridWorld(height=4, width=4, goal_position=[(0, 0), (3, 3)])
    agent = MonteCarlo(env=env, epsilon=0.0)
    num_episodes = [1, 10, 50, 100, 200, 500, 1000, 5000, 10000, 50_000, 100_000]

    ##### Step 1: Try monte carlo prediction to estimate the value function #####
    for num_episode in num_episodes:
        start_time = time.time()
        agent.prediction(num_episode)
        end_time = time.time()
        print(f"-----Monte carlo prediction with {num_episode} episode-----")
        print(f"Time: {end_time-start_time}")
        agent.print_value_function()
        print("\n")
    
    ##### Step 2: Try monte carlo estimation and control to estimate the q function and optimal policy pi #####
    # TODO: Why there are so many q values equal to zero at the end of the experiment?
    for num_episode in num_episodes:
        start_time = time.time()
        agent.estimation_and_control(num_episode)
        end_time = time.time()
        print(f"-----Monte carlo estimation and control with {num_episode} episode-----")
        print(f"Time: {end_time-start_time}")
        agent.print_q_function()
        agent.print_policy()
        print("\n")

    ##### Step 3: Try monte carlo estimation and control with epsilon greedy #####
    agent = MonteCarlo(env=env, epsilon=0.1)
    for num_episode in num_episodes:
        start_time = time.time()
        agent.estimation_and_control(num_episode)
        end_time = time.time()
        print(f"-----Monte carlo estimation and control (epsilon-greedy) with {num_episode} episode-----")
        print(f"Time: {end_time-start_time}")
        agent.print_q_function()
        agent.print_policy()
        print("\n")