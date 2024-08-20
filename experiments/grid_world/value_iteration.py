"""
    Use Value Iteration to Solve a 4 x 4 GridWorld
"""

import time

from utils.misc_utils import set_seed
from envs.grid_world import GridWorld
from traditional_algos.dynamic_programming.value_iteration.value_iteration import ValueIteration

if __name__ == '__main__':
    set_seed()
    ##### Step 0: Build Environment and Initiate Policy #####
    env = GridWorld()

    theta = 1e-4
    agent = ValueIteration(env=env, theta=theta)

    ##### Step 1: Get the Optimal Policy using Value Iteration #####
    ### Step 1.2 Record iterations ###
    agent.reset()
    agent.iterate(print_each_iter=True, print_result=True)  # 4 iterations
    
    ### Step 1.2 Record time consuming ###
    agent.reset()
    start_time = time.time()
    agent.iterate()
    end_time = time.time()
    print(f"Time consuming: {end_time-start_time}") # about 8e-4 seconds