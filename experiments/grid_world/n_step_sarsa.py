"""
    Use N Step Sarsa to solve Grid World, used for debugging 
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.misc_utils import set_seed
from traditional_algos.td_learning.sarsa import NStepSarsa
from envs.grid_world import GridWorld

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))
    ##### 0. Load environment and agent #####
    env = GridWorld(width=3, height=3, max_episode_length=False)
    agent = NStepSarsa(env=env, n=2)

    ##### 1. Solve the problem #####
    num_episode = 1000
    agent.estimation_and_control(num_episode=num_episode)
    agent.print_q_function()