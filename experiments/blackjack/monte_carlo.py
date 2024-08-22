"""
    Use Monte Carlo Method to solve Black Jack
"""

import gymnasium as gym

from utils.misc_utils import set_seed
from traditional_algos.monte_carlo import MonteCarlo

if __name__ == '__main__':
    set_seed()
    ##### 0. Load environment and initiate policy #####
    env = gym.make('Blackjack-v1', natural=False, sab=True)
    policy = MonteCarlo()