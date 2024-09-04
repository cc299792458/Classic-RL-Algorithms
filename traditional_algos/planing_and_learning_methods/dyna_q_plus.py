"""
    Dyna Q+
"""

import numpy as np
import gymnasium as gym

from tqdm import tqdm
from traditional_algos.planing_and_learning_methods import DynaQ

# TODO: Why timesteps are different in the episode 0?

class DynaQPlus(DynaQ):
    """
        DynaQ+ with an enhanced model, exploration bonus (only in simulations), and untried action consideration
    """
    def __init__(self, env: gym.Env, gamma=1.0, epsilon=0.1, alpha=0.1, planning_steps=1, kappa=0.001, initial_policy=None) -> None:
        super().__init__(env, gamma, epsilon, alpha, planning_steps, initial_policy)
        self.kappa = kappa  # Exploration bonus parameter
        self.last_update_timestep = {}  # Tracks the timestep of the last update for each (state, action) pair
        self.total_timestep = 0  # Total timestep counter

    def reset(self):
        super().reset()
        self.last_update_timestep = {}  # Reset the timestep tracking model
        self.total_timestep = 0  # Reset the total timestep counter

    def update_q_function(self, state, action, reward, next_state, add_bonus=False):
        # If this state-action pair is being updated for the first time, give a large bonus (only if add_bonus=True)
        if (state, action) not in self.last_update_timestep:
            timesteps_since_last_update = self.total_timestep  # First time being updated, use total_timestep as a large bonus
        else:
            last_timestep = self.last_update_timestep[(state, action)]  # No need for a default value
            timesteps_since_last_update = self.total_timestep - last_timestep

        # Add exploration bonus only if specified (used in simulations)
        bonus = self.kappa * np.sqrt(timesteps_since_last_update) if add_bonus else 0

        # Update Q-function with or without the bonus
        td_error = reward + bonus + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

        # Update the last update timestep for this (state, action) pair
        self.last_update_timestep[(state, action)] = self.total_timestep

    def simulate(self):
        # Perform `self.planning_steps` simulations using the stored model
        for _ in range(self.planning_steps):
            if not self.model:
                continue

            # Randomly pick a state and action
            state = self.env.choose_random_state()
            action = np.random.choice(np.arange(self.num_action))

            # Check if this (state, action) pair is in the model, if not initialize it
            if (state, action) not in self.model:
                next_state = state  # Initial assumption: leads back to the same state
                reward = 0  # Initial assumption: reward is zero
                self.model[(state, action)] = (next_state, reward)
            
            next_state, reward = self.model[(state, action)]

            # Simulated updates include the exploration bonus
            self.update_q_function(state, action, reward, next_state, add_bonus=True)

            # Improve policy based on the updated Q-function
            self.improve_policy(state)

    def estimation_and_control(self, num_episode):
        self.reset()
        with tqdm(total=num_episode) as pbar:
            for _ in range(num_episode):
                state, info = self.env.reset()
                done = False
                while not done:
                    action = np.random.choice(np.arange(self.num_action), p=self.policy[state])
                    next_state, reward, terminated, truncated, info = self.env.step(action=action)
                    done = terminated or truncated

                    # Update Q-function based on real environment interaction (no bonus)
                    self.update_q_function(state, action, reward, next_state, add_bonus=False)

                    # Store the experience in the model
                    self.model[(state, action)] = (next_state, reward)

                    # Improve policy based on new Q values
                    self.improve_policy(state)

                    # Simulate additional learning steps using the model
                    self.simulate()

                    # Move to the next state
                    state = next_state

                    # Increment total timestep after each environment step
                    self.total_timestep += 1

                pbar.update(1)
