"""
    Value Iteration
"""

import numpy as np
import gymnasium as gym

from utils.gym_utils import get_observation_shape

class ValueIteration:
    """
        Value Iteration Algorithm for solving Markov Decision Processes (MDPs).
        This implementation updates the value function using the Bellman optimality equation and iterates until convergence, 
        directly deriving the optimal policy from the final value function.
    """
    def __init__(self, env: gym.Env, gamma=1.0, theta=1e-6) -> None:
        self.env = env
        self.gamma = gamma
        self.theta = theta  # Convergence threshold
        self.observation_shape = get_observation_shape(self.env.observation_space)

        self.reset()

    def reset(self):
        """Reset the policy to a uniform random policy and reset the value function."""
        self.policy = np.ones([*self.observation_shape, self.env.action_space.n]) / self.env.action_space.n
        self.value_function = np.zeros(self.observation_shape)

    def iterate(self, inplace=True, print_each_iter=False, print_result=False):
        iteration = 0
        tolerance = 1e-8
        while True:
            iteration += 1
            delta = 0

            if not inplace:
                new_value_function = self.value_function.copy()  # Use copy to avoid modifying the original directly

            for state in np.ndindex(self.observation_shape):
                if self.env.is_terminal_state(state[0]):
                    continue  # Skip terminal states
            
                q = np.zeros(self.env.action_space.n)
                for action in range(self.env.action_space.n):
                    for prob, next_state, reward, terminated in self.env.P[state[0]][action]:
                        q[action] += prob * (reward + self.gamma * self.value_function[next_state])
                
                v = np.max(q)
                delta = max(delta, np.abs(v - self.value_function[state]))
                
                if not inplace:
                    new_value_function[state] = v
                else:
                    self.value_function[state] = v
            
            if not inplace:
                self.value_function = new_value_function.copy()

            if print_each_iter:
                print(f"---------- Iteration {iteration} ----------")
                self.print_value_function()
                print("\n")

            if delta < self.theta:
                for state in np.ndindex(self.observation_shape):
                    if self.env.is_terminal_state(state[0]):
                        continue  # Skip terminal states

                    q = np.zeros(self.env.action_space.n)
                    for action in range(self.env.action_space.n):
                        for prob, next_state, reward, terminated in self.env.P[state[0]][action]:
                            q[action] += prob * (reward + self.gamma * self.value_function[next_state])

                    best_actions = np.argwhere(np.abs(q - np.max(q)) <= tolerance).flatten()
                    self.policy[state] = np.zeros(self.env.action_space.n)
                    self.policy[state][best_actions] = 1.0 / len(best_actions)
                
                if print_result:
                    print(f"---------- Value Function ----------")
                    self.print_value_function()
                    print(f"---------- Policy ----------")
                    self.print_policy()
                
                break

    def print_value_function(self):
        if hasattr(self.env, '_print_value_function'):
            self.env._print_value_function(self.value_function)
        else:
            print(self.value_function)

    def print_policy(self):
        if hasattr(self.env, '_print_policy'):
            self.env._print_policy(self.policy)
        else:
            print(self.policy)