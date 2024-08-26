"""
    Sarsa
"""

import numpy as np
import gymnasium as gym

from tqdm import tqdm
from utils.gym_utils import get_observation_shape

class Sarsa:
    def __init__(self, env: gym.Env, gamma=1.0, epsilon=0.1, alpha=0.1, initial_policy=None) -> None:
        self.env = env
        self.gamma = gamma  # Discounting rate
        self.epsilon = epsilon  # Epsilon-greedy
        self.alpha = alpha  # Update step size
        self.observation_shape = get_observation_shape(env.observation_space)
        self.num_action = self.env.action_space.n

        self.initial_policy = initial_policy
        self.reset()

    def reset(self):
        """
            Reset the policy to a uniform random policy and reset the value and Q functions.
        """
        if self.initial_policy is None:
            self.policy = np.ones((*self.observation_shape, self.num_action)) / self.num_action
        else:
            self.policy = self.initial_policy
        self.Q = np.zeros((*self.observation_shape, self.num_action))  # Action-value function

    def estimation_and_control(self, num_episode):
        self.reset()
        with tqdm(total=num_episode) as pbar:
            for _ in range(num_episode):
                state, info = self.env.reset()
                done = False
                action = np.random.choice(np.arange(self.num_action), p=self.policy[state])
                while not done:
                    next_state, reward, terminated, truncated, info = self.env.step(action=action)
                    done = terminated or truncated

                    # Update Q function
                    next_action = self.update_q_function(state, action, reward, next_state)

                    # Update policy based on new Q values
                    self.improve_policy(state) 

                    # Move to the next state
                    state = next_state
                    # Next action is selected before updating the q function
                    action = next_action

                pbar.update(1)
        
    def update_q_function(self, state, action, reward, next_state):
        # Choose next action from next state using updated policy
        next_action = np.random.choice(np.arange(self.num_action), p=self.policy[next_state])

        td_error = reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

        return next_action
        
    def improve_policy(self, state):
        tolerance = 1e-8
        max_q_value = np.max(self.Q[state])
        best_actions = np.argwhere(np.abs(self.Q[state] - max_q_value) <= tolerance).flatten()

        # Update the policy to give equal probability to these best actions
        self.policy[state] = np.zeros(self.num_action)
        self.policy[state][best_actions] = 1.0 / len(best_actions)
        
        # Implement ε-greedy exploration
        if self.epsilon > 0:
            self.policy[state] = (1 - self.epsilon) * self.policy[state] + (self.epsilon / self.num_action)
    
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_alpha(self, alpha):
        self.alpha = alpha

    def print_value_function(self):
        """Print the current value function."""
        if hasattr(self.env, '_print_value_function'):
            self.env._print_value_function(self.value_function)
        else:
            print(self.value_function)

    def print_q_function(self):
        """Print the current Q function."""
        if hasattr(self.env, '_print_q_function'):
            self.env._print_q_function(self.Q)
        else:
            print(self.Q)

    def print_policy(self):
        """Print the current policy."""
        if hasattr(self.env, '_print_policy'):
            self.env._print_policy(self.policy)
        else:
            print(self.policy)

class ExpectedSarsa(Sarsa):
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

                    # Update Q function
                    next_action = self.update_q_function(state, action, reward, next_state)

                    # Update policy based on new Q values
                    self.improve_policy(state) 

                    # Move to the next state
                    state = next_state
                    
                pbar.update(1)

    def update_q_function(self, state, action, reward, next_state):
        # Calculate the expected Q value
        expected_q = np.sum(self.policy[next_state] * self.Q[next_state])
        
        td_error = reward + self.gamma * expected_q - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error