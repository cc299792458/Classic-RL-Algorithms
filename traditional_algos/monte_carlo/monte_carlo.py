"""
    Monte Carlo Method
"""

import numpy as np
import gymnasium as gym

from tqdm import tqdm
from utils.gym_utils import get_observation_shape

class MonteCarlo:
    def __init__(self, env: gym.Env, gamma=1.0, epsilon=0.1, initial_policy=None, first_visit=True) -> None:
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.first_visit = first_visit
        self.observation_shape = get_observation_shape(env.observation_space)
        self.num_action = self.env.action_space.n

        self.initial_policy = initial_policy
        self.reset()

    def reset(self):
        """Reset the policy to a uniform random policy and reset the value and Q functions."""
        if self.initial_policy is None:
            self.policy = np.ones((*self.observation_shape, self.num_action)) / self.num_action
        else:
            self.policy = self.initial_policy

        self.value_function = np.zeros(self.observation_shape)
        self.Q = np.zeros((*self.observation_shape, self.num_action))  # Action-value function

        # For incremental update
        self.state_counts = np.zeros(self.observation_shape)
        self.action_counts = np.zeros((*self.observation_shape, self.num_action))
        # For Monte Carlo, we need to track returns
        # self.returns = {state: [] for state in np.ndindex(self.observation_shape)}
        # self.returns_q = {state: {action: [] for action in range(self.num_action)} for state in np.ndindex(self.observation_shape)}

    def prediction(self, num_episode):
        """Monte Carlo prediction for estimating the value function."""
        self.reset()
        with tqdm(total=num_episode) as pbar:
            for _ in range(num_episode):
                episode = self.generate_episode()
                self.update_value_function(episode)
                pbar.update(1)

    def estimation_and_control(self, num_episode):
        """Monte Carlo estimation for estimating the Q function and policy."""
        self.reset()
        with tqdm(total=num_episode) as pbar:
            for _ in range(num_episode):
                episode = self.generate_episode()
                self.update_q_function(episode)
                self.improve_policy()
                pbar.update(1)

    def generate_episode(self):
        """Generate an episode following the current policy."""
        episode = []
        state, info = self.env.reset()
        done = False

        while not done:
            action = np.random.choice(np.arange(self.num_action), p=self.policy[state])
            next_state, reward, terminated, truncated, info = self.env.step(action=action)
            done = terminated or truncated
            episode.append((state, action, reward))
            state = next_state

        return episode

    def update_value_function(self, episode):
        """Incrementally update the value function based on the generated episode."""
        G = 0

        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            
            if (self.first_visit and not any(state==x[0] for x in episode[:episode.index((state, action, reward))])) or not self.first_visit:
                # Increment the visit count
                self.state_counts[state] += 1
                
                # Calculate the step size (learning rate)
                alpha = 1.0 / self.state_counts[state]
                
                # Incremental update of V(s)
                self.value_function[state] += alpha * (G - self.value_function[state])

    def update_q_function(self, episode):
        """Incrementally update the Q function based on the generated episode."""
        G = 0

        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            
            if (self.first_visit and not any(state==x[0] and action==x[1] for x in episode[:episode.index((state, action, reward))])) or not self.first_visit:
                # Increment the visit count
                self.action_counts[state][action] += 1
                
                # Calculate the step size (learning rate)
                alpha = 1.0 / self.action_counts[state][action]
                
                # Incremental update of Q(s, a)
                self.Q[state][action] += alpha * (G - self.Q[state][action])

    # def update_value_function(self, episode):
    #     """
    #         Update the value function based on the generated episode.
    #     """
    #     G = 0
    #     for state, action, reward in reversed(episode):
    #         G = self.gamma * G + reward
    #         # if hasattr(self.env, 'is_terminal_state') and self.env.is_terminal_state(state):
    #         #     continue
    #         # Only episodes with first-visited states are used for updating value function
    #         if not any(state==x[0] for x in episode[:episode.index((state, action, reward))]):
    #             self.returns[state].append(G)
    #             self.value_function[state] = np.mean(self.returns[state])
    

    # def update_q_function(self, episode):
    #     """
    #         Update the Q function based on the generated episode.
    #     """
    #     G = 0
    #     for state, action, reward in reversed(episode):
    #         G = self.gamma * G + reward
    #         # if hasattr(self.env, 'is_terminal_state') and self.env.is_terminal_state(state):
    #         #     continue
    #         if not any(state==x[0] and action==x[1] for x in episode[:episode.index((state, action, reward))]):
    #             self.returns_q[state][action].append(G)
    #             self.Q[state][action] = np.mean(self.returns_q[state][action])

    def improve_policy(self):
        """Update the policy based on the updated Q function."""
        tolerance = 1e-8  # A small tolerance value for comparisons

        for state in np.ndindex(self.observation_shape):
            max_q_value = np.max(self.Q[state])
            best_actions = np.argwhere(np.abs(self.Q[state] - max_q_value) <= tolerance).flatten()

            # Update the policy to give equal probability to these best actions
            self.policy[state] = np.zeros(self.num_action)
            self.policy[state][best_actions] = 1.0 / len(best_actions)
            
            # Implement ε-greedy exploration
            if self.epsilon > 0:
                self.policy[state] = (1 - self.epsilon) * self.policy[state] + (self.epsilon / self.num_action)

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


class OffPolicyMonteCarlo(MonteCarlo):
    """Implement Off-Policy Monte Carlo via Importance Sampling."""
    pass
