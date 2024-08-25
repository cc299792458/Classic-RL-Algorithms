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

class NStepSarsa(Sarsa):
    def __init__(self, env: gym.Env, gamma=1, epsilon=0.1, alpha=0.1, initial_policy=None, n=1) -> None:
        super().__init__(env, gamma, epsilon, alpha, initial_policy)
        self.n = n  # Number of steps to back up

    def estimation_and_control(self, num_episode):
        self.reset()
        with tqdm(total=num_episode) as pbar:
            for _ in range(num_episode):
                state, info = self.env.reset()
                done = False
                action = np.random.choice(np.arange(self.num_action), p=self.policy[state])
                
                # Initialize the lists to store the episode trajectory
                states = [state]
                actions = [action]
                rewards = []
                
                t = 0  # Time step.
                T = np.inf  # Time when the episode ends
                
                while True:
                    if t < T:
                        next_state, reward, terminated, truncated, info = self.env.step(action=actions[-1])
                        rewards.append(reward)
                        done = terminated or truncated

                        if done:
                            T = t + 1
                        else:
                            states.append(next_state)
                            next_action = np.random.choice(np.arange(self.num_action), p=self.policy[next_state])
                            actions.append(next_action)
                    
                    tau = t - self.n + 1  # Time whose estimate is being updated
                    
                    if tau >= 0:
                        # Update Q-function using n-step return
                        self.update_q_function(states, actions, rewards, tau, T)

                    if tau == T - 1:
                        break

                    t += 1
                    if t < T:
                        action = actions[-1]

                pbar.update(1)

    def update_q_function(self, states, actions, rewards, tau, T):
        """
        Update the Q-function based on the n-step return G(tau).
        
        :param states: List of states in the trajectory.
        :param actions: List of actions in the trajectory.
        :param rewards: List of rewards in the trajectory.
        :param tau: Time whose estimate is being updated.
        :param T: Time when the episode ends.
        """
        # Compute G(tau) with n-step backup
        G = sum([self.gamma**(i-tau-1) * rewards[i] for i in range(tau+1, min(tau+self.n, T) + 1)])
        if tau + self.n < T:
            G += self.gamma**self.n * self.Q[states[tau + self.n]][actions[tau + self.n]]
        
        state_tau = states[tau]
        action_tau = actions[tau]
        
        # Update the Q-function for state_tau and action_tau
        self.Q[state_tau][action_tau] += self.alpha * (G - self.Q[state_tau][action_tau])

        # Improve the policy based on the updated Q-function
        self.improve_policy(state_tau)

    def set_n(self, n):
        self.n = n
