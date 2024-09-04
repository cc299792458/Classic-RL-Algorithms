"""
    Dyna Q
"""

import numpy as np
import gymnasium as gym
from tqdm import tqdm
from utils.gym_utils import get_observation_shape

# TODO: Why timesteps are different in the episode 0?

class DynaQ:
    """
        DynaQ with a deterministic model
    """
    def __init__(self, env: gym.Env, gamma=1.0, epsilon=0.1, alpha=0.1, planning_steps=1, initial_policy=None) -> None:
        self.env = env
        self.gamma = gamma 
        self.epsilon = epsilon
        self.alpha = alpha 
        self.planning_steps = planning_steps  # Simulation times
        self.observation_shape = get_observation_shape(env.observation_space)
        self.num_action = self.env.action_space.n

        self.initial_policy = initial_policy
        self.reset()

    def reset(self):
        if self.initial_policy is None:
            self.policy = np.ones((*self.observation_shape, self.num_action)) / self.num_action
        else:
            self.policy = self.initial_policy
        self.Q = np.zeros((*self.observation_shape, self.num_action))  # Action-value function
        self.model = {}  # Dictionary to store a deterministic model: (state, action) -> (next_state, reward)

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
                    self.update_q_function(state, action, reward, next_state)
                    
                    # Store the experience in the model
                    self.model[(state, action)] = (next_state, reward)

                    # Update policy based on new Q values
                    self.improve_policy(state) 

                    # Simulate additional learning steps using the model
                    self.simulate()

                    # Move to the next state
                    state = next_state

                pbar.update(1)
    
    def update_q_function(self, state, action, reward, next_state):
        td_error = reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

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

    def simulate(self):
        # Perform `self.planning_steps` simulations using the stored model
        for _ in range(self.planning_steps):
            if not self.model:
                continue

            # Randomly pick a state and action from the model
            state, action = list(self.model.keys())[np.random.randint(len(self.model))]
            next_state, reward = self.model[(state, action)]

            # Update Q-function using simulated experience
            self.update_q_function(state, action, reward, next_state)

            # Improve policy based on the updated Q-function
            self.improve_policy(state)
    
    def set_planning_steps(self, planning_steps):
        self.planning_steps = planning_steps
