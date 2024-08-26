"""
    N Step Sarsa
"""

import numpy as np
import gymnasium as gym

from tqdm import tqdm
from .sarsa import Sarsa

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
        # Calculate the sum of rewards with discounting
        G = 0
        for i in range(tau + 1, min(tau + self.n, T) + 1):
            G += self.gamma**(i - tau - 1) * rewards[i - 1]  # rewards[i - 1] because rewards are indexed from 0

        # If the trajectory does not terminate within n steps, add the estimated future value
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
