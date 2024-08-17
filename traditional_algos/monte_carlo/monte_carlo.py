import numpy as np
import gymnasium as gym

from envs import GridWorld

class MonteCarlo:
    def __init__(self, env: gym.Env, gamma=1.0, epsilon=0.0) -> None:
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_state = self.env.observation_space.n
        self.num_action = self.env.action_space.n

        self.reset()

    def reset(self):
        """
            Reset the policy to a uniform random policy and reset the value and Q functions.
        """
        self.policy = np.ones([self.num_state, self.num_action]) / self.num_action
        self.value_function = np.zeros(self.num_state)
        self.Q = np.zeros((self.num_state, self.num_action))  # Action-value function
        # For Monte Carlo, we need to track returns
        self.returns = {state: [] for state in range(self.num_state)}
        self.returns_q = {state: {action: [] for action in range(self.num_action)} for state in range(self.num_state)}

    def prediction(self, num_episode):
        """
            First-vist monte-carlo prediction, for estimating value function.
            P.S, the name comes from chapter 5.1 of the RL book, monte carlo prediction
        """
        self.reset()
        for _ in range(num_episode):
            episode = self.generate_episode()
            self.update_value_function(episode)
    
    def estimation_and_control(self, num_episode):
        """
            First-visit monte-carlo esimation, for estimating Q function and policy pi.
            P.S, the name comes from chapter 5.2 and 5.3 of the RL book, 
            monte carlo esimation and monte carlo control
        """
        self.reset()
        for _ in range(num_episode):
            episode = self.generate_episode()
            self.update_q_function(episode)
            self.improve_policy()

    def generate_episode(self):
        """
            Generate an episode following the current policy.
        """
        episode = []
        state = self.env.reset()
        done = False
        while not done:
            action = np.random.choice(np.arange(self.num_action), p=self.policy[state])
            next_state, reward, done, info = self.env.step(action=action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def update_value_function(self, episode):
        """
            Update the value function based on the generated episode.
        """
        G = 0
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            if self.env.is_terminal_state(state):
                continue
            # Only episodes with first-visited states are used for updating value function
            if not any(state==x[0] for x in episode[:episode.index((state, action, reward))]):
                self.returns[state].append(G)
                self.value_function[state] = np.mean(self.returns[state])
    
    def update_q_function(self, episode):
        """
            Update the Q function based on the generated episode.
        """
        G = 0
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward
            if self.env.is_terminal_state(state):
                continue
            if not any(state==x[0] and action==x[1] for x in episode[:episode.index((state, action, reward))]):
                self.returns_q[state][action].append(G)
                self.Q[state][action] = np.mean(self.returns_q[state][action])

    def improve_policy(self):
        tolerance = 1e-8    # This is a too small tolerance
        for state in range(self.num_state):
            if self.env.is_terminal_state(state):
                continue  # Skip terminal states
            # Find the best actions with Q-values close to the maximum
            max_q_value = np.max(self.Q[state])
            best_actions = np.argwhere(np.abs(self.Q[state] - max_q_value) <= tolerance).flatten()

            # Update the policy to give equal probability to these best actions
            self.policy[state] = np.zeros(self.num_action)
            self.policy[state][best_actions] = 1.0 / len(best_actions)
            
            # Implement ε-greedy exploration
            if self.epsilon > 0:
                self.policy[state] = (1 - self.epsilon) * self.policy[state] + (self.epsilon / self.num_action)

    def print_value_function(self):
        if hasattr(self.env, '_print_value_function'):
            self.env._print_value_function(self.value_function)
        else:
            print(self.value_function)

    def print_q_function(self):
        if hasattr(self.env, '_print_q_function'):
            self.env._print_q_function(self.Q)
        else:
            print(self.Q)

    def print_policy(self):
        if hasattr(self.env, '_print_policy'):
            self.env._print_policy(self.policy)
        else:
            print(self.policy)