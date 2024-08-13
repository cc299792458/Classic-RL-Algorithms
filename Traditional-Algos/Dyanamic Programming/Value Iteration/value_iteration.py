import numpy as np
import gymnasium as gym

from envs.grid_world import GridWorld

class ValueIteration:
    def __init__(self, env: gym.Env, gamma=1.0, theta=1e-6) -> None:
        self.env = env
        self.gamma = gamma
        self.theta = theta  # Convergence threshold
        self.num_state = self.env.observation_space.n

        self.reset()

    def reset(self):
        """Reset the policy to a uniform random policy and reset the value function."""
        self.policy = np.ones([self.num_state, self.env.action_space.n]) / self.env.action_space.n
        self.value_function = np.zeros(self.num_state)

    def iterate(self, inplace=True, print_each_iter=False, print_result=False):
        iteration = 0
        tolerance = 1e-8
        while True:
            iteration += 1
            delta = 0

            if not inplace:
                new_value_function = self.value_function.copy()  # Use copy to avoid modifying the original directly

            for state in range(self.num_state):
                if self.env.is_terminal_state(state):
                    continue  # Skip terminal states
            
                q = np.zeros(self.env.action_space.n)
                for action in range(self.env.action_space.n):
                    for prob, next_state, reward, done in self.env.P[state][action]:
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
                for state in range(self.num_state):
                    if self.env.is_terminal_state(state):
                        continue  # Skip terminal states

                    q = np.zeros(self.env.action_space.n)
                    for action in range(self.env.action_space.n):
                        for prob, next_state, reward, done in self.env.P[state][action]:
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
        if isinstance(self.env, GridWorld):
            self._print_gridworld_value_function()
        else:
            print(self.value_function)

    def print_policy(self):
        if isinstance(self.env, GridWorld):
            self._print_gridworld_policy()
        else:
            print(self.policy)

    def _print_gridworld_value_function(self):
        grid_size = self.env.grid_size
        max_value = max(self.value_function)
        min_value = min(self.value_function)
        max_width = max(len(f"{max_value:.2f}"), len(f"{min_value:.2f}"))  # Ensure consistent width for alignment

        for i in range(grid_size):
            row_values = []
            for j in range(grid_size):
                value = self.value_function[i * grid_size + j]
                row_values.append(f"{value:>{max_width}.2f}")  # Align values by setting a consistent column width
            print(" | ".join(row_values))  # Use " | " as a separator between columns
            if i < grid_size - 1:
                print("-" * (max_width * grid_size + (grid_size - 1) * 3))  # Print separator line between rows

    def _print_gridworld_policy(self):
        grid_size = self.env.grid_size
        policy_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}  # Corrected action to arrow mapping
        max_width = 4  # Define a fixed width for each cell to accommodate up to 4 arrows
        
        for i in range(grid_size):
            row_policy = []
            for j in range(grid_size):
                state = i * grid_size + j
                if self.env.is_terminal_state(state):
                    cell_content = 'T'.center(max_width)  # Center 'T' within the fixed width
                else:
                    best_actions = np.argwhere(self.policy[state] == np.max(self.policy[state])).flatten()
                    arrows = ''.join([policy_arrows[action] for action in best_actions])
                    cell_content = arrows.center(max_width)  # Center arrows within the fixed width
                row_policy.append(cell_content)
            
            # Join the row content with ' | ' separator
            print(" | ".join(row_policy))
            
            if i < grid_size - 1:
                print("-" * (max_width * grid_size + (grid_size - 1) * 3))  # Print separator line between rows



if __name__ == '__main__':
    ##### Step 0: Build Environment and Initiate Policy #####
    env = GridWorld()

    theta = 1e-4
    agent = ValueIteration(env=env, theta=theta)

    ##### Step 1: Get the Optimal Policy using Value Iteration #####
    ### Step 1.2 Record iterations ###
    agent.reset()
    agent.iterate(print_each_iter=True, print_result=True)  # 4 iterations
    
    ### Step 1.2 Record time consuming ###
    import time
    agent.reset()
    start_time = time.time()
    agent.iterate()
    end_time = time.time()
    print(f"Time consuming: {end_time-start_time}") # about 8e-4 seconds