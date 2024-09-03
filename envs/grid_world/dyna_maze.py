import numpy as np
import gymnasium as gym

from .grid_world import GridWorld

class DynaMaze(GridWorld):
    def __init__(self, width=9, height=6, walls=None, max_episode_length=False):
        super().__init__(width, height, max_episode_length)
        
        # Set walls if provided, otherwise default to an empty list
        self.walls = walls if walls is not None else []
        
        # Rebuild the transition matrix with walls
        self.P = self._create_transition_matrix()

    def _create_transition_matrix(self):
        """
        Create the transition matrix P[state][action] -> [(probability, next_state, reward, terminated)]
        considering the walls.
        """
        P = {}
        for x in range(self.height):
            for y in range(self.width):
                state = x * self.width + y
                P[state] = {a: [] for a in range(self.action_space.n)}
                
                if (x, y) in self.walls:
                    # If the current position is a wall, the agent cannot enter it
                    for action in range(self.action_space.n):
                        P[state][action].append((1.0, state, -1, False))
                else:
                    # Define possible transitions for each action
                    for action in range(self.action_space.n):
                        new_x, new_y = x, y
                        if action == 0:  # up
                            new_x = max(x - 1, 0)
                        elif action == 1:  # down
                            new_x = min(x + 1, self.height - 1)
                        elif action == 2:  # right
                            new_y = min(y + 1, self.width - 1)
                        elif action == 3:  # left
                            new_y = max(y - 1, 0)

                        # Check if the next position is a wall
                        if (new_x, new_y) in self.walls:
                            new_x, new_y = x, y  # Stay in the same position

                        next_state = new_x * self.width + new_y
                        terminated = self.is_terminal_state(next_state)
                        reward = -1  # Constant reward for each step

                        # Add this transition to the P dictionary
                        P[state][action].append((1.0, next_state, reward, terminated))
                    
        return P

    def set_walls(self, walls):
        """
        Set new wall positions and rebuild the transition matrix.
        """
        self.walls = walls
        self.P = self._create_transition_matrix()

if __name__ == '__main__':
    # Example of using the DynaMaze environment
    walls = [(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)]  # Example wall positions
    env = DynaMaze(walls=walls)
    env.reset()
    env.render()
    state, reward, terminated, truncated, info = env.step(2)  # Example action: move right
    env.render()
    print(f"Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

    # Change the wall positions
    new_walls = [(1, 1), (2, 1), (3, 1)]
    env.set_walls(new_walls)
    env.reset()
    env.render()
