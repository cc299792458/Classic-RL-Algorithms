"""
    Dyna Maze

    This comes from the Example 8.1 of the RL Book.
"""

import numpy as np

from envs.grid_world import GridWorld

class DynaMaze(GridWorld):
    """
        DynaMaze extends GridWorld by adding walls as obstacles, making certain cells inaccessible to the agent. 
        The agent must find a path from the start position to the goal while navigating around these walls. 
        The walls can be modified dynamically to simulate changing environments.
    """
    def __init__(self, height=6, width=9, walls=None, max_episode_length=False, start_position=(0, 0), goal_position=None):
        """
             Initialize the DynaMaze environment.
        """
        self.walls = walls if walls is not None else []

        super().__init__(height=height, 
                         width=width, 
                         max_episode_length=max_episode_length, 
                         start_position=start_position, 
                         goal_position=goal_position)

    def _create_transition_matrix(self):
        """
        Create the transition matrix P[state][action] -> [(probability, next_state, reward, terminated)]
        without considering the walls, i.e., no transitions from wall states.
        """
        P = {}
        for x in range(self.height):
            for y in range(self.width):
                state = x * self.width + y
                if (x, y) in self.walls:
                    continue  # Skip creating transitions for wall positions
                
                P[state] = {a: [] for a in range(self.action_space.n)}

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
                        next_state = state  # Stay in the same position if hitting a wall
                    else:
                        next_state = new_x * self.width + new_y

                    terminated = self.is_terminal_state(next_state)
                    reward = -1  # Constant reward for each step

                    # Add this transition to the P dictionary
                    P[state][action].append((1.0, next_state, reward, terminated))
                    
        return P

    def render(self):
        """
        Render the maze with the following symbols:
        S: Start position
        G: Goal position
        W: Wall
        A: Agent's current position
        .: Empty space
        """
        grid = np.full((self.height, self.width), '.', dtype=str)

        # Mark walls with 'W'
        for (x, y) in self.walls:
            grid[x, y] = 'W'

        # Mark the start position with 'S'
        start_x, start_y = self._state_to_xy(self.start_state)
        grid[start_x, start_y] = 'S'

        # Mark the goal position with 'G'
        for goal_state in self.goal_state:
            goal_x, goal_y = self._state_to_xy(goal_state)
            grid[goal_x, goal_y] = 'G'

        # Mark the current position with 'A'
        x, y = self._state_to_xy(self.position)
        grid[x, y] = 'A'

        # Print the grid
        for row in grid:
            print(' '.join(row))

    def set_walls(self, walls):
        """
        Set new wall positions and rebuild the transition matrix.
        """
        self.walls = walls
        self.P = self._create_transition_matrix()

if __name__ == '__main__':
    # Example of using the DynaMaze environment
    walls = [(0, 7), (1, 2), (1, 7), (2, 2), (2, 7), (3, 2), (4, 5)]  # Example wall positions
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
