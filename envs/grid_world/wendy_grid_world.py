import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from envs.grid_world import GridWorld

class WindyGridWorld(GridWorld):
    def __init__(self, width=10, height=7, wind_strength=None):
        super().__init__(width=width, height=height)

        if wind_strength is None:
            self.wind_strength = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        else:
            self.wind_strength = wind_strength

        self.goal_state = (3, 7)

    def _calculate_next_state(self, action):
        x, y = divmod(self.position, self.width)

        if action == 0:  # Up
            dx, dy = -1, 0
        elif action == 1:  # Down
            dx, dy = 1, 0
        elif action == 2:  # Right
            dx, dy = 0, 1
        elif action == 3:  # Left
            dx, dy = 0, -1
        else:
            dx, dy = 0, 0

        # Apply wind effect
        dx -= self.wind_strength[y]

        new_x = max(min(x + dx, self.height - 1), 0)
        new_y = max(min(y + dy, self.width - 1), 0)

        return new_x, new_y

    def step(self, action):
        self.current_step += 1
        new_x, new_y = self._calculate_next_state(action)
        self.position = new_x * self.width + new_y

        terminated = self.is_goal_state(new_x, new_y)
        reward = -1

        truncated = self.current_step >= self.max_episode_length
        info = {'terminal': terminated}

        return self.position, reward, terminated, truncated, info

    def is_goal_state(self, x, y):
        return (x, y) == self.goal_state

    def reset(self):
        self.position = 3 * self.width
        self.current_step = 0
        return self.position, {}

def animate_trajectory(env, trajectory, grid_size, goal_state):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set up grid
    ax.set_xlim(-0.5, grid_size[1] - 0.5)
    ax.set_ylim(grid_size[0] - 0.5, -0.5)
    ax.set_xticks(np.arange(grid_size[1]))
    ax.set_yticks(np.arange(grid_size[0]))
    ax.set_xticklabels(np.arange(grid_size[1]))
    ax.set_yticklabels(np.arange(grid_size[0]))
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

    # Plot goal
    ax.scatter(goal_state[1], goal_state[0], color='r', marker='*', s=200, label='Goal')

    # Initialize plot elements
    line, = ax.plot([], [], 'k--', lw=2)  # Path as a dashed line
    point, = ax.plot([], [], 'bo', markersize=10)  # Current position as a solid dot

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(frame):
        xdata, ydata = [], []
        for (state, action) in trajectory[:frame+1]:
            x, y = divmod(state, env.width)
            xdata.append(y)
            ydata.append(x)

        if len(xdata) > 0 and len(ydata) > 0:
            line.set_data(xdata[:-1], ydata[:-1])
            point.set_data([xdata[-1]], [ydata[-1]])  # Wrap in list to ensure sequence
        return line, point

    ani = FuncAnimation(fig, update, frames=len(trajectory), init_func=init, blit=True, interval=50, repeat=False)
    plt.show()

if __name__ == '__main__':
    env = WindyGridWorld()

    # Generate a trajectory
    state, _ = env.reset()
    trajectory = []
    done = False

    while not done:
        action = env.action_space.sample()  # Randomly sample an action
        next_state, reward, terminated, truncated, info = env.step(action)
        trajectory.append((state, action))
        state = next_state
        done = terminated or truncated

    # Animate the trajectory
    animate_trajectory(env, trajectory, grid_size=(env.height, env.width), goal_state=env.goal_state)
