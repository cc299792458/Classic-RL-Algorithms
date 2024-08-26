import gymnasium as gym
import time

# Create the environment
env = gym.make("MountainCar-v0", render_mode="human")

state, info = env.reset()
done = False
while not done:
    # Select an action (0: push left, 1: do nothing, 2: push right)
    action = env.action_space.sample()  # Take a random action
    next_state, reward, terminated, truncated, info = env.step(action)
    env.render()
    state = next_state
    done = terminated or truncated
    time.sleep(0.05)

env.close()
