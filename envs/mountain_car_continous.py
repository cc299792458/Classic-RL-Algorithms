import time
import gymnasium as gym

# Create the environment
env = gym.make("MountainCarContinuous-v0", render_mode="human")

state, info = env.reset()
done = False
while not done:
    # Select an action
    action = env.action_space.sample()  # Take a random action
    next_state, reward, terminated, truncated, info = env.step(action)
    env.render()
    state = next_state
    done = terminated or truncated
    time.sleep(0.05)

env.close()