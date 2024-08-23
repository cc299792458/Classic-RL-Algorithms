import gymnasium as gym

env = gym.make('CliffWalking-v0')
state, info = env.reset()

done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    state = next_state
    done = terminated or truncated

print(f"Episode finished with total reward: {total_reward}")