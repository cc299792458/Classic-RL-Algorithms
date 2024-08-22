"""
    Black Jack from OpenAI Gymnasium
"""

import gymnasium as gym

if __name__ == '__main__':
    env = gym.make('Blackjack-v1', natural=False, sab=True)

    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    initial_state = []
    for i in range(10):
        state = env.reset()
        initial_state.append(state)
    print(f"Sampled initial State: {initial_state}")

    state = env.reset()
    print(f"Initial state: {state}")
    done = False
    while not done:
        # Randomly choose to hit (1) or stick (0)
        action = env.action_space.sample()
        state, reward, done, _, _ = env.step(action)

        print(f"Action Taken: {'Hit' if action == 1 else 'Stick'}")
        print(f"New State: {state}, Reward: {reward}, Done: {done}")

    print(f"Final Reward: {reward}")

    # Close the environment
    env.close()