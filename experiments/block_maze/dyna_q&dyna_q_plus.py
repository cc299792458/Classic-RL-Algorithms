"""
    Use Dyna Q and Dyna Q+ to solve Block Maze
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.misc_utils import set_seed
from envs.grid_world import DynamicMaze
from traditional_algos.planing_and_learning_methods import DynaQ, DynaQPlus

class DynaQWithLogging(DynaQ):
    def estimation_and_control(self, total_timesteps):
        self.reset()
        cumulative_rewards = []
        total_reward = 0
        timesteps = 0  # Track total timesteps

        while timesteps < total_timesteps:
            state, _ = self.env.reset()
            done = False
            while not done and timesteps < total_timesteps:
                action = np.random.choice(np.arange(self.num_action), p=self.policy[state])
                next_state, reward, terminated, truncated, info = self.env.step(action=action)
                done = terminated or truncated

                # Update Q function and model
                self.update_q_function(state, action, reward, next_state)
                self.model[(state, action)] = (next_state, reward)

                # Update policy
                self.improve_policy(state=state)
                
                # Simulate additional steps
                self.simulate()

                state = next_state
                timesteps += 1
                total_reward += reward  # Accumulate rewards for this episode

                # Record cumulative reward at every timestep
                cumulative_rewards.append((timesteps, total_reward))

        return cumulative_rewards
    
class DynaQPlusWithLogging(DynaQPlus):
    def estimation_and_control(self, total_timesteps):
        self.reset()
        cumulative_rewards = []
        total_reward = 0
        timesteps = 0  # Track total timesteps

        while timesteps < total_timesteps:
            state, _ = self.env.reset()
            done = False
            while not done and timesteps < total_timesteps:
                action = np.random.choice(np.arange(self.num_action), p=self.policy[state])
                next_state, reward, terminated, truncated, info = self.env.step(action=action)
                done = terminated or truncated

                # Update Q function and model
                self.update_q_function(state, action, reward, next_state)
                self.model[(state, action)] = (next_state, reward)

                # Update policy
                self.improve_policy(state=state)
                
                # Simulate additional steps
                self.simulate()

                state = next_state
                timesteps += 1
                total_reward += reward  # Accumulate rewards for this episode

                # Record cumulative reward at every timestep
                cumulative_rewards.append((timesteps, total_reward))

        return cumulative_rewards

if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))

    ##### Create the Environment and the Agent #####
    # Block Maze setup
    height = 6
    width = 9
    original_walls = [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)]  # Initial wall positions
    new_walls = [(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8)]  # Walls after change
    change_time_step = 1000
    start_position = (5, 3)
    goal_position = (0, 8)

    print("The Block Maze environment")
    env = DynamicMaze(
        height=height, 
        width=width, 
        original_walls=original_walls, 
        new_walls=new_walls, 
        change_time_step=change_time_step, 
        max_episode_length=False, 
        start_position=start_position, 
        goal_position=goal_position
    )
    env.render()

    ##### Solve the Problem using DynaQ and DynaQPlus #####
    total_timesteps = 3000  # Set the total timesteps limit
    num_runs = 5

    agents = {
        "DynaQ": DynaQWithLogging(env=env, gamma=0.95, epsilon=0.1, alpha=0.1, planning_steps=5),
        # "DynaQPlus": DynaQPlusWithLogging(env=env, gamma=0.95, epsilon=0.1, alpha=0.1, planning_steps=5)
    }

    cumulative_rewards_results = {}
    
    for agent_name, agent in agents.items():
        print(f"Running {agent_name}")
        all_cumulative_rewards = []
        with tqdm(total=num_runs, desc=f"{agent_name}") as pbar:
            for _ in range(num_runs):
                env.reset(reset_timestep=True)
                cumulative_rewards = agent.estimation_and_control(total_timesteps=total_timesteps)
                all_cumulative_rewards.extend(cumulative_rewards)  # Accumulate rewards across runs
                pbar.update(1)
            
        cumulative_rewards_results[agent_name] = all_cumulative_rewards

    ##### Plot Cumulative Rewards vs. Total Timesteps for DynaQ and DynaQPlus
    plt.figure(figsize=(12, 8))

    for agent_name, cumulative_rewards in cumulative_rewards_results.items():
        timesteps, rewards = zip(*cumulative_rewards)
        plt.plot(timesteps, rewards, label=agent_name)

    plt.xlabel('Total Timesteps')
    plt.ylabel('Cumulative Rewards')
    plt.title('Cumulative Rewards vs. Total Timesteps for DynaQ and DynaQPlus in Block Maze')
    plt.legend()
    plt.grid(True)

    # Save and show the plot
    plot_path = os.path.join(log_dir, 'block_maze_cumulative_rewards_vs_timesteps.png')
    plt.savefig(plot_path)
    plt.show()
