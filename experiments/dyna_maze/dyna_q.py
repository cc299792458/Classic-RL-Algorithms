"""
    Use Dyna Q to solve Dyna Maze
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.misc_utils import set_seed
from envs.grid_world import DynaMaze
from traditional_algos.planing_and_learning_methods import DynaQ

class DynaQWithLogging(DynaQ):
    def estimation_and_control(self, num_episode):
        self.reset()
        episode_lengths = []
        policy_update_counts = []
        timesteps = 0
        cumulative_policy_updates = 0  # Initialize cumulative policy updates

        for _ in range(num_episode):
            state, _ = self.env.reset()
            done = False
            while not done:
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
                cumulative_policy_updates += (1 + self.planning_steps)  # Increment the cumulative policy update counter

            episode_lengths.append(timesteps)
            policy_update_counts.append(cumulative_policy_updates)  # Record cumulative updates
            timesteps = 0  # Reset timesteps for the next episode

        return episode_lengths, policy_update_counts
    
if __name__ == '__main__':
    set_seed()
    log_dir = os.path.dirname(os.path.abspath(__file__))
    ##### Create the Environment and the Agent #####
    # Create the environment
    height = 6
    width = 9
    start_position = (2, 0)
    goal_position = (0, 8)
    walls = [(0, 7), (1, 2), (1, 7), (2, 2), (2, 7), (3, 2), (4, 5)]

    print("The Dyna Maze environment")
    env = DynaMaze(
        height=height, 
        width=width, 
        walls=walls, 
        max_episode_length=False, 
        start_position=start_position, 
        goal_position=goal_position
    )
    env.render()

    # Create the agent
    agent = DynaQWithLogging(env=env, gamma=0.95, epsilon=0.1, alpha=0.1)

    ##### Solve the Problem with Different Planning Steps #####
    num_episodes = 50
    num_runs = 30

    planning_steps_list = [0, 3, 5, 10, 20, 35, 50]
    timesteps_per_episode = {}
    policy_updates = {}
    
    for planning_steps in planning_steps_list:
        print(f"Running DynaQ with planning_steps={planning_steps}")
        all_episode_lengths = []
        all_policy_updates = []
        agent.set_planning_steps(planning_steps=planning_steps)
        with tqdm(total=num_runs, desc=f"Planning steps = {planning_steps}") as pbar:
            for _ in range(num_runs):
                episode_lengths, policy_update_counts = agent.estimation_and_control(num_episode=num_episodes)
                all_episode_lengths.append(episode_lengths)
                all_policy_updates.append(policy_update_counts)
                pbar.update(1)
            
        timesteps_per_episode[planning_steps] = np.mean(all_episode_lengths, axis=0)
        policy_updates[planning_steps] = np.mean(all_policy_updates, axis=0)

    ##### Plot Timesteps per Episode and Cumulative Policy Updates
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 8))

    color_map = ['tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:red']

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Timesteps per Episode', color='tab:blue')
    
    # Create only one ax2 for cumulative policy updates
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulative Policy Updates', color='tab:red')

    for idx, planning_steps in enumerate(planning_steps_list):
        color = color_map[idx % len(color_map)]
        
        # Plot Timesteps per Episode on ax1
        ax1.plot(range(num_episodes), timesteps_per_episode[planning_steps], label=f'Timesteps (planning_steps={planning_steps})', color=color)
        
        # Plot Cumulative Policy Updates on ax2
        ax2.plot(range(num_episodes), policy_updates[planning_steps], linestyle='--', label=f'Policy Updates (planning_steps={planning_steps})', color=color)

    # Set the tick parameters for both axes
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()  # Make sure everything fits without overlap
    plt.title(f'Dyna Q\nTimesteps and Cumulative Policy Updates vs. Episode for Different Planning Steps')
    plt.grid(True)

    # Show the legend for both axes
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)

    # Save and show the plot
    plot_path = os.path.join(log_dir, f'dyna_maze_timesteps_vs_policy_updates_all.png')
    plt.savefig(plot_path)
    plt.show()
