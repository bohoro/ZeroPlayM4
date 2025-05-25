import gymnasium as gym
import torch
import numpy as np
import os
from collections import deque
import matplotlib.pyplot as plt

# Ensure dql_agent.py is in the same directory or Python path
# This line assumes train_lunar_lander.py and dql_agent.py are in the same directory.
from dql_agent import DQNAgent, DQL_CONFIG


def train_lunar_lander():
    """
    Trains a DQNAgent on the LunarLander-v3 environment using dql_agent.py.
    Saves model checkpoints, final model, and a rewards plot in the 'results/' folder.
    """
    # --- 1. Setup ---
    print(f"Using DQL_CONFIG: {DQL_CONFIG}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the 'LunarLander-v3' environment
    # Note: 'LunarLander-v2' is the standard ID in Gymnasium.
    # If 'LunarLander-v3' does not exist, you might need to change this to 'LunarLander-v2'.
    # To visualize training (slower): env = gym.make('LunarLander-v3', render_mode='human')
    try:
        env = gym.make("LunarLander-v3")
    except gym.error.NameNotFound:
        print("Warning: 'LunarLander-v3' not found. Trying 'LunarLander-v2'.")
        try:
            env = gym.make("LunarLander-v2")
        except Exception as e:
            print(f"Error creating environment: {e}")
            return

    # Extract state_size and action_size from the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f"Environment: {env.spec.id if env.spec else 'Unknown'}")
    print(f"State size: {state_size}, Action size: {action_size}")

    # Initialize the DQNAgent using DQL_CONFIG
    agent = DQNAgent(state_size, action_size, DQL_CONFIG)

    # Create a directory named 'results/' if it doesn't already exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved in '{os.path.abspath(results_dir)}/'")

    # Hyperparameters from DQL_CONFIG
    num_episodes = DQL_CONFIG.get("num_episodes", 1000)
    max_t = DQL_CONFIG.get("max_t", 1000)  # Max steps per episode

    # For tracking scores
    scores_deque = deque(maxlen=100)  # Stores scores of last 100 episodes
    all_scores = []  # Stores scores of all episodes for plotting
    best_avg_score = -np.inf

    # --- 2. Training Loop ---
    print(f"\nStarting training for {num_episodes} episodes...")
    for i_episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        # Ensure state is a float32 numpy array (LunarLander usually provides float32)
        state = np.array(state, dtype=np.float32)
        episode_reward = 0

        for t in range(max_t):
            # Select an action using agent.select_action(state)
            action_tensor = agent.select_action(state)  # state is np.array
            action_scalar = action_tensor.item()  # Convert tensor to Python scalar int

            # Execute the action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action_scalar)
            done = terminated or truncated

            # Prepare next_state for replay buffer: np.array or None if terminal
            next_state_for_buffer = (
                np.array(next_state, dtype=np.float32) if not done else None
            )

            # Store the transition in the agent's replay buffer
            # agent.memory.push expects: state (np.array), action_tensor (torch.Tensor),
            # next_state (np.array/None), reward (float), done (bool)
            agent.memory.push(
                state, action_tensor, next_state_for_buffer, float(reward), done
            )

            # Update the current state
            state = np.array(next_state, dtype=np.float32)  # For the next iteration

            # Call agent.learn() to update the policy network
            agent.learn()

            episode_reward += reward

            if done:
                break

        scores_deque.append(episode_reward)
        all_scores.append(episode_reward)
        avg_score = np.mean(scores_deque)

        print(
            f"Episode {i_episode}\tReward: {episode_reward:.2f}\tAvg Score (100 ep): {avg_score:.2f}\tEpsilon: {agent.epsilon:.4f}\tSteps: {t+1}"
        )

        # --- 3. Saving Results ---
        # Periodically save the model checkpoint
        if i_episode % 100 == 0:
            checkpoint_path = os.path.join(
                results_dir, f"lunar_lander_checkpoint_episode_{i_episode}.pth"
            )
            agent.save(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # Save model if it's the best average score so far
        if avg_score > best_avg_score and len(scores_deque) >= 100:
            best_avg_score = avg_score
            best_model_path = os.path.join(results_dir, "lunar_lander_best_model.pth")
            agent.save(best_model_path)
            print(
                f"New best average score: {best_avg_score:.2f}. Model saved to {best_model_path}"
            )

        # Check for solving condition (LunarLander is considered solved if average reward is >= 200 over 100 consecutive trials)
        if avg_score >= 200.0 and len(scores_deque) >= 100:
            print(
                f"\nEnvironment solved in {i_episode} episodes!\tAverage Score: {avg_score:.2f}"
            )
            # Optionally, break training early
            # break

    # Save the final model
    final_model_path = os.path.join(results_dir, "lunar_lander_final.pth")
    agent.save(final_model_path)
    print(f"\nTraining finished. Final model saved to {final_model_path}")

    # Save scores and plot
    np.save(os.path.join(results_dir, "all_scores.npy"), np.array(all_scores))
    plt.figure(figsize=(12, 6))
    plt.plot(all_scores, label="Episode Reward")
    moving_avg = [
        np.mean(all_scores[max(0, i - 100) : i + 1]) for i in range(len(all_scores))
    ]
    plt.plot(moving_avg, label="Moving Average (100 episodes)", linestyle="--")
    plt.title(f'Training Rewards for {env.spec.id if env.spec else "LunarLander"}')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(results_dir, "rewards_plot.png")
    plt.savefig(plot_path)
    print(f"Rewards plot saved to {plot_path}")

    env.close()


if __name__ == "__main__":
    train_lunar_lander()
