import gymnasium as gym
import torch
import numpy as np
import os
import argparse
import sys

# Add the project root directory to the Python path to allow imports from 'core'
# Assumes the 'scripts' directory is directly under the project root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from the 'core' directory
from core.dql_agent import DQNAgent, DQL_CONFIG


def test_lunar_lander(model_path: str, num_episodes: int = 10, render: bool = True):
    """
    Tests a trained DQNAgent on the LunarLander-v2 environment.

    Args:
        model_path (str): Path to the saved model state_dict (.pth file).
        num_episodes (int): Number of episodes to run for testing.
        render (bool): Whether to render the environment.
    """
    # --- 1. Setup ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the 'LunarLander-v2' environment
    render_mode = "human" if render else None
    try:
        env = gym.make("LunarLander-v3", render_mode=render_mode)
    except Exception as e:
        print(f"Error creating LunarLander-v3 environment: {e}")
        print(
            "Please ensure you have 'box2d-py' installed (pip install gymnasium[box2d])."
        )
        return

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f"Environment: LunarLander-v3")
    print(f"State size: {state_size}, Action size: {action_size}")

    # Initialize the DQNAgent using DQL_CONFIG for network architecture
    # The DQL_CONFIG should match the one used during training for hidden_size etc.
    agent = DQNAgent(state_size, action_size, DQL_CONFIG)

    # Load the trained model weights
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        env.close()
        return

    try:
        agent.load(model_path)  # agent.load also sets policy_net.eval()
        print(f"Model loaded successfully from '{model_path}'")
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return

    # Ensure the agent is in evaluation mode (already done by agent.load, but good for clarity)
    agent.policy_net.eval()

    total_rewards = []
    max_steps_per_episode = int(DQL_CONFIG.get(
        "max_t", 1000
    ))  # Max steps per episode from config

    # --- 2. Testing Loop ---
    print(f"\nStarting testing for {num_episodes} episodes...")
    for i_episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32)
        episode_reward = 0

        for t in range(max_steps_per_episode):
            # Select action greedily using the policy network directly
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                action_values = agent.policy_net(state_tensor)
                # Choose the action with the highest Q-value
                action_tensor = action_values.max(1)[1].view(1, 1)
            action = action_tensor.item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = np.array(next_state, dtype=np.float32)
            episode_reward += reward

            if done:
                break

        total_rewards.append(episode_reward)
        print(f"Episode {i_episode}\tReward: {episode_reward:.2f}\tSteps: {t+1}")

    env.close()
    print(f"\nTesting finished over {num_episodes} episodes.")
    if total_rewards:
        print(f"Average Reward: {np.mean(total_rewards):.2f}")
        print(f"Min Reward: {np.min(total_rewards):.2f}")
        print(f"Max Reward: {np.max(total_rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a trained DQN agent on LunarLander-v3."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/lunar_lander_final.pth",
        help="Path to the saved model file.",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to test."
    )
    parser.add_argument("--no_render", action="store_true", help="Disable rendering.")
    args = parser.parse_args()

    test_lunar_lander(
        model_path=args.model_path,
        num_episodes=args.episodes,
        render=not args.no_render,
    )
