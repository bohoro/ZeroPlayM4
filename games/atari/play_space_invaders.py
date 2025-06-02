# play_space_invaders.py
from ale_py import ALEInterface
from typing import Dict, Tuple, Any, Optional

ale = ALEInterface()
import gymnasium as gym
from gymnasium.utils.play import play

# Define the game ID
game_id = "ALE/SpaceInvaders-v5"

# --- Key Mapping ---
# The play utility will print the default mapping when it starts.
# You can optionally provide your own mapping.
# For Space Invaders, the actions are typically:
# 0: NOOP (No Operation)
# 1: FIRE
# 2: RIGHT
# 3: LEFT
# 4: RIGHTFIRE
# 5: LEFTFIRE

# Example custom mapping (uses arrow keys and space)
# Keys are mapped to pygame key constants
# Find constants here: https://www.pygame.org/docs/ref/key.html
try:
    import pygame

    key_mapping: Optional[Dict[Tuple[Any, ...], int]] = {
        (pygame.K_LEFT,): 3,  # Left arrow -> LEFT action
        (pygame.K_RIGHT,): 2,  # Right arrow -> RIGHT action
        (pygame.K_SPACE,): 1,  # Space bar -> FIRE action
        # Example combo keys (if needed, less common for Atari):
        # (pygame.K_LEFT, pygame.K_SPACE): 5, # Left + Space -> LEFTFIRE
        # (pygame.K_RIGHT, pygame.K_SPACE): 4, # Right + Space -> RIGHTFIRE
    }
    print("Using custom key mapping.")
except ImportError:
    print("Pygame not found, using default key mapping (if available).")
    key_mapping = None  # Let play() use its default or print instructions

# --- Launch and Play ---
print(f"Launching {game_id} for keyboard play...")
print("Press ESC to exit.")

try:
    # Create the environment - render_mode='rgb_array' is required by play()
    # play() handles the actual rendering window.
    # Set frameskip=1 to ensure every frame is processed and rendered,
    # making short-lived objects like shots visible.
    env = gym.make(game_id, render_mode="rgb_array", frameskip=3)

    # Start the interactive play session
    play(
        env, keys_to_action=key_mapping, fps=18, zoom=5
    )  # zoom=5 makes the window 5x larger

    env.close()  # Close env though play() might handle it on exit

except Exception as e:
    print(f"\nAn error occurred: {e}")
