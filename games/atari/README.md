# Atari Game Environments

This directory contains the implementation specific to Atari games.

## Environment Wrapper (`environment.py`)

The `AtariEnv` class wraps the standard Gymnasium Atari environments (e.g., `ALE/Pong-v5`) to make them compatible with the core MuZero algorithm's expected `Game` interface. This involves handling observations, actions, rewards, and potentially game-specific logic like frame stacking or preprocessing.

## Configuration (`config.py`)
Contains mappings or settings specific to Atari games used during experiments.