#!/bin/bash

# Script to create the initial directory structure for the MuZero project.
# Usage: ./create_muzero_project.sh [ProjectName]
# If ProjectName is not provided, defaults to 'muzero-m4-project'.

# Set the project name - use argument $1 if provided, otherwise default
PROJECT_NAME="ZeroPlayM4"

echo "Creating project structure in directory: $PROJECT_NAME"

# Create root project directory
#mkdir "$PROJECT_NAME"
#cd "$PROJECT_NAME" || exit # Exit if cd fails

# --- Create Core Structure ---
mkdir -p core
touch core/.gitkeep

# --- Create Games Structure (with examples) ---
mkdir -p games
touch games/.gitkeep

# Example Game Category 4: Atari
mkdir -p games/atari
touch games/atari/.gitkeep

# --- Create Configs Structure ---
mkdir -p configs
touch configs/.gitkeep

# --- Create Results Structure ---
# Note: Subdirs are often created dynamically by the training script
mkdir -p results
# Add .gitkeep if you want to commit the empty dir structure
touch results/.gitkeep

# --- Create Scripts Structure ---
mkdir -p scripts
touch scripts/.gitkeep

# --- Create Utils Structure ---
mkdir -p utils
touch utils/.gitkeep

# --- Create Tests Structure ---
mkdir -p tests
touch tests/.gitkeep
mkdir -p tests/core
touch tests/core/_.gitkeep
mkdir -p tests/games
touch tests/games/.gitkeep

# --- Create Root Files ---
touch requirements.txt
exit 0