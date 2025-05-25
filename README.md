# ZeroPlayM4
An AI agent learning to master compelling games through MuZero-style self-play, developed and optimized for MacBook Pro M4.

## Summary

This project aims to build and train an AI agent, based on MuZero principles, that learns to master games (like Atari) entirely through self-play. The system utilizes an open-source MuZero implementation, adapted and run on a MacBook Pro M4 (Apple Silicon), leveraging libraries like PyTorch/TensorFlow with Metal acceleration. The goal is to explore and showcase model-based reinforcement learning techniques where the AI learns the game's rules and optimal strategies on its own.

## Project Structure 

```
ZeroPlayM4/
│
├── core/                     # Core algorithm logic (game-agnostic)
│
├── games/                    # Game-specific modules
│   └── atari/                # atari 
├── configs/                  # Experiment-specific configuration files (optional)
│   ├── README.md             # Explains how to use experiment configs
│
├── results/                  # Output directory for models, logs, etc.
│   ├── atari/
│   │   ├── models/           # Saved model checkpoints
│   │   └── logs/             # TensorBoard logs, evaluation results
│   ├── ...                   # Auto-generated for each game run
│
├── scripts/                  # High-level executable scripts
│
├── utils/                    # General utility functions (not game or core specific)
│
├── tests/                    # Unit and integration tests
│   ├── core/
│   ├── games/
│   │   └── ...
│   └── ...
│
├── .gitignore                # Standard git ignore file
├── LICENSE                   # Your chosen open-source license file (e.g., MIT.txt)
├── README.md                 # Project overview, setup instructions, usage examples
└── requirements.txt          # Python dependencies (pip install -r requirements.txt)
```

## Setup

This project uses [Conda](https://docs.conda.io/en/latest/miniconda.html) to manage dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ZeroPlayM4.git # Replace with your repo URL if different
    cd ZeroPlayM4
    ```

2.  **Create and activate the Conda environment:**
    The `environment.yml` file specifies all the necessary dependencies.
    ```bash
    conda env create -f environment.yml
    conda activate zeroplaym4
    ```
    This command creates a new Conda environment named `zeroplaym4` and installs PyTorch, Gymnasium (with Atari support), Pygame, and other required packages.

3.  **You're ready to go!** You can now run the scripts within the activated `zeroplaym4` environment. For example, to see a pretrained lunar lander agent:
    ```bash
    python scripts/test_lunar_lander_agent.py --model_path results/lunar_lander_final.pth --episodes 10
    ```



## Appendix

* [Arcade Learning Environments](https://ale.farama.org/) 
* [MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://www.youtube.com/watch?v=We20YSAJZSE)
* [MuZero General](https://github.com/werner-duvaud/muzero-general)
* [Bayesian-Elo](https://www.remi-coulom.fr/Bayesian-Elo/)


