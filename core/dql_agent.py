import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

"""
This file implements a Deep Q-Learning (DQN) agent.
"""

# Default Configs
DQL_CONFIG = {
    "hidden_size": 128,
    "learning_rate": 5e-4,
    "buffer_size": 50000,
    "batch_size": 128,
    "gamma": 0.99,
    "epsilon_start": 0.9,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.99,
    "target_update_freq": 100,  # Number of learning steps
    "num_episodes": 1550,
    "max_t": 1000,  # Max steps per episode
}


class QNetwork(nn.Module):
    """
    Define the Q-Network the neural network architecture used to approximate the
    Q-value function. The Q-value Q(s, a) represents the expected future reward
    for taking action 'a' in state 's'.
    """

    def __init__(self, state_size, action_size, hidden_size=64):
        """
        Initialize the Q-Network.

        Args:
            state_size: The dimensionality of the input state space.
            action_size: The number of possible actions the agent can take.
            hidden_size: The number of neurons in each hidden layer.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: The input state.

        Returns:
            Q-values for each action.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Define the Replay Buffer
"""
An instance of Transition will store an experience tuple, which consists of:
    state: The state observed by the agent.
    action: The action taken by the agent.
    next_state: The state the agent transitioned to after taking the action.
    reward: The reward received for taking the action.
    done: A boolean indicating whether the episode terminated after this transition.
"""
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayBuffer:
    """
    This class implements the experience replay mechanism, a key component of DQN.
    Storing experiences and sampling them randomly helps to break correlations
    between consecutive samples, stabilizing learning.
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Define the DQL Agent
class DQNAgent:
    """
    This is the main class that brings everything together to implement the DQN algorithm.
    """

    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config  # Hyperparameters

        # Device
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Q-Networks
        self.policy_net = QNetwork(
            state_size, action_size, config.get("hidden_size", 64)
        ).to(self.device)
        self.target_net = QNetwork(
            state_size, action_size, config.get("hidden_size", 64)
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained directly

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.get("learning_rate", 1e-3)
        )

        # Replay memory
        self.memory = ReplayBuffer(config.get("buffer_size", 10000))

        # Epsilon for exploration
        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)

        # Other hyperparameters
        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.99)  # Discount factor
        self.target_update_freq = config.get(
            "target_update_freq", 10
        )  # How often to update target network
        self.steps_done = 0

    def select_action(self, state):
        """Selects an action using an epsilon-greedy policy."""
        sample = random.random()
        self.epsilon = max(
            self.epsilon_min, self.epsilon * self.epsilon_decay
        )  # Decay epsilon
        self.steps_done += 1

        if sample > self.epsilon:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_values = self.policy_net(state_tensor)
                return action_values.max(1)[1].view(1, 1)
        else:  # random action
            return torch.tensor(
                [[random.randrange(self.action_size)]],
                device=self.device,
                dtype=torch.long,
            )

    def learn(self):
        """Update value parameters using given batch of experience tuples."""
        if len(self.memory) < self.batch_size:
            return  # Not enough samples in memory

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )

        # Filter out None next_states and convert to tensor
        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        if len(non_final_next_states_list) > 0:
            non_final_next_states = torch.FloatTensor(
                np.array(non_final_next_states_list)
            ).to(self.device)
        else:
            non_final_next_states = torch.empty(0, self.state_size, device=self.device)

        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.cat(batch.action).to(
            self.device
        )  # Ensure actions are correctly formatted
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(self.device)

        # Ensure action_batch is of shape [batch_size, 1] if it's not already
        if action_batch.ndim == 1:
            action_batch = action_batch.unsqueeze(1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # TODO - make sure I am clear on what this is doing
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_next_states.nelement() > 0:  # Check if tensor is not empty
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(
                    non_final_next_states
                ).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss (or MSE loss)
        criterion = nn.SmoothL1Loss()  # Huber loss
        # criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )  # also update target net
        self.policy_net.eval()  # Set to evaluation mode if loading for inference
        self.target_net.eval()
