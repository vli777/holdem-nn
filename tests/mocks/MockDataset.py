import numpy as np
import torch
from torch.utils.data import Dataset
from config import config


class MockDataset(Dataset):
    def __init__(self, size=100):
        """
        Initializes a mock dataset with random data constrained by config.

        Args:
            size (int): Number of samples in the dataset.
        """
        self.size = size

        # Generate random data constrained by config values
        # State features
        self.states = np.random.rand(
            size, config.input_dim + 10 + 5 + 3
        )  # Match input_dim
        self.actions = np.random.randint(0, config.output_dim, size)  # Actions
        self.positions = np.random.randint(0, 10, size)  # Positions
        self.player_ids = np.random.randint(0, 5, size)  # Player IDs
        self.recent_actions = np.random.randint(0, 3, size)  # Recent actions

        # Combine all attributes into a structured array
        precomputed_data = [
            (
                self.states[i],
                self.actions[i],
                self.positions[i],
                self.player_ids[i],
                self.recent_actions[i],
            )
            for i in range(size)
        ]
        self.data = np.array(precomputed_data, dtype=object)

        # Define labels for Stratified K-Fold (typically the target variable)
        self.labels = self.actions.copy()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (states, actions, positions, player_ids, recent_actions)
        """
        state, action, position, player_id, recent_action = self.data[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(position, dtype=torch.long),
            torch.tensor(player_id, dtype=torch.long),
            torch.tensor(recent_action, dtype=torch.long),
        )
