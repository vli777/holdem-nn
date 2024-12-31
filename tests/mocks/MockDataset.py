import torch
from torch.utils.data import Dataset
from config import config
import numpy as np


class MockDataset(Dataset):
    def __init__(self, size=100):
        """
        Initializes a mock dataset with random data mimicking PokerDataset.

        Args:
            size (int): Number of samples in the dataset.
        """
        self.size = size

        # Generate random data for each required attribute
        self.states = torch.randn(size, config.input_dim).numpy()  # State features
        self.actions = np.random.randint(
            0, config.output_dim, size
        )  # Action labels (e.g., fold, call, raise)
        self.positions = np.random.randint(
            0, 10, size
        )  # Player positions (e.g., seat positions 0-9)
        self.player_ids = np.random.randint(
            0, 5, size
        )  # Player IDs (e.g., player numbers 0-4)
        self.recent_actions = np.random.randint(
            0, 3, size
        )  # Recent actions encoded as integers

        # Combine all attributes into a structured array
        precomputed_data = []
        for i in range(size):
            precomputed_data.append(
                (
                    self.states[i],
                    self.actions[i],
                    self.positions[i],
                    self.player_ids[i],
                    self.recent_actions[i],
                )
            )
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
