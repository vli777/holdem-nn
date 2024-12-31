import torch
from torch.utils.data import Dataset
from config import config


class MockDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = [
            (
                torch.rand(config.input_dim),  # state: Random tensor matching input_dim
                torch.randint(0, config.output_dim, (1,)).item(),  # action_label: 0, 1, ..., output_dim-1
                torch.randint(0, 6, (1,)).item(),  # position: 0-5
                torch.randint(0, 6, (1,)).item(),  # player_id: 0-5
                torch.randint(0, 3, (1,)).item(),  # recent_action: 0, 1, or 2
            )
            for _ in range(size)
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
