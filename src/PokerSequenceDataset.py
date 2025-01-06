import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
from config import config

DATA_PATH = config.data_path


class PokerSequenceDataset(Dataset):
    def __init__(self, hdf5_path, max_seq_len=None):
        """
        Initialize the dataset by loading game sequences from an HDF5 file.

        Args:
            hdf5_path (str): Path to the HDF5 file.
            max_seq_len (int, optional): Maximum sequence length. If None, use the longest sequence.
        """
        self.hdf5_file = h5py.File(hdf5_path, "r")
        self.game_keys = list(self.hdf5_file.keys())
        self.max_seq_len = (
            max_seq_len if max_seq_len is not None else self._get_max_seq_len()
        )

    def _get_max_seq_len(self):
        """
        Determine the maximum sequence length in the dataset.

        Returns:
            int: Maximum sequence length.
        """
        max_len = 0
        for key in self.game_keys:
            seq_len = len(self.hdf5_file[key]["states"])
            if seq_len > max_len:
                max_len = seq_len
        return max_len

    def __len__(self):
        return len(self.game_keys)

    def __getitem__(self, idx):
        """
        Retrieve a single game sequence.

        Args:
            idx (int): Index of the game sequence.

        Returns:
            dict: Dictionary containing padded sequences and masks.
        """
        key = self.game_keys[idx]
        states = self.hdf5_file[key]["states"][:]
        actions = self.hdf5_file[key]["actions"][:]
        player_ids = self.hdf5_file[key]["player_ids"][:]
        positions = self.hdf5_file[key]["positions"][:]
        recent_actions = self.hdf5_file[key]["recent_actions"][:]
        strategies = self.hdf5_file[key]["strategies"][:]
        bluffing_probabilities = self.hdf5_file[key]["bluffing_probs"][:]

        seq_len = len(states)
        if seq_len > self.max_seq_len:
            # Truncate sequences longer than max_seq_len
            states = states[-self.max_seq_len :]
            actions = actions[-self.max_seq_len :]
            player_ids = player_ids[-self.max_seq_len :]
            positions = positions[-self.max_seq_len :]
            recent_actions = recent_actions[-self.max_seq_len :]
            strategies = strategies[-self.max_seq_len :]
            bluffing_probabilities = bluffing_probabilities[-self.max_seq_len :]
            mask = np.ones(self.max_seq_len, dtype=bool)
        else:
            pad_length = self.max_seq_len - seq_len
            if pad_length > 0:
                states = np.pad(
                    states, ((pad_length, 0), (0, 0)), "constant", constant_values=0
                )
                actions = np.pad(
                    actions, (pad_length, 0), "constant", constant_values=-1
                )
                player_ids = np.pad(
                    player_ids, (pad_length, 0), "constant", constant_values=-1
                )
                positions = np.pad(
                    positions, (pad_length, 0), "constant", constant_values=-1
                )
                recent_actions = np.pad(
                    recent_actions, (pad_length, 0), "constant", constant_values=-1
                )
                strategies = np.pad(
                    strategies, (pad_length, 0), "constant", constant_values=-1
                )
                bluffing_probabilities = np.pad(
                    bluffing_probabilities,
                    (pad_length, 0),
                    "constant",
                    constant_values=-1,
                )
                mask = np.concatenate(
                    (np.zeros(pad_length, dtype=bool), np.ones(seq_len, dtype=bool))
                )
            else:
                mask = np.ones(seq_len, dtype=bool)

        return {
            "states": torch.tensor(states, dtype=torch.float32),  # [seq_len, input_dim]
            "actions": torch.tensor(actions, dtype=torch.long),  # [seq_len]
            "player_ids": torch.tensor(player_ids, dtype=torch.long),  # [seq_len]
            "positions": torch.tensor(positions, dtype=torch.long),  # [seq_len]
            "recent_actions": torch.tensor(
                recent_actions, dtype=torch.long
            ),  # [seq_len]
            "strategies": torch.tensor(strategies, dtype=torch.long),  # [seq_len]
            "bluffing_probabilities": torch.tensor(
                bluffing_probabilities, dtype=torch.float32
            ),  # [seq_len]
            "mask": torch.tensor(mask, dtype=torch.bool),  # [seq_len]
        }


def poker_collate_fn(batch):
    """
    Custom collate function to batch poker game sequences.

    Args:
        batch (list): List of individual game sequence dictionaries.

    Returns:
        dict: Batched tensors.
    """
    states = torch.stack(
        [item["states"] for item in batch]
    )  # [batch_size, seq_len, input_dim]
    actions = torch.stack([item["actions"] for item in batch])  # [batch_size, seq_len]
    player_ids = torch.stack(
        [item["player_ids"] for item in batch]
    )  # [batch_size, seq_len]
    positions = torch.stack(
        [item["positions"] for item in batch]
    )  # [batch_size, seq_len]
    recent_actions = torch.stack(
        [item["recent_actions"] for item in batch]
    )  # [batch_size, seq_len]
    strategies = torch.stack(
        [item["strategies"] for item in batch]
    )  # [batch_size, seq_len]
    bluffing_probabilities = torch.stack(
        [item["bluffing_probabilities"] for item in batch]
    )  # [batch_size, seq_len]
    masks = torch.stack([item["mask"] for item in batch])  # [batch_size, seq_len]

    return {
        "states": states,  # [batch_size, seq_len, input_dim]
        "actions": actions,  # [batch_size, seq_len]
        "player_ids": player_ids,  # [batch_size, seq_len]
        "positions": positions,  # [batch_size, seq_len]
        "recent_actions": recent_actions,  # [batch_size, seq_len]
        "strategies": strategies,  # [batch_size, seq_len]
        "bluffing_probabilities": bluffing_probabilities,  # [batch_size, seq_len]
        "mask": masks,  # [batch_size, seq_len]
    }
