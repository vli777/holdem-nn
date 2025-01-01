import h5py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging


class PokerDataset(Dataset):
    def __init__(self, data_path):
        self.logger = logging.getLogger(__name__)
        try:
            if isinstance(data_path, Path):
                data_path = str(data_path)

            self.hdf5_file = h5py.File(data_path, "r")
            self.state = self.hdf5_file["state"]
            self.action = self.hdf5_file["action"]
            self.position = self.hdf5_file["position"]
            self.player_id = self.hdf5_file["player_id"]
            self.recent_action = self.hdf5_file["recent_action"]

            self.length = self.state.shape[0]

            if self.length == 0:
                raise ValueError("Loaded dataset is empty!")

        except Exception as e:
            self.logger.error(f"Failed to load data from {data_path}: {e}")
            raise

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.length}"
            )

        state = torch.tensor(self.state[idx], dtype=torch.float32)
        state = torch.nn.functional.normalize(state, p=2, dim=0)

        return (
            state,
            torch.tensor(self.action[idx], dtype=torch.long),
            torch.tensor(self.position[idx], dtype=torch.long),
            torch.tensor(self.player_id[idx], dtype=torch.long),
            torch.tensor(self.recent_action[idx], dtype=torch.long),
        )

    def close(self):
        if self.hdf5_file:
            self.hdf5_file.close()
            self.hdf5_file = None
            self.logger.info("HDF5 file closed.")

    def __del__(self):
        self.close()
