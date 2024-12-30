from torch.utils.data import Dataset
import torch
import numpy as np
import logging


class PokerDataset(Dataset):
    def __init__(self, data_path):
        self.logger = logging.getLogger(__name__)
        try:
            # Load data based on file extension
            if data_path.endswith(".npz"):
                with np.load(data_path, allow_pickle=True) as data:
                    raw_data = (
                        data["updated_data"]
                        if "updated_data" in data
                        else data["arr_0"]
                    )
            elif data_path.endswith(".npy"):
                raw_data = np.load(data_path, allow_pickle=True)
            else:
                raise ValueError("Unsupported file format. Use .npy or .npz.")

            if len(raw_data) == 0:
                raise ValueError("Loaded dataset is empty!")

            precomputed_data = []
            for game in raw_data:
                for action in game:
                    try:
                        # Ensure all required keys are present
                        required_keys = [
                            "state",
                            "action",
                            "position",
                            "player_id",
                            "recent_action",
                        ]
                        for key in required_keys:
                            if key not in action:
                                raise KeyError(f"Missing key: {key}")

                        # Validate action field
                        action_label = action["action"]
                        if action_label not in [0, 1, 2]:
                            raise ValueError(f"Invalid encoded action: {action_label}")

                        state = action["state"]
                        action_label = action["action"]
                        position = action["position"]
                        player_id = action["player_id"]
                        recent_action = action["recent_action"]

                        precomputed_data.append(
                            (state, action_label, position, player_id, recent_action)
                        )
                    except KeyError as e:
                        self.logger.error(f"Missing keys in action: {e}")
                        raise  # Raise KeyError immediately
                    except ValueError as e:
                        self.logger.error(f"Invalid action value: {e}")
                        raise
                    except Exception as e:
                        self.logger.error(f"Error processing action: {e}")
                        raise

            if len(precomputed_data) == 0:
                raise ValueError("No valid data points were parsed from the dataset.")

            self.data = np.array(precomputed_data, dtype=object)

        except Exception as e:
            self.logger.error(f"Failed to load data from {data_path}: {e}")
            raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.data)}"
            )
        state, action_label, position, player_id, recent_action = self.data[idx]

        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action_label, dtype=torch.long),
            torch.tensor(position, dtype=torch.long),
            torch.tensor(player_id, dtype=torch.long),
            torch.tensor(recent_action, dtype=torch.long),
        )
