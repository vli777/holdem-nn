import torch
from torch.utils.data import Dataset
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)

class PokerDataset(Dataset):
    def __init__(self, data_path):
        """Loads poker game data from .npy file."""
        try:
            raw_data = np.load(data_path, allow_pickle=True)
        except Exception as e:
            logging.error(f"Failed to load data from {data_path}: {e}")
            raise

        self.data = []

        # Parse each game's actions into state-action pairs
        for game in raw_data:
            for action in game:
                try:
                    state = self.encode_state(action)  # Encoded game state
                    action_label = self.encode_action(action["action"])  # Action as a label
                    if action_label != -1:  # Skip unknown actions
                        self.data.append((state, action_label))
                except KeyError as e:
                    logging.warning(f"Missing key {e} in action data, skipping...")
                except Exception as e:
                    logging.error(f"Unexpected error: {e}, skipping...")

    @staticmethod
    def encode_state(action):
        """Encodes game state as a feature vector."""
        hole_cards = PokerDataset.cards_to_vector(action.get("hole_cards", []))
        community_cards = PokerDataset.cards_to_vector(action.get("community_cards", []))
        # Normalize to [0, 1]
        hand_strength = [min(max(action.get("hand_strength", 0), 0), 1)]
        pot_odds = [min(max(action.get("pot_odds", 0), 0), 1)]
        return np.concatenate([hole_cards, community_cards, hand_strength, pot_odds])

    @staticmethod
    def cards_to_vector(cards):
        """Encodes cards as a one-hot vector."""
        vector = np.zeros(52)
        for card in cards:
            if hasattr(card, "to_int") and 0 <= card.to_int() < 52:
                vector[card.to_int()] = 1
        return vector

    @staticmethod
    def encode_action(action):
        """Encodes action as a numerical label."""
        action_map = {"fold": 0, "call": 1, "raise": 2}
        return action_map.get(action, -1)  # Default to -1 for unknown actions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action_label = self.data[idx]
        return torch.tensor(state, dtype=torch.float32), torch.tensor(action_label, dtype=torch.long)
