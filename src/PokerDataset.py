import torch
from torch.utils.data import Dataset
import numpy as np
import logging


class PokerDataset(Dataset):
    def __init__(self, data_path):
        self.logger = logging.getLogger(__name__)

        try:
            # Load data based on file extension
            if data_path.endswith('.npz'):
                with np.load(data_path, allow_pickle=True) as data:
                    raw_data = data['updated_data'] if 'updated_data' in data else data['arr_0']
            elif data_path.endswith('.npy'):
                raw_data = np.load(data_path, allow_pickle=True)
            else:
                raise ValueError("Unsupported file format. Use .npy or .npz.")

            if len(raw_data) == 0:
                raise ValueError("Loaded dataset is empty!")

            # Parse each game's actions into state-action pairs and precompute
            # features
            precomputed_data = []
            for game in raw_data:
                for action in game:
                    try:
                        state = self.encode_state(action)
                        action_label = self.encode_action(action["action"])
                        position = action.get("position", 0)
                        player_id = action.get("player_id", 0)
                        recent_action = action.get("recent_action", 0)

                        if action_label != -1:
                            precomputed_data.append(
                                (state, action_label, position,
                                 player_id, recent_action)
                            )
                    except KeyError as e:
                        missing_keys = {k for k in ["hole_cards", "community_cards", "action", "hand_strength", "pot_odds"]
                                        if k not in action}
                        self.logger.warning(
                            f"Missing keys: {missing_keys}, skipping action.")
                    except Exception as e:
                        self.logger.error(
                            f"Unexpected error: {e}, skipping...")

            if len(precomputed_data) == 0:
                raise ValueError(
                    "No valid data points were parsed from the dataset.")

            # Convert precomputed data to a NumPy array for faster access
            self.data = np.array(precomputed_data, dtype=object)

        except Exception as e:
            self.logger.error(f"Failed to load data from {data_path}: {e}")
            raise

    @staticmethod
    def encode_state(action):
        """Encodes game state as a feature vector."""
        hole_cards = PokerDataset.cards_to_vector(action.get("hole_cards", []))
        community_cards = PokerDataset.cards_to_vector(
            action.get("community_cards", []))
        # Normalize to [0, 1]
        hand_strength = [min(max(action.get("hand_strength", 0), 0), 1)]
        pot_odds = [min(max(action.get("pot_odds", 0), 0), 1)]
        return np.concatenate(
            [hole_cards, community_cards, hand_strength, pot_odds])

    @staticmethod
    def cards_to_vector(cards):
        """
        Encodes a list of eval7.Card objects as a one-hot vector.
        Args:
            cards (list[eval7.Card]): List of eval7.Card objects.
        Returns:
            np.ndarray: A one-hot vector representing the cards.
        """
        ranks = "23456789TJQKA"
        suits = ["s", "h", "d", "c"]
        vector = np.zeros(52)  # 52 cards in a deck

        rank_map = {r: i for i, r in enumerate(ranks)}
        suit_map = {s: i for i, s in enumerate(suits)}

        for card in cards:
            rank = ranks[card.rank -
                         2] if isinstance(card.rank, int) else card.rank
            suit = suits[card.suit] if isinstance(
                card.suit, int) else card.suit

            card_index = rank_map[rank] * 4 + suit_map[suit]
            vector[card_index] = 1

        return vector

    @staticmethod
    def encode_action(action):
        """Encodes action as a numerical label."""
        action_map = {"fold": 0, "call": 1, "raise": 2}
        if isinstance(action, int):
            return action if action in action_map.values() else -1
        return action_map.get(action, -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.data)}")
        state, action_label, position, player_id, recent_action = self.data[idx]

        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action_label, dtype=torch.long),
            torch.tensor(position, dtype=torch.long),
            torch.tensor(player_id, dtype=torch.long),
            torch.tensor(recent_action, dtype=torch.long),
        )
