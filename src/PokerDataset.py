import torch
from torch.utils.data import Dataset
import numpy as np
import logging

logging.basicConfig(level=logging.WARNING)


class PokerDataset(Dataset):
    def __init__(self, data_path):
        try:
            raw_data = np.load(data_path, allow_pickle=True)
            if raw_data.size == 0:
                raise ValueError("Loaded dataset is empty!")
        except Exception as e:
            logging.error(f"Failed to load data from {data_path}: {e}")
            raise

        self.data = []

        # Parse each game's actions into state-action pairs
        for game in raw_data:
            for action in game:
                try:
                    state = self.encode_state(action)
                    action_label = self.encode_action(action["action"])                    
                    if action_label != -1:                        
                        self.data.append((state, action_label))
                except KeyError as e:
                    logging.warning(f"Missing key {e} in action data, skipping...")
                except Exception as e:
                    logging.error(f"Unexpected error: {e}, skipping...")


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
        ranks = "23456789TJQKA"  # Rank order
        suits = ["s", "h", "d", "c"]  # Suit order mapped from integers
        vector = np.zeros(52)  # 52 cards in a deck

        for card in cards:
            # Convert rank to string if it's an integer
            rank = ranks[card.rank - 2] if isinstance(card.rank, int) else card.rank

            # Convert suit to string if it's an integer
            suit = suits[card.suit] if isinstance(card.suit, int) else card.suit

            rank_index = ranks.index(rank)
            suit_index = suits.index(suit)
            card_index = rank_index * 4 + suit_index  # Unique index for each card
            vector[card_index] = 1  # Set the corresponding position in the vector

        return vector

    @staticmethod
    def encode_action(action):
        """Encodes action as a numerical label."""
        action_map = {"fold": 0, "call": 1, "raise": 2}
        
        # If action is already an integer and valid, return it directly
        if isinstance(action, int) and action in {0, 1, 2}:
            return action

        # Otherwise, map string actions
        return action_map.get(action, -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        state, action_label = self.data[idx]
        return torch.tensor(state, dtype=torch.float32), torch.tensor(action_label, dtype=torch.long)
