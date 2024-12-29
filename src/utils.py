import numpy as np
import logging

def encode_state(hole_cards, community_cards, hand_strength, pot_odds):
    """
    Encodes the player's state as a feature vector.

    Args:
        hole_cards (list[eval7.Card]): The player's hole cards.
        community_cards (list[eval7.Card]): The shared community cards.
        hand_strength (float): The strength of the player's hand (normalized to [0, 1]).
        pot_odds (float): The current pot odds (normalized to [0, 1]).

    Returns:
        np.ndarray: A concatenated feature vector representing the player's state.
    """
    if not hole_cards:
        raise ValueError("Hole cards are empty. This is not valid.")
    
    # Handle empty community cards gracefully (pre-flop)
    community_cards = community_cards or []

    # Encode cards into one-hot vectors
    hole_cards_vector = cards_to_vector(hole_cards)
    community_cards_vector = cards_to_vector(community_cards)

    # Concatenate the final state
    state = np.concatenate([hole_cards_vector, community_cards_vector, [hand_strength], [pot_odds]])
    return state


def encode_action(action):
    """
    Encodes an action (fold, call, raise) into a numerical label.

    Args:
        action (str): The action taken by the player.

    Returns:
        int: Encoded action (0 for fold, 1 for call, 2 for raise).
    """
    action_map = {"fold": 0, "call": 1, "raise": 2}
    if action not in action_map:
        raise ValueError(f"Unexpected action: {action}")
    return action_map[action]


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
        rank = ranks[card.rank -
                     2] if isinstance(card.rank, int) else card.rank

        # Convert suit to string if it's an integer
        suit = suits[card.suit] if isinstance(card.suit, int) else card.suit

        rank_index = ranks.index(rank)
        suit_index = suits.index(suit)
        card_index = rank_index * 4 + suit_index  # Unique index for each card
        vector[card_index] = 1  # Set the corresponding position in the vector

    return vector

def validate_dataset(dataset):
    logging.info("Validating dataset encodings...")
    invalid_samples = []

    for idx, (state, action_label, position, player_id, recent_action) in enumerate(dataset):
        try:
            # Check state dimensions
            expected_state_dim = 106  # Update if your state dimension changes
            assert len(state) == expected_state_dim, f"Invalid state dimension at index {idx}: {len(state)}"

            # Check valid action range
            assert action_label in [0, 1, 2], f"Invalid action label at index {idx}: {action_label}"

            # Check valid position range
            max_positions = 10  # Update if needed
            assert 0 <= position < max_positions, f"Invalid position at index {idx}: {position}"

            # Check valid player ID range
            num_players = 6  # Update if needed
            assert 0 <= player_id < num_players, f"Invalid player ID at index {idx}: {player_id}"

            # Check valid recent action range
            assert recent_action in [0, 1, 2], f"Invalid recent action at index {idx}: {recent_action}"

        except AssertionError as e:
            logging.error(str(e))
            invalid_samples.append(idx)

    if invalid_samples:
        logging.error(f"Validation failed for {len(invalid_samples)} samples: {invalid_samples}")
        raise ValueError(f"Dataset contains invalid encodings. See logs for details.")
    else:
        logging.info("Dataset validation passed. All encodings are valid.")
