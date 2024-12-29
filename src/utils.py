import numpy as np

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



