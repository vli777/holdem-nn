import numpy as np
import logging
import random
from treys import Evaluator, Card, Deck

evaluator = Evaluator()


def evaluate_hand(hole_cards, community_cards):
    """
    Evaluate the strength and type of a poker hand.

    Args:
        hole_cards (list[int]): Player's hole cards as Treys card integers.
        community_cards (list[int]): Community cards on the table as Treys card integers.

    Returns:
        tuple: (hand_strength (int), hand_type (str))
    """
    evaluator = Evaluator()
    hand_strength = evaluator.evaluate(community_cards, hole_cards)
    normalized_strength = 1 - (hand_strength / 7462.0)  # Normalize strength (0-1)
    return normalized_strength


def decide_action(
    hand_strength, pot_odds, bluffing_probability, player_type="balanced"
):
    """
    Decide an action based on hand strength, pot odds, and bluffing.

    Args:
        hand_strength (int): Treys hand strength (1-7462).
        pot_odds (float): Current pot odds (0-1).
        bluffing_probability (float): Chance of bluffing.
        player_type (str): Type of player strategy (e.g., "tight-aggressive", "loose-passive", "balanced").

    Returns:
        str: Action ("fold", "call", "raise").
    """
    # Normalize hand strength (lower is better in Treys)
    normalized_strength = 1 - (hand_strength / 7462.0)

    if player_type == "tight-aggressive":
        if normalized_strength > pot_odds:
            return "raise" if random.random() > 0.3 else "call"
        return "fold"
    elif player_type == "loose-passive":
        if random.random() < bluffing_probability:
            return "call"
        return "fold" if normalized_strength < 0.2 else "call"
    else:  # Balanced player
        if random.random() < bluffing_probability:
            return "raise" if random.random() > 0.5 else "call"
        if normalized_strength > pot_odds:
            return "raise" if random.random() > 0.7 else "call"
        elif normalized_strength > 0.3:
            return "call"
        return "fold"


def encode_state(hole_cards, community_cards, hand_strength, pot_odds):
    """
    Encodes the player's state as a feature vector, optimized for opponent modeling.

    Args:
        hole_cards (list[int]): The player's hole cards (Treys integer format).
        community_cards (list[int]): The shared community cards (Treys integer format).
        hand_strength (int): Treys hand strength (1-7462).
        pot_odds (float): The current pot odds (normalized to [0, 1]).

    Returns:
        np.ndarray: A feature vector representing the player's state.
    """
    # Normalize hand strength (1 is best, 0 is worst)
    normalized_strength = 1 - (hand_strength / 7462.0)

    # Combine cards and numerical features
    cards_vector = np.array(hole_cards + community_cards, dtype=np.float32)
    state = np.concatenate([cards_vector, [normalized_strength, pot_odds]])

    return state


def encode_action(action):
    """
    Encodes an action ("fold", "call", "raise") into a numerical label.

    Args:
        action (str): The action taken by the player.

    Returns:
        int: Encoded action (0 for "fold", 1 for "call", 2 for "raise").
    """
    action_map = {"fold": 0, "call": 1, "raise": 2}
    if action not in action_map:
        raise ValueError(f"Unexpected action: {action}")
    return action_map[action]


def calculate_pot_odds(current_pot, bet_amount):
    """
    Calculate pot odds given the current pot size and the bet amount.
    """
    return (
        bet_amount / (current_pot + bet_amount) if current_pot + bet_amount > 0 else 0
    )


def calculate_cards_needed(community_cards, include_opponents, remaining_deck_size):
    """
    Calculate the number of cards needed based on community cards and opponent modeling.

    Args:
        community_cards (list[int]): Current community cards (Treys integer format).
        include_opponents (bool): Whether to include opponent modeling.
        remaining_deck_size (int): Number of cards remaining in the deck.

    Returns:
        int: Number of cards needed.

    Raises:
        ValueError: If the number of cards needed exceeds the remaining deck size.
    """
    cards_needed = (
        7 - len(community_cards) if include_opponents else 5 - len(community_cards)
    )
    logging.debug(
        f"Cards needed: {cards_needed}, Community cards: {Card.print_pretty_cards(community_cards)}"
    )

    if remaining_deck_size < cards_needed:
        logging.error(
            f"Cannot proceed. Needed: {cards_needed}, Available: {remaining_deck_size}"
        )
        raise ValueError(
            f"Insufficient cards in deck. Needed: {cards_needed}, Available: {remaining_deck_size}"
        )

    return cards_needed


def filter_remaining_deck(deck, excluded_cards):
    """
    Filter the remaining deck after excluding specified cards.

    Args:
        deck (Deck): The original deck (Treys Deck object).
        excluded_cards (list[int]): The list of cards to exclude (Treys integer format).

    Returns:
        list[int]: The remaining deck after exclusion.
    """
    # Ensure all excluded cards are integers
    excluded_cards_set = set(excluded_cards)
    logging.debug(
        f"Excluded Cards: {Card.print_pretty_cards(list(excluded_cards_set))}"
    )

    # Remove excluded cards from the deck
    remaining_deck = [card for card in deck.cards if card not in excluded_cards_set]

    # Validation: Check for any missing cards
    missing_cards = [card for card in excluded_cards_set if card not in deck.cards]
    if missing_cards:
        logging.warning(
            f"Some excluded cards were not found in the deck: {Card.print_pretty_cards(missing_cards)}"
        )

    logging.debug(f"Deck After Filtering: {Card.print_pretty_cards(remaining_deck)}")
    return remaining_deck


def randomize_sample_action():
    deck = Deck()
    deck.shuffle()

    hole_cards = [deck.draw(1), deck.draw(1)]
    community_cards = [deck.draw(1) for _ in range(3)]  # Flop example

    hand_strength = evaluate_hand(hole_cards, community_cards)
    pot_odds = random.uniform(0.1, 0.9)

    return {
        "hole_cards": hole_cards,
        "community_cards": community_cards,
        "hand_strength": hand_strength,
        "pot_odds": pot_odds,
    }
