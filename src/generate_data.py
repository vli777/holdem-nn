import eval7
import random
import numpy as np


def evaluate_hand(hole_cards, community_cards):
    """Evaluate the strength of a poker hand."""
    hand = eval7.Hand(hole_cards + community_cards)
    return hand.evaluate() / eval7.Hand.MAX_RANK


def decide_action(hand_strength, pot_odds, bluffing_probability, player_type="balanced"):
    """
    Decide an action based on hand strength, pot odds, and bluffing.
    Args:
        hand_strength (float): Estimated strength of the player's hand (0-1).
        pot_odds (float): Current pot odds (0-1).
        bluffing_probability (float): Chance of bluffing.
        player_type (str): Type of player strategy (e.g., "tight-aggressive", "loose-passive", "balanced").
    Returns:
        str: Action (fold, call, raise).
    """
    if player_type == "tight-aggressive":
        if hand_strength > pot_odds:
            return "raise" if random.random() > 0.3 else "call"
        return "fold"
    elif player_type == "loose-passive":
        if random.random() < bluffing_probability:
            return "call"
        return "fold" if hand_strength < 0.2 else "call"
    else:  # Balanced player
        if random.random() < bluffing_probability:
            return "raise" if random.random() > 0.5 else "call"
        if hand_strength > pot_odds:
            return "raise" if random.random() > 0.7 else "call"
        elif hand_strength > 0.3:
            return "call"
        return "fold"


def monte_carlo_hand_strength(hole_cards, community_cards, num_simulations=500):
    """Estimate win probability via Monte Carlo simulation."""
    deck = eval7.Deck()
    used_cards = hole_cards + community_cards
    for card in used_cards:
        deck.cards.remove(card)

    wins = 0
    ties = 0

    for _ in range(num_simulations):
        deck.shuffle()
        opponent_hole_cards = deck.deal(2)
        remaining_community_cards = community_cards + deck.deal(5 - len(community_cards))

        player_hand = eval7.Hand(hole_cards + remaining_community_cards)
        opponent_hand = eval7.Hand(opponent_hole_cards + remaining_community_cards)

        player_score = player_hand.evaluate()
        opponent_score = opponent_hand.evaluate()

        if player_score > opponent_score:
            wins += 1
        elif player_score == opponent_score:
            ties += 1

    total = num_simulations
    return (wins + 0.5 * ties) / total  # Adjusted for ties


def encode_state(hole_cards, community_cards, hand_strength, pot_odds):
    """Encodes game state as a feature vector."""
    hole_cards_vector = cards_to_vector(hole_cards)
    community_cards_vector = cards_to_vector(community_cards)
    return np.concatenate([hole_cards_vector, community_cards_vector, [hand_strength], [pot_odds]])


def cards_to_vector(cards):
    """Encodes cards as a one-hot vector."""
    vector = np.zeros(52)
    for card in cards:
        vector[card.to_int()] = 1
    return vector


def encode_action(action):
    """Encodes action as a numerical label."""
    action_map = {"fold": 0, "call": 1, "raise": 2}
    return action_map[action]


def simulate_texas_holdem(num_players=6, num_games=1000, bluffing_probability=0.2):
    """Simulate Texas Hold'em with multiple players."""
    deck = eval7.Deck()
    game_data = []

    for _ in range(num_games):
        deck.shuffle()

        # Deal hole cards
        player_hole_cards = [deck.deal(2) for _ in range(num_players)]

        # Initialize state
        community_cards = []
        pot_size = 2 * num_players  # Starting pot size
        player_stacks = [100 for _ in range(num_players)]  # Equal stacks
        actions = [{} for _ in range(num_players)]  # Placeholder for each player's actions

        # Simulate betting rounds
        for round_idx, num_cards in enumerate([0, 3, 1, 1]):  # Pre-flop, flop, turn, river
            community_cards += deck.deal(num_cards)

            for player in range(num_players):
                if actions[player].get("action") == 0:  # Skip folded players
                    continue

                hand_strength = monte_carlo_hand_strength(player_hole_cards[player], community_cards)
                pot_odds = pot_size / (player_stacks[player] + pot_size)
                action = decide_action(hand_strength, pot_odds, bluffing_probability)

                # Update actions and encode state
                encoded_state = encode_state(player_hole_cards[player], community_cards, hand_strength, pot_odds)
                encoded_action = encode_action(action)

                actions[player] = {
                    "state": encoded_state,
                    "action": encoded_action
                }

        game_data.append(actions)

    return game_data


# Save the simulation data
game_data = simulate_texas_holdem(num_games=1000, bluffing_probability=0.2)
np.save("texas_holdem_data.npy", game_data)