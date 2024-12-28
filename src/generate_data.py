import eval7
import random
import numpy as np
import os
import time
from multiprocessing import Pool


def evaluate_hand(hole_cards, community_cards):
    """
    Evaluate the strength and type of a poker hand.
    Args:
        hole_cards (list[eval7.Card]): Player's hole cards.
        community_cards (list[eval7.Card]): Community cards on the table.
    Returns:
        tuple: (hand_strength (int), hand_type (str))
    """
    # Combine hole cards and community cards
    full_hand = hole_cards + community_cards

    # Evaluate hand strength
    hand_strength = eval7.evaluate(full_hand)

    # Get hand type (e.g., 'Pair', 'Straight', etc.)
    hand_type = eval7.handtype(hand_strength)

    return hand_strength, hand_type


def decide_action(hand_strength, pot_odds,
                  bluffing_probability, player_type="balanced"):
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


def monte_carlo_hand_strength(hole_cards, community_cards, num_simulations=1000):
    """
    Estimate win probability using Monte Carlo simulation.
    Args:
        hole_cards (list[eval7.Card]): Player's hole cards.
        community_cards (list[eval7.Card]): Community cards on the table.
        num_simulations (int): Number of simulations to run.
    Returns:
        float: Estimated win probability (0 to 1).
    """
    def simulate_once(_):
        # Create a fresh deck for every simulation
        deck = eval7.Deck()

        # Remove known cards from the deck
        used_cards = hole_cards + community_cards
        for card in used_cards:
            deck.cards.remove(card)

        # Shuffle the deck
        deck.shuffle()

        # Deal opponent's hole cards and remaining community cards
        opponent_hole_cards = deck.deal(2)
        remaining_community_cards = community_cards + \
            deck.deal(5 - len(community_cards))

        # Evaluate hands
        player_hand_strength = eval7.evaluate(
            hole_cards + remaining_community_cards)
        opponent_hand_strength = eval7.evaluate(
            opponent_hole_cards + remaining_community_cards)

        return 1 if player_hand_strength > opponent_hand_strength else 0.5 if player_hand_strength == opponent_hand_strength else 0

    # Compute win probability
    with Pool() as pool:
        results = pool.map(simulate_once, range(num_simulations))

    return sum(results) / num_simulations


def encode_state(hole_cards, community_cards, hand_strength, pot_odds):
    """Encodes game state as a feature vector."""
    hole_cards_vector = cards_to_vector(hole_cards)
    community_cards_vector = cards_to_vector(community_cards)
    return np.concatenate(
        [hole_cards_vector, community_cards_vector, [hand_strength], [pot_odds]])


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


def encode_action(action):
    """Encodes action as a numerical label."""
    action_map = {"fold": 0, "call": 1, "raise": 2}
    return action_map[action]


def simulate_texas_holdem(num_players=6, num_games=1000, bluffing_strategy=None):
    """
    Simulate Texas Hold'em games and collect state-action pairs.
    Args:
        num_players (int): Number of players in the game.
        num_games (int): Number of games to simulate.
        bluffing_probability (float): Bluffing probability for players.
    Returns:
        list: Encoded state-action pairs for training.
    """
    game_data = []
    bluffing_probability = bluffing_strategy() if bluffing_strategy else 0.2

    for _ in range(num_games):
        # Create a fresh deck for each game
        deck = eval7.Deck()
        deck.shuffle()

        # Deal hole cards for all players
        player_hole_cards = [deck.deal(2) for _ in range(num_players)]

        # Assign player positions (e.g., 0=dealer, 1=small blind, etc.)
        positions = list(range(num_players))
        random.shuffle(positions)  # Shuffle positions for each game

        # Initialize community cards and game state
        community_cards = []
        actions = []
        player_ids = list(range(num_players))  # Unique player IDs for the game
        current_pot = 0  # Total pot size
        min_bet = 2  # Starting bet (small blind)

        # Pre-flop round
        for player in range(num_players):
            bet_to_call = min_bet  # Simplify for simulation
            current_pot += bet_to_call
            hand_strength = monte_carlo_hand_strength(
                player_hole_cards[player], community_cards
            )
            pot_odds = current_pot / (current_pot + bet_to_call) 
            action = decide_action(
                hand_strength, pot_odds, bluffing_probability)

            encoded_state = encode_state(
                player_hole_cards[player], community_cards, hand_strength, pot_odds
            )
            encoded_action = encode_action(action)

            actions.append({
                "state": encoded_state,
                "action": encoded_action,
                "position": positions[player],  # Include player position
                "player_id": player_ids[player],  # Include player ID
                "recent_action": encoded_action,  # Track recent action
                "bet_to_call": bet_to_call,
                "pot_odds": pot_odds
            })

        # Simulate flop, turn, river rounds
        # Flop (3 cards), turn (1 card), river (1 card)
        for round_cards in [3, 1, 1]:
            community_cards += deck.deal(round_cards)

            for player in range(num_players):
                if actions[player]["action"] == 0:  # Skip folded players
                    continue
                
                bet_to_call = random.randint(2, 10) # Random bet size for each round
                current_pot += bet_to_call
                hand_strength = monte_carlo_hand_strength(
                    player_hole_cards[player], community_cards
                )
                pot_odds = current_pot / (current_pot + bet_to_call)
                action = decide_action(
                    hand_strength, pot_odds, bluffing_probability)

                encoded_state = encode_state(
                    player_hole_cards[player], community_cards, hand_strength, pot_odds
                )
                encoded_action = encode_action(action)

                actions[player] = {
                    "state": encoded_state,
                    "action": encoded_action,
                    "position": positions[player],
                    "player_id": player_ids[player],
                    "recent_action": encoded_action,
                    "bet_to_call": bet_to_call,
                    "pot_odds": pot_odds
                }

        # Append game actions
        game_data.append(actions)

    return game_data


def append_simulation_data(file_path, new_data):
    """
    Append new training data to the existing dataset.
    Args:
        file_path (str): Path to the dataset file.
        new_data (list): New game data to append.
    """
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Created directory: {dir_name}")
        
    if os.path.exists(file_path):
        # Load existing data
        existing_data = np.load(file_path, allow_pickle=True).tolist()
        # Append new data
        updated_data = existing_data + new_data
    else:
        # No existing data; use new data as the dataset
        updated_data = new_data

    # Save the updated dataset
    np.savez_compressed(file_path, updated_data)
    print(f"Data saved to {file_path}. Total samples: {len(updated_data)}")


if __name__ == "__main__":
    # For timing purposes
    start_time = time.time()

    # Generate new simulation data
    game_data = simulate_texas_holdem(
        num_games=1000, bluffing_strategy=random.uniform(0.2, 1))

    # Define the file path
    file_path = "data/texas_holdem_data.npy"

    # Append new data to the file
    append_simulation_data(file_path, game_data)

    # Display time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
