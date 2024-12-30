import logging
from tqdm import tqdm
import eval7
import random
import numpy as np
import os
import time
from multiprocessing import Pool
from utils import encode_state, encode_action


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


def simulate_once(args):
    """
    Simulates one hand of Texas Hold'em between a player and an opponent.
    
    Args:
        args (tuple): A tuple containing:
            - hole_cards (list[str]): The player's hole cards (e.g., ['As', 'Kd']).
            - community_cards (list[str]): Known community cards (e.g., ['2h', '3c']).
    
    Returns:
        float: 1 if player wins, 0.5 if tie, 0 if opponent wins.
    """
    hole_cards, community_cards = args

    # Deserialize the cards
    hole_cards = [eval7.Card(c) for c in hole_cards]
    community_cards = [eval7.Card(c) for c in community_cards]

    # Deck setup
    deck = eval7.Deck()
    for card in hole_cards + community_cards:
        deck.cards.remove(card)

    # Check if deck has enough cards to continue
    if len(deck.cards) < 7 - len(community_cards):  # 2 for opponent + remaining community
        raise ValueError("Not enough cards left in the deck to complete the hand.")

    deck.shuffle()

    # Deal opponent hole cards and remaining community cards
    opponent_hole_cards = deck.deal(2)
    remaining_community_cards = community_cards + deck.deal(5 - len(community_cards))

    # Evaluate hand strengths
    player_strength = eval7.evaluate(hole_cards + remaining_community_cards)
    opponent_strength = eval7.evaluate(opponent_hole_cards + remaining_community_cards)

    # Determine outcome
    if player_strength > opponent_strength:
        return 1  # Player wins
    elif player_strength == opponent_strength:
        return 0.5  # Tie
    else:
        return 0  # Opponent wins


def monte_carlo_hand_strength(
        hole_cards, community_cards, num_simulations=1000, pool=None):
    """Estimate win probability using Monte Carlo simulation."""
    # Serialize cards
    hole_cards_serialized = [str(card) for card in hole_cards]
    community_cards_serialized = [str(card) for card in community_cards]

    args = [(hole_cards_serialized, community_cards_serialized)] * \
        num_simulations

    if pool is None:
        with Pool() as pool:
            results = pool.map(simulate_once, args)
    else:
        results = pool.map(simulate_once, args)

    return sum(results) / num_simulations


def simulate_texas_holdem(num_players=6, num_games=1000,
                          bluffing_strategy=None):
    """Simulate Texas Hold'em games and collect state-action pairs."""
    game_data = []
    bluffing_probability = bluffing_strategy() if bluffing_strategy else 0.2

    for _ in tqdm(range(num_games), desc="Simulating Games"):
        deck = eval7.Deck()
        deck.shuffle()
        player_hole_cards = [deck.deal(2) for _ in range(num_players)]
        positions = list(range(num_players))
        random.shuffle(positions)
        community_cards = []
        actions = []
        player_ids = list(range(num_players))
        current_pot = 0
        min_bet = 2

        for player in range(num_players):
            bet_to_call = min_bet
            current_pot += bet_to_call
            
            hand_strength = monte_carlo_hand_strength(
                player_hole_cards[player], community_cards)
            if hand_strength == 0.0:
                raise ValueError("Monte Carlo simulation returned zero hand strength.")

            pot_odds = current_pot / (current_pot + bet_to_call)
            action = decide_action(
                hand_strength, pot_odds, bluffing_probability)

            encoded_state = encode_state(
                player_hole_cards[player],
                community_cards,
                hand_strength,
                pot_odds)
            encoded_action = encode_action(action)
            actions.append({"state": encoded_state,
                            "action": encoded_action,
                            "position": positions[player],
                            "player_id": player_ids[player],
                            "recent_action": encoded_action,
                            "bet_to_call": bet_to_call,
                            "pot_odds": pot_odds})

        for round_cards in [3, 1, 1]:  # Flop, Turn, River
            if len(deck.cards) < round_cards:
                # print(f"Not enough cards in deck for {round_cards} community cards.")
                break

            dealt_cards = deck.deal(round_cards)
            if not dealt_cards or len(dealt_cards) != round_cards:
                raise ValueError(f"Failed to deal {round_cards} cards from the deck.")

            community_cards += dealt_cards
            # print(f"Dealt {round_cards} cards: {dealt_cards}")
            # print(f"Updated community cards: {community_cards}")

            for player in range(num_players):
                # Skip folded players
                if actions[player]["action"] == 0:
                    # print(f"Player {player} has folded. Skipping...")
                    continue
                
                # Simulate a random bet
                bet_to_call = random.randint(2, 10)
                current_pot += bet_to_call

                # Calculate hand strength
                hand_strength = monte_carlo_hand_strength(player_hole_cards[player], community_cards)
                if hand_strength == 0.0:
                    raise ValueError(f"Hand strength is zero for player {player}. Community Cards: {community_cards}")
                
                # Calculate pot odds
                pot_odds = current_pot / (current_pot + bet_to_call)

                # Decide action
                action = decide_action(hand_strength, pot_odds, bluffing_probability)
                # print(f"Player {player} - Hand Strength: {hand_strength}, Pot Odds: {pot_odds}, Action: {action}")
                
                # Encode state and update actions
                encoded_state = encode_state(
                    player_hole_cards[player],
                    community_cards,
                    hand_strength,
                    pot_odds
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

                # Debugging output for actions
                # print(f"Updated action for Player {player}: {actions[player]}")

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

    updated_data = []
    if os.path.exists(file_path):
        # Load existing data from .npz
        with np.load(file_path, allow_pickle=True) as data:
            logging.info(f"Available keys: {list(data.keys())}")
            # Dynamically fetch the primary dataset key
            primary_key = list(data.keys())[0]  # Assume first key is primary
            existing_data = data[primary_key].tolist()
        updated_data = existing_data + new_data
    else:
        # No existing data; use new data as the dataset
        updated_data = new_data

    # Save the updated dataset
    np.savez_compressed(file_path, updated_data=updated_data)
    print(f"Data saved to {file_path}. Total samples: {len(updated_data)}")


if __name__ == "__main__":
    start_time = time.time()
    with Pool() as pool:
        game_data = simulate_texas_holdem(
            num_games=100,
            bluffing_strategy=lambda: random.uniform(
                0.2,
                1))

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(BASE_DIR, "data", "texas_holdem_data.npz")
    append_simulation_data(DATA_PATH, game_data)
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")
