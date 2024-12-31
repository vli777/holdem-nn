import logging
from tqdm import tqdm
import random
import numpy as np
import os
import time
from multiprocessing import Pool
from utils import calculate_pot_odds, decide_action, encode_state, encode_action, evaluate_hand
from treys import Deck

def simulate_texas_holdem(num_players: int = 6, num_games: int = 1000, bluffing_strategy=None) -> list:
    """
    Simulate Texas Hold'em games and collect state-action pairs.

    Args:
        num_players (int): Number of players in the game.
        num_games (int): Number of games to simulate.
        bluffing_strategy (callable): Function to determine bluffing probability.

    Returns:
        list: A list of state-action pairs for each game.
    """
    game_data = []
    bluffing_probability = bluffing_strategy() if bluffing_strategy else 0.2

    for _ in tqdm(range(num_games), desc="Simulating Games"):
        deck = Deck()

        # Initial setup
        player_hole_cards = [deck.draw(2) for _ in range(num_players)]
        community_cards = []
        actions = [{} for _ in range(num_players)]
        current_pot = 0
        min_bet = 2

        # Pre-flop betting round
        for player in range(num_players):
            bet_to_call = min_bet
            current_pot += bet_to_call

            # Evaluate hand strength
            hand_strength = evaluate_hand(player_hole_cards[player], community_cards)

            # Calculate pot odds
            pot_odds = calculate_pot_odds(current_pot, bet_to_call)

            # Decide action
            action = decide_action(hand_strength, pot_odds, bluffing_probability)

            # Encode state and action
            encoded_state = encode_state(
                player_hole_cards[player], community_cards, hand_strength, pot_odds
            )
            encoded_action = encode_action(action)

            actions[player] = {
                "state": encoded_state,
                "action": encoded_action,
                "player_id": player,
                "recent_action": encoded_action,
                "bet_to_call": bet_to_call,
                "pot_odds": pot_odds,
            }

        # Flop, Turn, River rounds
        for round_cards in [3, 1, 1]:  # Flop, Turn, River
            if len(deck.cards) < round_cards:
                logging.error(
                    f"Not enough cards to deal {round_cards}. Remaining: {len(deck.cards)}"
                )
                continue

            community_cards += deck.draw(round_cards)

            for player in range(num_players):
                if actions[player]["action"] == 0:  # Skip folded players
                    continue

                bet_to_call = random.randint(2, 10)
                current_pot += bet_to_call

                hand_strength = evaluate_hand(player_hole_cards[player], community_cards)
                pot_odds = calculate_pot_odds(current_pot, bet_to_call)
                action = decide_action(hand_strength, pot_odds, bluffing_probability)

                encoded_state = encode_state(
                    player_hole_cards[player], community_cards, hand_strength, pot_odds
                )
                encoded_action = encode_action(action)

                actions[player] = {
                    "state": encoded_state,
                    "action": encoded_action,
                    "player_id": player,
                    "recent_action": encoded_action,
                    "bet_to_call": bet_to_call,
                    "pot_odds": pot_odds,
                }

        game_data.append(actions)

    return game_data

def simulate_texas_holdem_parallel(num_players: int = 6, num_games: int = 1000) -> list:
    """
    Parallel simulation of Texas Hold'em games.

    Args:
        num_players (int): Number of players in the game.
        num_games (int): Number of games to simulate.

    Returns:
        list: Results of simulated games.
    """
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(
            lambda _: simulate_texas_holdem(num_players=num_players, num_games=1),
            range(num_games),
        )
    return [game for result in results for game in result]  # Flatten results

def append_simulation_data(file_path: str, new_data: list):
    """
    Append new training data to the existing dataset.

    Args:
        file_path (str): Path to the dataset file.
        new_data (list): New game data to append.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    updated_data = []
    if os.path.exists(file_path):
        with np.load(file_path, allow_pickle=True) as data:
            existing_data = data.get("updated_data", []).tolist()
        updated_data = existing_data + new_data
    else:
        updated_data = new_data

    np.savez_compressed(file_path, updated_data=updated_data)
    logging.info(f"Data saved to {file_path}. Total samples: {len(updated_data)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()

    game_data = simulate_texas_holdem_parallel(num_games=100)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "texas_holdem_data.npz")

    append_simulation_data(DATA_PATH, game_data)

    elapsed_time = time.time() - start_time
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds")
