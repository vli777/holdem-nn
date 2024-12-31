import logging
import random
import numpy as np
import os
import time
from multiprocessing import Pool
from utils import (
    calculate_pot_odds,
    decide_action,
    encode_state,
    encode_action,
    evaluate_hand,
)
from treys import Deck


import logging
import random
import numpy as np
import os
import time
from multiprocessing import Pool
from treys import Deck
from utils import calculate_pot_odds, decide_action, encode_state, encode_action
from training.opponent_behavior import OpponentBehavior


def simulate_texas_holdem(
    num_players: int = 6, num_games: int = 1000, bluffing_strategy=None
) -> list:
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
    opponents = [
        OpponentBehavior(
            strategy=random.choice(["tight-aggressive", "loose-passive", "balanced"])
        )
        for _ in range(num_players - 1)
    ]

    def update_player_action(
        player, hole_cards, community_cards, current_pot, bet_to_call, opponent=None
    ):
        """
        Helper function to evaluate hand, decide action, and encode state-action.
        """
        normalized_hand_strength = evaluate_hand(hole_cards, community_cards)
        pot_odds = calculate_pot_odds(current_pot, bet_to_call)

        if opponent:
            action = opponent.decide_action(normalized_hand_strength, pot_odds, player)
        else:
            action = decide_action(
                normalized_hand_strength, pot_odds, bluffing_probability
            )

        return {
            "state": encode_state(
                hole_cards, community_cards, normalized_hand_strength, pot_odds
            ),
            "action": encode_action(action),
            "player_id": player,
            "recent_action": encode_action(action),
            "bet_to_call": bet_to_call,
            "pot_odds": pot_odds,
        }

    for _ in range(num_games):
        deck = Deck()
        player_hole_cards = [deck.draw(2) for _ in range(num_players)]
        community_cards = []
        actions = [{} for _ in range(num_players)]
        current_pot = 0
        min_bet = 2

        # Pre-flop betting round
        for player in range(num_players):
            bet_to_call = min_bet
            current_pot += bet_to_call
            if player == 0:
                actions[player] = update_player_action(
                    player,
                    player_hole_cards[player],
                    community_cards,
                    current_pot,
                    bet_to_call,
                )
            else:
                opponent = opponents[player - 1]
                actions[player] = update_player_action(
                    player,
                    player_hole_cards[player],
                    community_cards,
                    current_pot,
                    bet_to_call,
                    opponent=opponent,
                )

        # Flop, Turn, River rounds
        for round_cards in [3, 1, 1]:
            if len(deck.cards) < round_cards:
                logging.warning(
                    f"Not enough cards to deal {round_cards}. Skipping round."
                )
                continue
            community_cards += deck.draw(round_cards)

            for player in range(num_players):
                if actions[player].get("action") == 0:  # Skip folded players
                    continue
                bet_to_call = random.randint(2, 10)
                current_pot += bet_to_call
                if player == 0:
                    actions[player] = update_player_action(
                        player,
                        player_hole_cards[player],
                        community_cards,
                        current_pot,
                        bet_to_call,
                    )
                else:
                    opponent = opponents[player - 1]
                    actions[player] = update_player_action(
                        player,
                        player_hole_cards[player],
                        community_cards,
                        current_pot,
                        bet_to_call,
                        opponent=opponent,
                    )

        game_data.append(actions)

    return game_data


def simulate_games_for_worker(args):
    """
    Simulate Texas Hold'em games for a specific worker.
    Args:
        args (tuple): (num_players, games_per_worker)
    Returns:
        list: Simulated games.
    """
    num_players, num_games, worker_id = args
    results = []
    for game in range(num_games):
        if game % 10 == 0:  # Log every 10 games
            logging.info(f"Worker {worker_id} completed {game}/{num_games} games.")
        results.extend(simulate_texas_holdem(num_players=num_players, num_games=1))
    return results


def simulate_texas_holdem_parallel(num_players: int = 6, num_games: int = 1000) -> list:
    """
    Parallel simulation of Texas Hold'em games.

    Args:
        num_players (int): Number of players in the game.
        num_games (int): Number of games to simulate.

    Returns:
        list: Results of simulated games.
    """
    num_workers = os.cpu_count()
    base_games_per_worker = num_games // num_workers
    remainder = num_games % num_workers

    # Distribute the leftover remainder among the first workers
    args = []
    for worker_id in range(num_workers):
        # Give each worker base amount + possibly 1 extra if remainder is not zero
        worker_games = base_games_per_worker + (1 if worker_id < remainder else 0)
        args.append((num_players, worker_games, worker_id))

    logging.info("Starting parallel game simulation...")
    with Pool(processes=num_workers) as pool:
        results = pool.map(simulate_games_for_worker, args)

    logging.info("Game simulation completed.")
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
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    start_time = time.time()

    game_data = simulate_texas_holdem_parallel(num_games=10)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "texas_holdem_data.npz")

    append_simulation_data(DATA_PATH, game_data)

    elapsed_time = time.time() - start_time
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds")
