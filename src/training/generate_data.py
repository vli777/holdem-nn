import logging
import numpy as np
import os
import time
from multiprocessing import Pool
from training.texas_holdem_game import TexasHoldemGame

import logging
import numpy as np
import os
import time
from multiprocessing import Pool


def run_simulation(num_players: int, num_hands: int) -> list:
    """
    Simulate 'num_hands' hands of Texas Hold'em
    and returns the collected state-action pairs.
    """
    game_data = []

    # Create the game instance (your OOP version)
    game = TexasHoldemGame(num_players=num_players, starting_chips=1000)

    # Play multiple hands
    for _ in range(num_hands):
        game.play_hand()
        # game.game_data will have the state-action pairs for that hand
        # We'll collect them and then reset game.game_data
        game_data.extend(game.game_data)
        game.game_data = (
            []
        )  # optional: clear between hands if you want each hand's data separate

    return game_data


def simulate_games_for_worker(args):
    """
    Worker function for multiprocessing.
    Args:
        args (tuple): (num_players, games_per_worker, worker_id)
    Returns:
        list: The combined game data from all hands this worker played.
    """
    num_players, num_hands, worker_id = args
    results = []

    data = run_simulation(num_players=num_players, num_hands=num_hands)
    results.extend(data)
    return results


def simulate_texas_holdem_parallel(num_players: int = 6, num_games: int = 1000) -> list:
    """
    Parallel simulation of Texas Hold'em games using the OOP approach.
    """
    num_workers = os.cpu_count()
    base_games_per_worker = num_games // num_workers
    remainder = num_games % num_workers

    # Distribute leftover games among the first workers
    args = []
    for worker_id in range(num_workers):
        worker_games = base_games_per_worker + (1 if worker_id < remainder else 0)
        args.append((num_players, worker_games, worker_id))

    logging.info("Starting parallel game simulation (OOP)...")
    with Pool(processes=num_workers) as pool:
        results = pool.map(simulate_games_for_worker, args)

    logging.info("Game simulation completed.")

    return [gd for r in results for gd in r]


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

    game_data = simulate_texas_holdem_parallel(num_players=6, num_games=100000)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "texas_holdem_data.npz")

    append_simulation_data(DATA_PATH, game_data)

    elapsed_time = time.time() - start_time
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds")
