import logging
import os
import time
from multiprocessing import Pool
from config import config
from training.hdf5 import save_to_hdf5
from training.texas_holdem_game import TexasHoldemGame


def run_simulation(num_players: int, num_hands: int) -> list:
    """
    Simulate 'num_hands' hands of Texas Hold'em
    and return game-level sequences for each hand.

    Args:
        num_players (int): Number of players in each game.
        num_hands (int): Number of hands to simulate.

    Returns:
        list: List of game sequences, where each sequence is a list of dictionaries.
    """
    game_sequences = []
    game = TexasHoldemGame(num_players=num_players, starting_chips=1000)

    for _ in range(num_hands):
        game.play_hand()
        # Append a deep copy to avoid reference issues
        game_sequences.append(list(game.game_data))
        game.game_data = []  # Reset for the next hand

    return game_sequences


def simulate_games_for_worker(args):
    """
    Worker function for multiprocessing.
    Args:
        args (tuple): (num_players, games_per_worker, worker_id)
    Returns:
        list: The combined game data from all hands this worker played.
    """
    num_players, num_hands, worker_id = args
    results = run_simulation(num_players=num_players, num_hands=num_hands)
    return results


def simulate_texas_holdem_parallel(num_players: int = 6, num_games: int = 1000) -> list:
    """
    Parallel simulation of Texas Hold'em games
    """
    num_workers = os.cpu_count()
    base_games_per_worker = num_games // num_workers
    remainder = num_games % num_workers

    # Distribute leftover games among the first workers
    args = [
        (num_players, base_games_per_worker + (1 if i < remainder else 0), i)
        for i in range(num_workers)
    ]

    logging.info("Starting parallel game simulation ...")
    with Pool(processes=num_workers) as pool:
        results = pool.map(simulate_games_for_worker, args)

    logging.info("Game simulation completed.")

    return [gd for worker_result in results for gd in worker_result]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if config.debug else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    start_time = time.time()

    NUM_PLAYERS = config.num_players
    NUM_HANDS = config.num_hands
    DATA_PATH = config.data_path

    game_sequences = simulate_texas_holdem_parallel(
        num_players=NUM_PLAYERS, num_games=NUM_HANDS
    )

    save_to_hdf5(DATA_PATH, game_sequences)

    elapsed_time = time.time() - start_time
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds")
