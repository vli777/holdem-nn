import logging
import os
import time
from multiprocessing import Pool
from config import config
from training.hdf5 import save_to_hdf5
from training.texas_holdem_game import TexasHoldemGame


def run_simulation(num_players: int, num_games: int) -> list:
    """
    Simulate 'num_games' complete Texas Hold'em games
    and return game-level sequences for each game.

    Args:
        num_players (int): Number of players in each game.
        num_games (int): Number of complete games to simulate.

    Returns:
        list: List of game sequences, where each sequence is a list of dictionaries.
    """
    game_sequences = []
    
    for game_idx in range(num_games):
        game = TexasHoldemGame(num_players=num_players, starting_chips=1000)
        winner = game.play_game()
        # Append the complete game sequence
        game_sequences.append({
            'game_id': game_idx,
            'winner_id': winner.player_id,
            'final_chips': winner.chips,
            'sequence': list(game.game_data)
        })
        logging.info(f"Completed game {game_idx + 1}/{num_games}")

    return game_sequences


def simulate_games_for_worker(args):
    """
    Worker function for multiprocessing.
    Args:
        args (tuple): (num_players, games_per_worker, worker_id)
    Returns:
        list: The combined game data from all games this worker played.
    """
    num_players, num_games, worker_id = args
    results = run_simulation(num_players=num_players, num_games=num_games)
    return results


def simulate_texas_holdem_parallel(num_players: int = 6, num_games: int = 1000) -> list:
    """
    Parallel simulation of complete Texas Hold'em games
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
    NUM_GAMES = config.num_hands  # We'll keep using num_hands in config but it now represents complete games
    DATA_PATH = config.data_path

    game_sequences = simulate_texas_holdem_parallel(
        num_players=NUM_PLAYERS, num_games=NUM_GAMES
    )

    save_to_hdf5(DATA_PATH, game_sequences)

    elapsed_time = time.time() - start_time
    logging.info(f"Total execution time: {elapsed_time:.2f} seconds")
