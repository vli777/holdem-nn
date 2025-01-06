import os
import h5py
import numpy as np
import logging


def save_game_sequences_to_hdf5(hdf5_path, game_sequences):
    """
    Save game-level sequences to the HDF5 file in a structured manner.
    
    Args:
        hdf5_path (str): Path to the HDF5 file.
        game_sequences (list): List of game sequences, where each game sequence is a list of dictionaries.
    """
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, "a") as hdf5_file:
        for idx, game_sequence in enumerate(game_sequences):
            group_name = f"game_{idx}"
            if group_name in hdf5_file:
                logging.warning(f"Group {group_name} already exists. Skipping.")
                continue
            group = hdf5_file.create_group(group_name)
            
            # Extract fields
            states = [entry["state"] for entry in game_sequence]
            actions = [entry["action"] for entry in game_sequence]
            player_ids = [entry["player_id"] for entry in game_sequence]
            positions = [entry["position"] for entry in game_sequence]
            recent_actions = [entry["recent_action"] for entry in game_sequence]
            
            # Convert to numpy arrays
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.int64)
            player_ids = np.array(player_ids, dtype=np.int64)
            positions = np.array(positions, dtype=np.int64)
            recent_actions = np.array(recent_actions, dtype=np.int64)
            
            # Create datasets
            group.create_dataset("states", data=states, compression="gzip")
            group.create_dataset("actions", data=actions, compression="gzip")
            group.create_dataset("player_ids", data=player_ids, compression="gzip")
            group.create_dataset("positions", data=positions, compression="gzip")
            group.create_dataset("recent_actions", data=recent_actions, compression="gzip")
            
    logging.info(f"Saved {len(game_sequences)} game sequences to {hdf5_path}.")