import os
import h5py
import logging
import numpy as np
import uuid


def initialize_hdf5(
    file_path, state_dim, initial_size=0, chunk_size=1000, compression="gzip"
):
    """
    Initialize an HDF5 file with a dataset prepared for incremental writes.

    Args:
        file_path (str): Path to create the HDF5 file.
        state_dim (int): The dimensionality of the state vector.
        initial_size (int): Initial size of the dataset (number of samples).
        chunk_size (int): Size of chunks for efficient storage.
        compression (str): Compression method for HDF5 (e.g., 'gzip', 'lzf').
    """
    try:
        with h5py.File(file_path, "w") as f:
            f.create_dataset(
                "states",
                shape=(initial_size, state_dim),
                maxshape=(None, state_dim),
                chunks=(chunk_size, state_dim),
                compression=compression,
            )
            f.create_dataset(
                "labels",
                shape=(initial_size,),
                maxshape=(None,),
                chunks=(chunk_size,),
                compression=compression,
            )
        logging.info(
            f"Initialized HDF5 file at {file_path} with state_dim={state_dim}."
        )
    except Exception as e:
        logging.error(f"Failed to initialize HDF5 file at {file_path}: {e}")
        raise


def save_to_hdf5(hdf5_path, game_sequences):
    """
    Save game-level sequences to the HDF5 file in a structured manner.

    Args:
        hdf5_path (str): Path to the HDF5 file.
        game_sequences (list): List of game sequences, where each game sequence is a list of dictionaries.
    """
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    with h5py.File(hdf5_path, "a") as hdf5_file:
        for idx, game_sequence in enumerate(game_sequences):
            unique_id = uuid.uuid4()
            group_name = f"game_{unique_id}_{idx}"
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
            group.create_dataset(
                "recent_actions", data=recent_actions, compression="gzip"
            )

    logging.info(f"Saved {len(game_sequences)} game sequences to {hdf5_path}.")
