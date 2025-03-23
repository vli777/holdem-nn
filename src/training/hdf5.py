import os
import h5py
import logging
import numpy as np
import uuid
from typing import List, Dict, Any


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


def save_to_hdf5(file_path: str, game_sequences: List[Dict[str, Any]]):
    """
    Save game sequences to an HDF5 file.

    Args:
        file_path (str): Path to save the HDF5 file.
        game_sequences (List[Dict]): List of game sequences, each containing:
            - game_id: int
            - winner_id: int
            - final_chips: float
            - sequence: List[Dict] containing state-action records
    """
    with h5py.File(file_path, 'w') as f:
        for game in game_sequences:
            game_group = f.create_group(f"game_{game['game_id']}")
            
            # Save game metadata
            game_group.attrs['winner_id'] = game['winner_id']
            game_group.attrs['final_chips'] = game['final_chips']
            
            # Extract sequence data
            sequence = game['sequence']
            states = np.array([record['state'] for record in sequence])
            actions = np.array([record['action'] for record in sequence])
            player_ids = np.array([record['player_id'] for record in sequence])
            positions = np.array([record['position'] for record in sequence])
            recent_actions = np.array([record['recent_action'] for record in sequence])
            strategies = np.array([record['strategy'] for record in sequence])
            bluffing_probs = np.array([record['bluffing_probability'] for record in sequence])
            
            # Save sequence data
            game_group.create_dataset('states', data=states)
            game_group.create_dataset('actions', data=actions)
            game_group.create_dataset('player_ids', data=player_ids)
            game_group.create_dataset('positions', data=positions)
            game_group.create_dataset('recent_actions', data=recent_actions)
            game_group.create_dataset('strategies', data=strategies)
            game_group.create_dataset('bluffing_probs', data=bluffing_probs)
            
        logging.info(f"Saved {len(game_sequences)} complete game sequences to {file_path}")


def load_from_hdf5(file_path: str) -> List[Dict[str, Any]]:
    """
    Load game sequences from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        List[Dict]: List of game sequences with the same structure as save_to_hdf5 input.
    """
    game_sequences = []
    
    with h5py.File(file_path, 'r') as f:
        for game_key in f.keys():
            game_group = f[game_key]
            
            # Load game metadata
            game_data = {
                'game_id': int(game_key.split('_')[1]),
                'winner_id': game_group.attrs['winner_id'],
                'final_chips': game_group.attrs['final_chips'],
                'sequence': []
            }
            
            # Load sequence data
            states = game_group['states'][:]
            actions = game_group['actions'][:]
            player_ids = game_group['player_ids'][:]
            positions = game_group['positions'][:]
            recent_actions = game_group['recent_actions'][:]
            strategies = game_group['strategies'][:]
            bluffing_probs = game_group['bluffing_probs'][:]
            
            # Reconstruct sequence
            for i in range(len(states)):
                game_data['sequence'].append({
                    'state': states[i],
                    'action': actions[i],
                    'player_id': player_ids[i],
                    'position': positions[i],
                    'recent_action': recent_actions[i],
                    'strategy': strategies[i],
                    'bluffing_probability': bluffing_probs[i]
                })
            
            game_sequences.append(game_data)
    
    return game_sequences
