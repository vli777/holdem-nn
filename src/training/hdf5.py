import os
import h5py
import numpy as np
import logging


def initialize_hdf5(
    file_path, state_dim, initial_size=0, chunk_size=1000, compression="gzip"
):
    """
    Initialize an HDF5 file with the required datasets.

    Args:
        file_path (str): Path to the HDF5 file.
        state_dim (int): Dimensionality of the state data.
        initial_size (int, optional): Initial size of the datasets. Defaults to 0.
        chunk_size (int, optional): Chunk size for datasets. Defaults to 1000.
        compression (str, optional): Compression method. Defaults to "gzip".
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with h5py.File(file_path, "w") as hdf5_file:
        # Create datasets with appropriate shapes and chunks
        hdf5_file.create_dataset(
            "state",
            shape=(initial_size, state_dim),
            maxshape=(None, state_dim),
            dtype="float32",
            chunks=(
                (min(chunk_size, max(1, initial_size)), state_dim)
                if initial_size > 0
                else (chunk_size, state_dim)
            ),
            compression=compression,
        )

        for key in ["action", "player_id", "position", "recent_action"]:
            hdf5_file.create_dataset(
                key,
                shape=(initial_size,),
                maxshape=(None,),
                dtype="int64",
                chunks=(
                    (min(chunk_size, max(1, initial_size)),)
                    if initial_size > 0
                    else (chunk_size,)
                ),
                compression=compression,
            )

    logging.info(
        f"HDF5 file initialized at {file_path} with initial size {initial_size}."
    )


def append_to_hdf5(file_path, new_data, state_dim, chunk_size=1000, compression="gzip"):
    """
    Append new data to an existing HDF5 file or create a new one if it doesn't exist.

    Args:
        file_path (str): Path to the HDF5 file.
        new_data (list of dict): New data to append, with keys matching the datasets.
        state_dim (int): Dimensionality of the state data.
        chunk_size (int, optional): Chunk size for datasets. Defaults to 1000.
        compression (str, optional): Compression method. Defaults to "gzip".
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if not os.path.exists(file_path):
        # Initialize the HDF5 file if it doesn't exist
        initialize_hdf5(
            file_path,
            state_dim,
            initial_size=0,
            chunk_size=chunk_size,
            compression=compression,
        )

    # Append data to the existing HDF5 file
    try:
        with h5py.File(file_path, "a") as hdf5_file:
            for key in ["state", "action", "position", "player_id", "recent_action"]:
                dataset = hdf5_file[key]
                # Extract the data for the current key and convert to numpy array
                data_to_append = [entry[key] for entry in new_data]
                data_to_append = np.array(data_to_append, dtype=dataset.dtype)

                # Resize and append the new data
                dataset.resize(dataset.shape[0] + data_to_append.shape[0], axis=0)
                dataset[-data_to_append.shape[0] :] = data_to_append
        logging.info(f"Appended {len(new_data)} samples to {file_path}.")
    except OSError as e:
        raise OSError(f"Failed to open or append to HDF5 file {file_path}: {e}")
