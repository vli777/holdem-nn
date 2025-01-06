from config import config
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import pandas as pd


DATA_PATH = config.data_path

def plot_class_distribution(actions, title="Action Distribution"):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=actions, palette='viridis')
    plt.title(title)
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.show()


def plot_correlation_heatmap(dataset, data_format='hdf5'):
    """
    Plot a correlation heatmap for numerical features.

    Args:
        dataset (PokerSequenceDataset): The dataset object.
        data_format (str): 'hdf5'

    Returns:
        None
    """
    if data_format == 'hdf5':
        data_dict = {
            "sequence_length": [],
            "action": [],
            "player_id": [],
            "position": [],
            "recent_action": []
        }
        for key in dataset.game_keys:
            grp = dataset.hdf5_file[key]
            states = grp["states"][:]
            actions = grp["actions"][:]
            player_ids = grp["player_ids"][:]
            positions = grp["positions"][:]
            recent_actions = grp["recent_actions"][:]

            seq_len = len(states)
            data_dict["sequence_length"].append(seq_len)
            data_dict["action"].extend(actions)
            data_dict["player_id"].extend(player_ids)
            data_dict["position"].extend(positions)
            data_dict["recent_action"].extend(recent_actions)
        
    else:
        print("Unsupported data format for correlation heatmap.")
        return

    df = pd.DataFrame(data_dict)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()


def plot_player_behavior(dataset, player_id, data_format='hdf5'):
    """
    Plot action distribution for a specific player.

    Args:
        dataset (PokerSequenceDataset or npz data): The dataset object.
        player_id (int): The ID of the player to analyze.
        data_format (str): 'hdf5' or 'npz'.

    Returns:
        None
    """
    actions = []
    if data_format == 'hdf5':
        for key in dataset.game_keys:
            grp = dataset.hdf5_file[key]
            p_ids = grp["player_ids"][:]
            a_actions = grp["actions"][:]
            # Select actions where player_id matches
            actions.extend(a_actions[p_ids == player_id])
    elif data_format == 'npz':
        # Implement if needed
        pass
    else:
        print("Unsupported data format for player-specific analysis.")
        return

    plt.figure(figsize=(8, 5))
    sns.countplot(x=actions, palette='viridis')
    plt.title(f"Action Distribution for Player {player_id}")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.show()


def plot_seq_len_vs_actions(sequence_lengths, actions):
    """
    Plot average actions per sequence length.

    Args:
        sequence_lengths (list): List of sequence lengths.
        actions (list): List of actions corresponding to each sequence.

    Returns:
        None
    """
    df = pd.DataFrame({
        "sequence_length": sequence_lengths,
        "action": actions
    })

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="action", y="sequence_length", data=df, palette='coolwarm')
    plt.title("Sequence Length Distribution by Action")
    plt.xlabel("Action")
    plt.ylabel("Sequence Length")
    plt.show()


def plot_time_series(sequence_lengths):
    """
    Plot sequence lengths over the dataset index to identify trends.

    Args:
        sequence_lengths (list): List of sequence lengths.

    Returns:
        None
    """
    plt.figure(figsize=(14, 7))
    sns.lineplot(x=range(len(sequence_lengths)), y=sequence_lengths, color='blue')
    plt.title("Sequence Length Over Dataset Index")
    plt.xlabel("Dataset Index")
    plt.ylabel("Sequence Length")
    plt.show()


def calculate_padding_percentage(dataset, max_seq_len):
    """
    Calculate the percentage of padding across the dataset.

    Args:
        dataset (PokerSequenceDataset): The dataset object.
        max_seq_len (int): The maximum sequence length used for padding.

    Returns:
        float: Percentage of tokens that are padding.
    """
    total_tokens = max_seq_len * len(dataset)
    padded_tokens = 0
    for key in dataset.game_keys:
        seq_len = len(dataset.hdf5_file[key]["states"][:])
        if seq_len > max_seq_len:
            # No padding if truncated
            padded = 0
        else:
            padded = max_seq_len - seq_len
        padded_tokens += padded
    padding_percentage = (padded_tokens / total_tokens) * 100
    return padding_percentage


def inspect_hdf5_data(data_path, detailed=False):
    """
    Inspect or validate data in an HDF5 file for schema and type consistency,
    and perform Exploratory Data Analysis (EDA) using Seaborn for visualizations.

    Args:
        data_path (str or Path): Path to the HDF5 file.
        detailed (bool): If True, performs a thorough inspection of all elements.

    Returns:
        None
    """
    data_path = Path(data_path)
    if not data_path.exists():
        logging.error(f"Data file not found at {data_path}")
        return

    with h5py.File(data_path, "r") as hdf5_file:
        print("Groups in HDF5 file:")
        for group in hdf5_file.keys():
            print(f"  - {group}")

        # Initialize lists to collect data for EDA
        sequence_lengths = []
        action_counts = []
        player_id_counts = []
        position_counts = []
        recent_action_counts = []

        # Iterate through each group (game)
        for group in hdf5_file.keys():
            grp = hdf5_file[group]
            states = grp["states"][:]
            actions = grp["actions"][:]
            player_ids = grp["player_ids"][:]
            positions = grp["positions"][:]
            recent_actions = grp["recent_actions"][:]

            seq_len = len(states)
            sequence_lengths.append(seq_len)
            action_counts.extend(actions)
            player_id_counts.extend(player_ids)
            position_counts.extend(positions)
            recent_action_counts.extend(recent_actions)

        # Calculate padding percentage
        max_seq_len = max(sequence_lengths)
        dataset_size = len(hdf5_file.keys())
        padding_percentage = (sum([max_seq_len - sl for sl in sequence_lengths]) / (max_seq_len * dataset_size)) * 100
        print(f"\nDataset Size: {dataset_size} games")
        print(f"Maximum Sequence Length: {max_seq_len}")
        print(f"Padding Percentage: {padding_percentage:.2f}%")

        # Plot Sequence Length Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(sequence_lengths, bins=50, kde=True, color='skyblue')
        plt.title("Distribution of Sequence Lengths")
        plt.xlabel("Sequence Length")
        plt.ylabel("Frequency")
        plt.show()

        # Plot Action Distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(x=actions, palette='viridis')
        plt.title("Distribution of Actions")
        plt.xlabel("Action")
        plt.ylabel("Count")
        plt.show()

        # Plot Player ID Distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(x=player_ids, palette='magma')
        plt.title("Distribution of Player IDs")
        plt.xlabel("Player ID")
        plt.ylabel("Count")
        plt.show()

        # Plot Position Distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(x=positions, palette='coolwarm')
        plt.title("Distribution of Positions")
        plt.xlabel("Position")
        plt.ylabel("Count")
        plt.show()

        # Plot Recent Action Distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(x=recent_actions, palette='plasma')
        plt.title("Distribution of Recent Actions")
        plt.xlabel("Recent Action")
        plt.ylabel("Count")
        plt.show()

        if detailed:
            print("\nDetailed Validation:")
            # Schema and Type Consistency Checks
            required_datasets = {"states", "actions", "player_ids", "positions", "recent_actions"}
            for group in hdf5_file.keys():
                grp = hdf5_file[group]
                grp_keys = set(grp.keys())
                if not required_datasets.issubset(grp_keys):
                    logging.warning(f"Group '{group}' is missing some required datasets.")
                else:
                    # Additional type checks can be implemented here
                    pass

            print("Detailed inspection completed.")

    print("Data inspection completed.")


if __name__ == "__main__":
    # Call the function with detailed inspection
    # inspect_data(detailed=True)
    inspect_hdf5_data(data_path=config.data_path, detailed=True)
