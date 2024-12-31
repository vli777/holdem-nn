import numpy as np
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from config import config


DATA_PATH = config.data_path


def inspect_data(data_path=None, detailed=False):
    """
    Inspect or validate data in a .npz file.

    Args:
        data_path (Path): Path to the .npz file. Defaults to config.data_path.
        detailed (bool): If True, performs a detailed inspection of the data structure.

    Returns:
        None
    """
    if data_path is None:
        data_path = Path(config.data_path)

    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        return

    with np.load(data_path, allow_pickle=True) as data:
        print("Keys in .npz file:", data.files)

        # Check for specific keys in the .npz file
        key = (
            "updated_data"
            if "updated_data" in data
            else "arr_0" if "arr_0" in data else None
        )

        if not key:
            print("No recognizable key found in the .npz file.")
            return

        raw_data = data[key]
        print(
            f"Key '{key}' found. Data type: {type(raw_data)}, Length: {len(raw_data)}"
        )

        if not detailed:
            return

        # Detailed inspection
        if len(raw_data) > 0:
            first_element = raw_data[0]
            print(f"First element type: {type(first_element)}")
            print(f"Contents of first element:\n{first_element}\n")

            if isinstance(first_element, (list, tuple, np.ndarray)):
                print(f"Length of first element: {len(first_element)}")
                if len(first_element) > 0:
                    print(f"Type of first sub-element: {type(first_element[0])}")
                    print(f"Contents of first sub-element:\n{first_element[0]}\n")
                    if isinstance(first_element[0], np.ndarray):
                        print(f"Shape of first sub-element: {first_element[0].shape}")
            elif isinstance(first_element, dict):
                print(
                    "First element is a dictionary. Available keys:",
                    first_element.keys(),
                )
                for key, value in first_element.items():
                    if isinstance(value, np.ndarray):
                        print(f"Key '{key}': Array shape {value.shape}")
                    else:
                        print(f"Key '{key}': Type {type(value)}")
            else:
                print("First element is neither a list, tuple, ndarray, nor dict.")


if __name__ == "__main__":
    # Call the function with desired parameters
    inspect_data(detailed=True)
