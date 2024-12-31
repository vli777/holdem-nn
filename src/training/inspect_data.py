import numpy as np
from pathlib import Path


def inspect_data_structure(data_path):
    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        return

    with np.load(data_path, allow_pickle=True) as data:
        print("Keys in .npz file:", data.files)

        if "updated_data" in data:
            raw_data = data["updated_data"]
        elif "arr_0" in data:
            raw_data = data["arr_0"]
        else:
            print("No recognizable key found in the .npz file.")
            return

        print(f"\nType of 'raw_data': {type(raw_data)}")
        print(f"Length of 'raw_data': {len(raw_data)}\n")

        if len(raw_data) == 0:
            print("Dataset is empty.")
            return

        # Inspect the first element in raw_data
        first_element = raw_data[0]
        print(f"Type of first element in 'raw_data': {type(first_element)}")
        print(f"Contents of the first element:\n{first_element}\n")

        # If first_element is iterable, inspect its elements
        if isinstance(first_element, (list, tuple, np.ndarray)):
            print(f"Length of first_element: {len(first_element)}")
            print(f"Type of first_action: {type(first_element[0])}")
            print(f"Contents of first_action:\n{first_element[0]}\n")
        elif isinstance(first_element, dict):
            print("first_element is a dictionary. Available keys:")
            print(first_element.keys())
        else:
            print("first_element is neither a list, tuple, ndarray, nor dict.")


if __name__ == "__main__":
    # Adjust the path based on your script's location
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[
        2
    ]  # Assuming inspect_data_structure.py is in src/training/
    data_path = project_root / "data" / "texas_holdem_data.npz"
    inspect_data_structure(data_path)
