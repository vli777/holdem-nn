import numpy as np
from pathlib import Path
from config import config

DATA_PATH = config.data_path


def inspect_data(data_path=None, detailed=False):
    """
    Inspect or validate data in a .npz file for schema and type consistency.

    Args:
        data_path (Path): Path to the .npz file. Defaults to config.data_path.
        detailed (bool): If True, performs a thorough inspection of all elements.

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

        # Identify the key to load the dataset
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

        if len(raw_data) == 0:
            print("Dataset is empty.")
            return

        # Get the schema and types from the first element
        first_element = raw_data[0]
        print(f"First element type: {type(first_element)}")

        if not isinstance(first_element, dict):
            print("Data does not contain dictionaries. Validation is not applicable.")
            return

        required_keys = list(first_element.keys())
        key_types = {key: type(first_element[key]) for key in required_keys}

        print(f"Expected keys: {required_keys}")
        print(f"Expected types: {key_types}")

        # Detailed validation of all elements
        if detailed:
            print("Validating all elements for schema and type consistency...")
            schema_consistent = True

            for idx, element in enumerate(raw_data):
                if not isinstance(element, dict):
                    print(f"Element at index {idx} is not a dictionary.")
                    schema_consistent = False
                    break

                # Check if all required keys are present
                if set(element.keys()) != set(required_keys):
                    print(f"Inconsistent keys at index {idx}: {element.keys()}")
                    schema_consistent = False
                    break

                # Check types of values
                for key in required_keys:
                    if type(element[key]) != key_types[key]:
                        print(
                            f"Inconsistent type for key '{key}' at index {idx}: {type(element[key])} (expected {key_types[key]})"
                        )
                        schema_consistent = False
                        break

                if not schema_consistent:
                    break

            if schema_consistent:
                print("All elements have consistent schema and types.")
            else:
                print("Inconsistencies found in the dataset.")

if __name__ == "__main__":
    # Call the function with detailed inspection
    inspect_data(detailed=True)

