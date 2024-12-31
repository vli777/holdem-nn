import numpy as np
from pathlib import Path


def validate_data(data_path):
    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        return

    with np.load(data_path, allow_pickle=True) as data:
        print("Keys in .npz file:", data.files)
        if "updated_data" in data:
            sample = data["updated_data"][0]
            print("Sample data entry:", sample)
            print("Keys in sample:", sample.keys())
        else:
            print("Key 'updated_data' not found in the .npz file.")


if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[1]  # Adjust based on script location
    data_path = project_root / "data" / "texas_holdem_data.npz"
    validate_data(data_path)
