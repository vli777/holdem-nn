import numpy as np
import torch


def predict_action(model, game_history, current_round, device, max_seq_len=100):
    """
    Predict the next action based on game history.

    Args:
        model (nn.Module): Trained PokerTransformerModel.
        game_history (list): List of dictionaries containing game data up to the current round.
        current_round (int): The current round index (e.g., 0 for pre-flop).
        device (torch.device): Device to run the model on.
        max_seq_len (int): Maximum sequence length.

    Returns:
        str: Predicted action ("fold", "call", "raise").
    """
    model.eval()

    # Extract features from game_history
    states = [entry["state"] for entry in game_history]
    actions = [entry["action"] for entry in game_history]
    player_ids = [entry["player_id"] for entry in game_history]
    positions = [entry["position"] for entry in game_history]
    recent_actions = [entry["recent_action"] for entry in game_history]

    # Pad or truncate the sequence
    seq_len = len(states)
    if seq_len > max_seq_len:
        states = states[-max_seq_len:]
        actions = actions[-max_seq_len:]
        player_ids = player_ids[-max_seq_len:]
        positions = positions[-max_seq_len:]
        recent_actions = recent_actions[-max_seq_len:]
        seq_len = max_seq_len
    else:
        pad_length = max_seq_len - seq_len
        states = np.pad(
            states, ((pad_length, 0), (0, 0)), "constant", constant_values=0
        )
        actions = np.pad(actions, (pad_length, 0), "constant", constant_values=-1)
        player_ids = np.pad(player_ids, (pad_length, 0), "constant", constant_values=-1)
        positions = np.pad(positions, (pad_length, 0), "constant", constant_values=-1)
        recent_actions = np.pad(
            recent_actions, (pad_length, 0), "constant", constant_values=-1
        )
        seq_len = max_seq_len

    # Convert to tensors
    states = (
        torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(device)
    )  # [1, max_seq_len, input_dim]
    actions = (
        torch.tensor(actions, dtype=torch.long).unsqueeze(0).to(device)
    )  # [1, max_seq_len]
    player_ids = (
        torch.tensor(player_ids, dtype=torch.long).unsqueeze(0).to(device)
    )  # [1, max_seq_len]
    positions = (
        torch.tensor(positions, dtype=torch.long).unsqueeze(0).to(device)
    )  # [1, max_seq_len]
    recent_actions = (
        torch.tensor(recent_actions, dtype=torch.long).unsqueeze(0).to(device)
    )  # [1, max_seq_len]

    # Create mask
    mask = torch.ones((1, max_seq_len), dtype=torch.bool).to(device)
    if len(game_history) < max_seq_len:
        mask[:, : max_seq_len - len(game_history)] = False  # Padded steps

    with torch.no_grad():
        policy_logits = model(states, mask=mask)  # [1, max_seq_len, output_dim]

    current_round = seq_len  # Assuming 0-based indexing; adjust if necessary
    if current_round >= max_seq_len:
        current_round = max_seq_len - 1  # Prevent overflow

    # Extract logits for the current round
    current_round_logits = policy_logits[0, current_round, :]  # [output_dim]

    # Predict action
    predicted_action_idx = torch.argmax(current_round_logits).item()
    action_map = {0: "fold", 1: "call", 2: "raise"}
    predicted_action = action_map.get(predicted_action_idx, "unknown")

    return predicted_action
