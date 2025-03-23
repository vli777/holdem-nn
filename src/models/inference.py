import numpy as np
import torch
from typing import Tuple, Dict, Any


def predict_action(
    model: torch.nn.Module,
    game_history: list,
    current_round: int,
    device: torch.device,
    max_seq_len: int = 100,
    temperature: float = 0.8,
) -> Tuple[str, float]:
    """
    Predict the next action and its value based on game history.

    Args:
        model (nn.Module): Trained PokerTransformerModel.
        game_history (list): List of dictionaries containing game data up to the current round.
        current_round (int): The current round index (e.g., 0 for pre-flop).
        device (torch.device): Device to run the model on.
        max_seq_len (int): Maximum sequence length.
        temperature (float): Temperature for action sampling (higher = more random).

    Returns:
        Tuple[str, float]: Predicted action ("fold", "call", "raise") and its predicted value.
    """
    model.eval()

    # Extract features from game_history
    states = [entry["state"] for entry in game_history]
    actions = [entry["action"] for entry in game_history]
    player_ids = [entry["player_id"] for entry in game_history]
    positions = [entry["position"] for entry in game_history]
    recent_actions = [entry["recent_action"] for entry in game_history]
    strategies = [entry.get("strategy", 2) for entry in game_history]  # Default to balanced strategy
    bluffing_probs = [entry.get("bluffing_probability", 0.0) for entry in game_history]

    # Pad or truncate the sequence
    seq_len = len(states)
    if seq_len > max_seq_len:
        states = states[-max_seq_len:]
        actions = actions[-max_seq_len:]
        player_ids = player_ids[-max_seq_len:]
        positions = positions[-max_seq_len:]
        recent_actions = recent_actions[-max_seq_len:]
        strategies = strategies[-max_seq_len:]
        bluffing_probs = bluffing_probs[-max_seq_len:]
        seq_len = max_seq_len
    else:
        pad_length = max_seq_len - seq_len
        states = np.pad(states, ((pad_length, 0), (0, 0)), "constant", constant_values=0)
        actions = np.pad(actions, (pad_length, 0), "constant", constant_values=-1)
        player_ids = np.pad(player_ids, (pad_length, 0), "constant", constant_values=-1)
        positions = np.pad(positions, (pad_length, 0), "constant", constant_values=-1)
        recent_actions = np.pad(recent_actions, (pad_length, 0), "constant", constant_values=-1)
        strategies = np.pad(strategies, (pad_length, 0), "constant", constant_values=2)
        bluffing_probs = np.pad(bluffing_probs, (pad_length, 0), "constant", constant_values=0.0)
        seq_len = max_seq_len

    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(device)  # [1, max_seq_len, input_dim]
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(0).to(device)  # [1, max_seq_len]
    player_ids = torch.tensor(player_ids, dtype=torch.long).unsqueeze(0).to(device)  # [1, max_seq_len]
    positions = torch.tensor(positions, dtype=torch.long).unsqueeze(0).to(device)  # [1, max_seq_len]
    recent_actions = torch.tensor(recent_actions, dtype=torch.long).unsqueeze(0).to(device)  # [1, max_seq_len]
    strategies = torch.tensor(strategies, dtype=torch.long).unsqueeze(0).to(device)  # [1, max_seq_len]
    bluffing_probs = torch.tensor(bluffing_probs, dtype=torch.float32).unsqueeze(0).to(device)  # [1, max_seq_len]

    # Create mask
    mask = torch.ones((1, max_seq_len), dtype=torch.bool).to(device)
    if len(game_history) < max_seq_len:
        mask[:, : max_seq_len - len(game_history)] = False  # Padded steps

    with torch.no_grad():
        policy_logits, value_pred = model(
            x=states,
            player_ids=player_ids,
            positions=positions,
            recent_actions=recent_actions,
            strategies=strategies,
            bluffing_probabilities=bluffing_probs,
            mask=mask
        )  # [1, max_seq_len, output_dim], [1, 1]

    current_round = seq_len  # Assuming 0-based indexing; adjust if necessary
    if current_round >= max_seq_len:
        current_round = max_seq_len - 1  # Prevent overflow

    # Extract logits for the current round
    current_round_logits = policy_logits[0, current_round, :]  # [output_dim]

    # Apply temperature to logits
    current_round_logits = current_round_logits / temperature

    # Sample action using softmax
    action_probs = torch.softmax(current_round_logits, dim=0)
    predicted_action_idx = torch.multinomial(action_probs, 1).item()
    
    # Get predicted value
    predicted_value = value_pred[0, 0].item()

    # Map action index to action string
    action_map = {0: "fold", 1: "call", 2: "raise"}
    predicted_action = action_map.get(predicted_action_idx, "unknown")

    return predicted_action, predicted_value


def get_action_confidence(
    model: torch.nn.Module,
    game_history: list,
    current_round: int,
    device: torch.device,
    max_seq_len: int = 100,
) -> Dict[str, float]:
    """
    Get confidence scores for all possible actions.

    Args:
        model (nn.Module): Trained PokerTransformerModel.
        game_history (list): List of dictionaries containing game data.
        current_round (int): The current round index.
        device (torch.device): Device to run the model on.
        max_seq_len (int): Maximum sequence length.

    Returns:
        Dict[str, float]: Confidence scores for each action.
    """
    model.eval()

    # Extract and prepare features (same as predict_action)
    states = [entry["state"] for entry in game_history]
    actions = [entry["action"] for entry in game_history]
    player_ids = [entry["player_id"] for entry in game_history]
    positions = [entry["position"] for entry in game_history]
    recent_actions = [entry["recent_action"] for entry in game_history]
    strategies = [entry.get("strategy", 2) for entry in game_history]
    bluffing_probs = [entry.get("bluffing_probability", 0.0) for entry in game_history]

    # Pad or truncate sequence
    seq_len = len(states)
    if seq_len > max_seq_len:
        states = states[-max_seq_len:]
        actions = actions[-max_seq_len:]
        player_ids = player_ids[-max_seq_len:]
        positions = positions[-max_seq_len:]
        recent_actions = recent_actions[-max_seq_len:]
        strategies = strategies[-max_seq_len:]
        bluffing_probs = bluffing_probs[-max_seq_len:]
        seq_len = max_seq_len
    else:
        pad_length = max_seq_len - seq_len
        states = np.pad(states, ((pad_length, 0), (0, 0)), "constant", constant_values=0)
        actions = np.pad(actions, (pad_length, 0), "constant", constant_values=-1)
        player_ids = np.pad(player_ids, (pad_length, 0), "constant", constant_values=-1)
        positions = np.pad(positions, (pad_length, 0), "constant", constant_values=-1)
        recent_actions = np.pad(recent_actions, (pad_length, 0), "constant", constant_values=-1)
        strategies = np.pad(strategies, (pad_length, 0), "constant", constant_values=2)
        bluffing_probs = np.pad(bluffing_probs, (pad_length, 0), "constant", constant_values=0.0)
        seq_len = max_seq_len

    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(device)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(0).to(device)
    player_ids = torch.tensor(player_ids, dtype=torch.long).unsqueeze(0).to(device)
    positions = torch.tensor(positions, dtype=torch.long).unsqueeze(0).to(device)
    recent_actions = torch.tensor(recent_actions, dtype=torch.long).unsqueeze(0).to(device)
    strategies = torch.tensor(strategies, dtype=torch.long).unsqueeze(0).to(device)
    bluffing_probs = torch.tensor(bluffing_probs, dtype=torch.float32).unsqueeze(0).to(device)

    # Create mask
    mask = torch.ones((1, max_seq_len), dtype=torch.bool).to(device)
    if len(game_history) < max_seq_len:
        mask[:, : max_seq_len - len(game_history)] = False

    with torch.no_grad():
        policy_logits, value_pred = model(
            x=states,
            player_ids=player_ids,
            positions=positions,
            recent_actions=recent_actions,
            strategies=strategies,
            bluffing_probabilities=bluffing_probs,
            mask=mask
        )

    current_round = seq_len
    if current_round >= max_seq_len:
        current_round = max_seq_len - 1

    # Get action probabilities
    current_round_logits = policy_logits[0, current_round, :]
    action_probs = torch.softmax(current_round_logits, dim=0)

    # Convert to confidence scores
    action_map = {0: "fold", 1: "call", 2: "raise"}
    confidences = {
        action_map[i]: prob.item()
        for i, prob in enumerate(action_probs)
    }

    return confidences
