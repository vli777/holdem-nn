import torch
import eval7
from PokerDataset import PokerDataset
from PokerModel import PokerModel  # Ensure this is the correct model class

# Path to model weights
MODEL_PATH = "models/poker_model.pth"

# Action mapping
ACTION_MAP = {0: "fold", 1: "call", 2: "raise"}


def load_model(model_path, input_dim, hidden_dim, output_dim):
    """
    Load the trained model from the specified path.
    """
    model = PokerModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model


def predict_action(model, game_state):
    """
    Predict the action for a given game state.
    Args:
        model: Trained PokerModel.
        game_state (dict): Encoded game state.
    Returns:
        str: Predicted action ("fold", "call", or "raise").
    """
    # Encode the game state
    encoded_state = PokerDataset.encode_state(game_state)
    input_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Model prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_action = torch.argmax(output, dim=1).item()

    return ACTION_MAP[predicted_action]


def main():
    # Model parameters
    input_dim = 106  # Update based on your dataset format
    hidden_dim = 128
    output_dim = 3

    # Load the model
    print("Loading the model...")
    model = load_model(MODEL_PATH, input_dim, hidden_dim, output_dim)
    print("Model loaded successfully!")

    # Define a sample game state
    sample_action = {
        "hole_cards": [eval7.Card("As"), eval7.Card("Ah")],
        "community_cards": [eval7.Card("5d"), eval7.Card("9s"), eval7.Card("8c")],
        "hand_strength": 0.8,
        "pot_odds": 0.5
    }

    # Display raw game state
    print("Raw hand state:")
    print(f"Hole Cards: {[str(card) for card in sample_action['hole_cards']]}")
    print(f"Community Cards: {[str(card) for card in sample_action['community_cards']]}")
    print(f"Hand Strength: {sample_action['hand_strength']}")
    print(f"Pot Odds: {sample_action['pot_odds']}")

    # Predict action
    predicted_action_name = predict_action(model, sample_action)
    print(f"Predicted action: {predicted_action_name}")


if __name__ == "__main__":
    main()
