import torch
import eval7
from PokerDataset import PokerDataset
from PokerModel import PokerModel

# Load model
input_dim = 106  # Update based on your dataset format
hidden_dim = 128
output_dim = 3
model = PokerModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
model.load_state_dict(torch.load("models/poker_model.pth", weights_only=True))
model.eval()

# Sample game state
sample_action = {
    "hole_cards": [eval7.Card("As"), eval7.Card("Ah")],
    "community_cards": [eval7.Card("5d"), eval7.Card("9s"), eval7.Card("8c")],
    "hand_strength": 0.8,
    "pot_odds": 0.5
}

print("Raw hand state:")
print(f"Hole Cards: {[str(card) for card in sample_action['hole_cards']]}")
print(f"Community Cards: {[str(card) for card in sample_action['community_cards']]}")
print(f"Hand Strength: {sample_action['hand_strength']}")
print(f"Pot Odds: {sample_action['pot_odds']}")

# Encode the state
encoded_state = PokerDataset.encode_state(sample_action)
input_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Predict action
output = model(input_tensor)
predicted_action = torch.argmax(output, dim=1).item()

# Map prediction to action name
action_map = {0: "fold", 1: "call", 2: "raise"}
predicted_action_name = action_map[predicted_action]

print(f"Predicted action: {predicted_action_name}")

