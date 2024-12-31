import torch
from utils import encode_state


class PokerPredictor:
    def __init__(
        self, model_class, model_path, input_dim, hidden_dim, output_dim, **model_kwargs
    ):
        """
        Initialize the predictor by loading the model.
        Args:
            model_class (nn.Module): The model class (e.g., PokerModel, PokerLinformerModel).
            model_path (str): Path to the saved model weights.
            input_dim (int): Number of input features.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Number of output classes.
            **model_kwargs: Additional keyword arguments for the model (e.g., num_heads, num_layers, seq_len).
        """
        self.model = model_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **model_kwargs,
        )
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()  # Set the model to evaluation mode

        self.action_map = {0: "fold", 1: "call", 2: "raise"}

    def predict_action(self, sample_action):
        """
        Predict the action for a given poker hand state.
        Args:
            sample_action (dict): A dictionary containing the hand state:
                                - "hole_cards": List[eval7.Card]
                                - "community_cards": List[eval7.Card]
                                - "hand_strength": float
                                - "pot_odds": float
        Returns:
            str: Predicted action ("fold", "call", or "raise").
        """
        # Encode the state
        encoded_state = encode_state(**sample_action)
        input_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(0)

        # Predict
        policy_logits, _ = self.model(input_tensor)
        predicted_action = torch.argmax(policy_logits, dim=1).item()

        # Map prediction to action name
        return self.action_map[predicted_action]
