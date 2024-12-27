import eval7
from PokerLinformerModel import PokerLinformerModel
from PokerPredictor import PokerPredictor
from PokerEnsemblePredictor import PokerEnsemblePredictor
import logging

logging.basicConfig(level=logging.INFO)

# Path to model weights
MODEL_PATH = "models/poker_model.pth"


def main():
    # Model parameters
    input_dim = 106  # Update based on your dataset format
    hidden_dim = 128
    output_dim = 3
    num_heads = 4
    num_layers = 2
    seq_len = 1

    predictor = PokerPredictor(
        model_class=PokerLinformerModel,
        model_path=MODEL_PATH,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    sample_action = {
        "hole_cards": [eval7.Card("As"), eval7.Card("Ah")],
        "community_cards": [eval7.Card("5d"), eval7.Card("9s"), eval7.Card("8c")],
        "hand_strength": 0.8,
        "pot_odds": 0.5
    }

    predictor.display_hand(sample_action)
    print("Predicted Action:", predictor.predict_action(sample_action))

    ensemble_predictor = PokerEnsemblePredictor(
        model_class=PokerLinformerModel,
        model_paths=[f"models/poker_model_fold{i}.pth" for i in range(1, 6)],
        input_dim=106,
        hidden_dim=128,
        output_dim=3,
        seq_len=1,
        num_heads=4,
        num_layers=2,
    )

    print("Predicted Action (Ensemble):",
          ensemble_predictor.predict_action(sample_action))
    print("Predicted Action with Confidence:",
          ensemble_predictor.predict_with_confidence(sample_action, threshold=0.8))


if __name__ == "__main__":
    main()
