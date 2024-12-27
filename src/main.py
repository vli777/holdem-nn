import eval7
import logging
from PokerLinformerModel import PokerLinformerModel
from PokerPredictor import PokerPredictor
from PokerEnsemblePredictor import PokerEnsemblePredictor
from simulate import randomize_sample_action, play_out_game

# Logging setup
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

    # Initialize the single predictor
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

    # Randomize a sample action
    sample_action = randomize_sample_action()

    # Display the initial state
    print("\n--- Single Model Prediction ---")
    predictor.display_hand(sample_action)
    print("Predicted Action:", predictor.predict_action(sample_action))

    # Initialize the ensemble predictor
    ensemble_predictor = PokerEnsemblePredictor(
        model_class=PokerLinformerModel,
        model_paths=[f"models/poker_model_fold{i}.pth" for i in range(1, 6)],
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    # Perform ensemble predictions
    print("\n--- Ensemble Model Prediction ---")
    print("Predicted Action (Ensemble):",
          ensemble_predictor.predict_action(sample_action))
    print("Predicted Action with Confidence:",
          ensemble_predictor.predict_with_confidence(sample_action, threshold=0.8))

    # Simulate and play out the game
    print("\n--- Simulated Game ---")
    play_out_game(ensemble_predictor, sample_action, num_players=6)


if __name__ == "__main__":
    main()
