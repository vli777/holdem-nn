from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from . import PokerPredictor
import eval7

app = FastAPI()
predictor = PokerPredictor(
    model_path="models/poker_model.pth",
    input_dim=106,
    hidden_dim=128,
    output_dim=3
)

class HandState(BaseModel):
    hole_cards: List[str]
    community_cards: List[str]
    hand_strength: float
    pot_odds: float

@app.post("/predict")
def predict_action(hand_state: HandState):
    # Convert input strings to eval7.Card objects
    sample_action = {
        "hole_cards": [eval7.Card(card) for card in hand_state.hole_cards],
        "community_cards": [eval7.Card(card) for card in hand_state.community_cards],
        "hand_strength": hand_state.hand_strength,
        "pot_odds": hand_state.pot_odds
    }

    # Predict the action
    predicted_action = predictor.predict_action(sample_action)
    return {"predicted_action": predicted_action}
