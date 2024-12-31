import os
from fastapi.testclient import TestClient
from src.main import app
import pytest

client = TestClient(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "poker_model_full.pth")


@pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="Model file not found; skipping predict tests.",
)
@pytest.mark.parametrize("predictor_type", ["standard", "ensemble"])
def test_predict_action(predictor_type):
    response = client.post(
        f"/predict?predictor_type={predictor_type}",
        json={"hole_cards": ["Ah", "Kd"], "community_cards": ["Qs", "Jd", "9h"]},
    )
    assert response.status_code == 200
    assert "predicted_action" in response.json()
