from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_predict_action():
    response = client.post(
        "/predict?predictor_type=standard",
        json={"hole_cards": ["Ah", "Kd"], "community_cards": ["Qs", "Jd", "9h"]},
    )
    assert response.status_code == 200
    assert "predicted_action" in response.json()

    response = client.post(
        "/predict?predictor_type=ensemble",
        json={"hole_cards": ["Ah", "Kd"], "community_cards": ["Qs", "Jd", "9h"]},
    )
    assert response.status_code == 200
    assert "predicted_action" in response.json()
