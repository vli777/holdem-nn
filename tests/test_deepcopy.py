import eval7
from utils import deepcopy_sample_action


def test_deepcopy_sample_action():
    original_sample_action = {
        "hole_cards": [eval7.Card("As"), eval7.Card("Kd")],
        "community_cards": [eval7.Card("Tc"), eval7.Card("7h")],
        "hand_strength": 0.85,
        "pot_odds": 0.5,
    }

    sample_action_for_ensemble = deepcopy_sample_action(original_sample_action)

    # Check if dictionaries are structurally the same
    assert sample_action_for_ensemble == original_sample_action

    # Check if the objects are different instances
    assert all(id(card) != id(original_card)
               for card, original_card in zip(sample_action_for_ensemble["hole_cards"], original_sample_action["hole_cards"])), "Hole cards were not deep-copied"
    assert all(id(card) != id(original_card)
               for card, original_card in zip(sample_action_for_ensemble["community_cards"], original_sample_action["community_cards"])), "Community cards were not deep-copied"

    # Debugging outputs
    print(f"Original Hole Cards IDs: {[id(card) for card in original_sample_action['hole_cards']]}")
    print(f"Copied Hole Cards IDs: {[id(card) for card in sample_action_for_ensemble['hole_cards']]}")


