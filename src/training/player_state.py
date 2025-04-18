from typing import List, Optional
from training.opponent_behavior import OpponentBehavior


class PlayerState:
    __slots__ = [
        'player_id',
        'strategy',
        'chips',
        'bluffing_probability',
        'in_hand',
        'hole_cards',
        'current_bet',
        'last_action',
        'position',
        'opponent_behavior'
    ]

    def __init__(
        self,
        player_id: int,
        strategy: str = "balanced",
        starting_chips: int = 1000,
        bluffing_probability: float = 0.2,
    ):
        self.player_id = player_id
        self.strategy = strategy
        self.chips = starting_chips
        self.bluffing_probability = bluffing_probability
        self.in_hand = True
        self.hole_cards: List[int] = []
        self.current_bet = 0
        self.last_action: Optional[str] = None  # e.g. "fold", "call", "raise"
        self.position = player_id
        self.opponent_behavior = OpponentBehavior(
            strategy=strategy,
            bluffing_probability=bluffing_probability
        )

    def reset_for_new_hand(self) -> None:
        """Reset relevant fields when starting a new hand."""
        self.in_hand = True
        self.hole_cards.clear()  # More efficient than reassignment
        self.current_bet = 0
        self.last_action = None
