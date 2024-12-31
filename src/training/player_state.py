class PlayerState:
    def __init__(self, player_id, strategy="balanced", starting_chips=1000):
        self.player_id = player_id
        self.strategy = strategy
        self.chips = starting_chips
        self.in_hand = True
        self.hole_cards = []
        self.current_bet = 0
        self.last_action = None  # e.g. "fold", "call", "raise"

    def reset_for_new_hand(self):
        """Reset relevant fields when starting a new hand."""
        self.in_hand = True
        self.hole_cards = []
        self.current_bet = 0
        self.last_action = None
