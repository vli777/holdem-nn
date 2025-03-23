import logging
import random
from typing import List, Dict, Any, Tuple, Set
from config import config
from training.player_state import PlayerState
from treys import Deck

from utils import (
    calculate_pot_odds,
    encode_action,
    encode_state,
    evaluate_hand,
    encode_strategy,
)


class TexasHoldemGame:
    def __init__(self, num_players: int = 6, starting_chips: int = 1000):
        self.num_players = num_players
        self.starting_chips = starting_chips

        # Create players with random or assigned strategies
        strategies = ["tight-aggressive", "loose-passive", "balanced"]
        self.players = [
            PlayerState(
                player_id=i,
                strategy=random.choice(strategies),
                starting_chips=starting_chips,
                bluffing_probability=0.05 + random.random() * 0.15,
            )
            for i in range(num_players)
        ]

        # Deck, community cards, pot, minimal raise
        self.deck = Deck()
        self.community_cards: List[int] = []
        self.current_pot = 0
        self.minimum_raise = 2

        # For storing states/actions, etc.
        self.game_data: List[Dict[str, Any]] = []  # Will hold the final state-action records
        self.side_pots: List[Tuple[float, List[PlayerState]]] = []  # List of tuples: (pot_amount, eligible_players)

    def reset_for_new_hand(self) -> None:
        """Reset or re-init fields for a new hand."""
        self.deck.shuffle()  # Just shuffle existing deck instead of creating new one
        self.community_cards.clear()  # More efficient than reassignment
        self.current_pot = 0
        self.side_pots.clear()  # More efficient than reassignment
        for p in self.players:
            p.reset_for_new_hand()

    def deal_preflop(self) -> None:
        """Deal 2 hole cards to each player."""
        for p in self.players:
            p.hole_cards = self.deck.draw(2)

    def post_blinds(self) -> None:
        """Simple approach: small blind = 1 chip, big blind = 2 chips."""
        if self.num_players < 2:
            return

        sb_player = self.players[0]  # Small blind is first player after dealer
        bb_player = self.players[1]  # Big blind is next player

        sb = min(sb_player.chips, 1)
        sb_player.chips -= sb
        sb_player.current_bet = sb

        bb = min(bb_player.chips, 2)
        bb_player.chips -= bb
        bb_player.current_bet = bb

        self.current_pot = sb + bb

        logging.info(f"Player {sb_player.player_id} posts small blind of {sb}.")
        logging.info(f"Player {bb_player.player_id} posts big blind of {bb}.")

    def rotate_positions(self) -> None:
        """Rotate player positions clockwise."""
        first_player = self.players.pop(0)
        self.players.append(first_player)
        for idx, player in enumerate(self.players):
            player.position = idx

    def single_pass_betting_round(self, round_name: str = "pre-flop") -> bool:
        """
        Conduct a single pass of betting across all players.
        Returns True if a raise occurred, False otherwise.
        """
        logging.info(f"--- {round_name.upper()} BETTING ROUND (SINGLE PASS) ---")

        # Pre-calculate current highest bet
        current_highest_bet = max(p.current_bet for p in self.players if p.in_hand)
        
        # Use list comprehension for better performance
        active_players = [p for p in self.players if p.in_hand]
        action_happened = False

        for p in active_players:
            bet_to_call = max(current_highest_bet - p.current_bet, 0)

            # Cache hand evaluation results
            normalized_strength = evaluate_hand(p.hole_cards, self.community_cards)
            pot_odds = calculate_pot_odds(self.current_pot, bet_to_call)

            action_str = p.opponent_behavior.decide_action(
                hand_strength=normalized_strength,
                pot_odds=pot_odds,
                position=p.position,
            )

            previous_action = p.last_action
            p.last_action = action_str

            if action_str == "fold":
                p.in_hand = False
                logging.info(f"Player {p.player_id} folds.")
            elif action_str == "call":
                call_amount = min(p.chips, bet_to_call)
                p.chips -= call_amount
                p.current_bet += call_amount
                self.current_pot += call_amount
                logging.info(f"Player {p.player_id} calls {call_amount}.")
                if p.chips == 0:
                    logging.info(f"Player {p.player_id} is all-in.")
            elif action_str == "raise":
                raise_amount = bet_to_call + self.minimum_raise
                raise_amount = min(p.chips, raise_amount)
                p.chips -= raise_amount
                p.current_bet += raise_amount
                self.current_pot += raise_amount
                current_highest_bet = p.current_bet
                action_happened = True
                logging.info(f"Player {p.player_id} raises to {p.current_bet}.")
                if p.chips == 0:
                    logging.info(f"Player {p.player_id} is all-in.")

            encoded_state = encode_state(
                hole_cards=p.hole_cards,
                community_cards=self.community_cards,
                normalized_strength=normalized_strength,
                pot_odds=pot_odds,
                player_id=p.player_id,
                position=p.position,
                recent_action=encode_action(previous_action) if previous_action else 0,
                strategy=p.strategy,
                bluffing_probability=p.bluffing_probability,
            )
            encoded_act = encode_action(action_str)
            self.game_data.append({
                "state": encoded_state,
                "action": encoded_act,
                "player_id": p.player_id,
                "position": p.position,
                "recent_action": encode_action(previous_action) if previous_action else 0,
                "strategy": encode_strategy(p.strategy),
                "bluffing_probability": p.bluffing_probability,
            })

            if p.chips == 0 and (action_str == "call" or action_str == "raise"):
                logging.info(f"Player {p.player_id} has gone all-in with a bet of {p.current_bet}.")

            self.handle_side_pots()

        return action_happened

    def handle_side_pots(self) -> None:
        """
        Handle side pots based on players' all-in statuses.
        This method should be called after each betting round to adjust pots accordingly.
        """
        # Find all players who are all-in
        all_in_players = [p for p in self.players if p.chips == 0 and p.in_hand]

        if not all_in_players:
            return  # No side pots needed

        # Sort all-in players by their current bet
        all_in_players_sorted = sorted(all_in_players, key=lambda p: p.current_bet)

        for p in all_in_players_sorted:
            # The amount to cover in the main pot
            amount = p.current_bet

            # Eligible players are those who have bet at least this amount
            eligible_players = [player for player in self.players if player.current_bet >= amount]

            # Create a side pot
            side_pot = (amount * len(eligible_players), eligible_players.copy())

            # Add to side pots
            self.side_pots.append(side_pot)

            # Reduce each eligible player's current bet
            for player in eligible_players:
                player.current_bet -= amount

            # Reduce the main pot
            self.current_pot -= side_pot[0]

            logging.info(
                f"Created a side pot of {side_pot[0]} with players {[p.player_id for p in eligible_players]}."
            )

    def multi_betting_round(self, round_name: str = "pre-flop") -> None:
        """
        Conduct a multiple-pass betting round for a single street
        (pre-flop, flop, turn, or river).
        We repeat single_pass_betting_round until no new raises occur
        or until only one player remains.
        """
        logging.info(f"=== STARTING MULTI-PASS {round_name.upper()} BETTING ===")

        while True:
            # If only one player remains, betting is done
            if self.count_in_hand() < 2:
                logging.info("Betting ended because only one player remains.")
                break

            # Perform a single pass
            did_raise = self.single_pass_betting_round(round_name=round_name)

            # If no one raised, betting is complete
            if not did_raise:
                logging.info("No raise in this pass. Betting round ends.")
                break

        logging.info(f"=== MULTI-PASS {round_name.upper()} BETTING COMPLETE ===")

    def deal_flop(self) -> None:
        """Deal 3 community cards."""
        if len(self.deck.cards) < 3:
            logging.warning("Not enough cards to deal the flop.")
            return
        self.community_cards.extend(self.deck.draw(3))  # More efficient than +=

    def deal_turn(self) -> None:
        """Deal 1 community card (turn)."""
        if len(self.deck.cards) < 1:
            logging.warning("Not enough cards to deal the turn.")
            return
        self.community_cards.append(self.deck.draw(1))  # More efficient than +=

    def deal_river(self) -> None:
        """Deal 1 community card (river)."""
        if len(self.deck.cards) < 1:
            logging.warning("Not enough cards to deal the river.")
            return
        self.community_cards.append(self.deck.draw(1))  # More efficient than +=

    def play_hand(self) -> None:
        self.reset_for_new_hand()
        self.deal_preflop()
        self.post_blinds()
        # Pre-flop betting with multiple passes
        self.multi_betting_round(round_name="pre-flop")
        if self.count_in_hand() < 2:
            self.showdown()
            return

        # Flop
        self.deal_flop()
        self.reset_bets_for_next_round()
        self.multi_betting_round(round_name="flop")
        if self.count_in_hand() < 2:
            self.showdown()
            return

        # Turn
        self.deal_turn()
        self.reset_bets_for_next_round()
        self.multi_betting_round(round_name="turn")
        if self.count_in_hand() < 2:
            self.showdown()
            return

        # River
        self.deal_river()
        self.reset_bets_for_next_round()
        self.multi_betting_round(round_name="river")
        if self.count_in_hand() < 2:
            self.showdown()
            return

        self.showdown()

    def showdown(self) -> None:
        # Collect players still in hand
        remaining_players = [p for p in self.players if p.in_hand]
        if not remaining_players:
            return  # Everyone folded earlier

        # Pre-calculate all hand rankings
        player_rankings = [
            (p, evaluate_hand(p.hole_cards, self.community_cards))
            for p in remaining_players
        ]
        
        # Sort once and use for all pots
        player_rankings.sort(key=lambda x: x[1])
        
        # Process pots more efficiently
        for pot_name, pot_amount in [("main pot", self.current_pot)] + [
            (f"side pot {i+1}", pot[0]) 
            for i, pot in enumerate(self.side_pots)
        ]:
            if pot_amount == 0:
                continue
                
            # Determine eligible players
            eligible_players = (
                remaining_players 
                if pot_name == "main pot" 
                else self.side_pots[int(pot_name.split()[2]) - 1][1]
            )
            
            # Find winners more efficiently
            winners = [
                p for p, rank in player_rankings 
                if p in eligible_players and rank == player_rankings[0][1]
            ]
            
            if winners:
                split_pot = pot_amount / len(winners)
                for w in winners:
                    w.chips += split_pot

        # Reset the main pot and side pots
        self.current_pot = 0
        self.side_pots.clear()

    def reset_bets_for_next_round(self) -> None:
        """Reset each player's bet to 0 after a street ends."""
        for p in self.players:
            p.current_bet = 0

    def count_in_hand(self) -> int:
        """Return how many players are still in the hand."""
        return sum(p.in_hand for p in self.players)

    def is_game_over(self) -> bool:
        """Check if the game is over (only one player has chips)."""
        return len([p for p in self.players if p.chips > 0]) <= 1

    def get_game_data(self) -> List[Dict[str, Any]]:
        """Return or transform the recorded data for outside use."""
        return self.game_data

    def play_game(self) -> PlayerState:
        """Play a complete game until there's a single winner."""
        while not self.is_game_over():
            self.play_hand()
            # Rotate positions for the next hand
            self.rotate_positions()
            # Reset game data for the next hand
            self.game_data.clear()
        
        # Find the winner
        winner = next(p for p in self.players if p.chips > 0)
        logging.info(f"Game Over! Winner is Player {winner.player_id} with {winner.chips} chips.")
        return winner


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if config.debug else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    game = TexasHoldemGame(num_players=config.num_players)
    winner = game.play_game()
    logging.info(f"Game completed with winner: Player {winner.player_id}")
