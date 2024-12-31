import torch
from models.PokerLinformerModel import PokerLinformerModel
from config import config


class MockModel(PokerLinformerModel):
    def forward(self, states, position, player_ids, recent_actions):
        batch_size = states.size(0)
        policy_logits = torch.rand(
            batch_size, config.output_dim
        )  # Random policy logits
        value = torch.rand(batch_size, 1)  # Random value
        return policy_logits, value
