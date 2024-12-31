from linformer import Linformer
import torch.nn as nn
from .PlayerDynamicsAttention import PlayerDynamicsAttention


class PokerLinformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        seq_len,
        num_heads,
        num_layers,
        num_players=6,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.player_dynamics = PlayerDynamicsAttention(hidden_dim, num_players)

        # Linformer encoder
        self.linformer = Linformer(
            dim=hidden_dim,
            seq_len=seq_len,
            depth=num_layers,
            k=32,
            heads=num_heads,
            one_kv_head=True,
            share_kv=True,
            dropout=0.1,
        )

        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, positions=None, player_ids=None, actions=None):
        device = x.device

        if positions is not None:
            positions = positions.clamp(
                0, self.player_dynamics.position_embeddings.num_embeddings - 1
            ).to(device)
        if player_ids is not None:
            player_ids = player_ids.clamp(
                0, self.player_dynamics.player_embeddings.num_embeddings - 1
            ).to(device)
        if actions is not None:
            actions = actions.clamp(
                0, self.player_dynamics.action_embeddings.num_embeddings - 1
            ).to(device)

        # Input Projection
        x = self.input_projection(x)

        # Player Dynamics Attention
        if positions is not None and player_ids is not None and actions is not None:
            x = self.player_dynamics(x, player_ids, actions, positions)

        # Linformer Encoder
        x = self.linformer(x)

        # Policy and Value Heads
        x = x.squeeze(1)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value
