import torch.nn as nn


class PlayerDynamicsAttention(nn.Module):
    def __init__(self, hidden_dim, num_players, num_actions=3, max_positions=10):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.player_embeddings = nn.Embedding(num_players, hidden_dim)
        self.action_embeddings = nn.Embedding(num_actions, hidden_dim)
        self.position_embeddings = nn.Embedding(max_positions, hidden_dim)

        self.dropout = nn.Dropout(0.1)
        self.projection = nn.Linear(hidden_dim, hidden_dim)  # Non-linear projection
        self.layer_norm = nn.LayerNorm(hidden_dim)  # Normalize output

        nn.init.xavier_uniform_(self.player_embeddings.weight)
        nn.init.xavier_uniform_(self.action_embeddings.weight)
        nn.init.xavier_uniform_(self.position_embeddings.weight)

        assert num_players > 0, "Number of players must be positive."
        assert num_actions > 0, "Number of actions must be positive."
        assert max_positions > 0, "Maximum positions must be positive."

    def forward(self, x, player_ids, actions, positions):
        device = x.device

        player_ids = player_ids.to(device)
        actions = actions.to(device)
        positions = positions.to(device)

        # Validate embedding indices
        assert player_ids.max().item() < self.player_embeddings.num_embeddings, "Out-of-range player_ids"
        assert actions.max().item() < self.action_embeddings.num_embeddings, "Out-of-range actions"
        assert positions.max().item() < self.position_embeddings.num_embeddings, "Out-of-range positions"

        # Embedding lookups
        player_embed = self.player_embeddings(player_ids).unsqueeze(1)
        action_embed = self.action_embeddings(actions).unsqueeze(1)
        position_embed = self.dropout(self.position_embeddings(positions).unsqueeze(1))

        # Ensure x has the correct shape
        if x.ndim == 2:  # Shape: [batch, hidden_dim]
            x = x.unsqueeze(1)  # Shape: [batch, 1, hidden_dim]
        elif x.ndim == 3:  # Already [batch, 1, hidden_dim]
            pass
        else:
            raise ValueError(f"Unexpected input shape for x: {x.shape}")

        assert (
            x.shape == player_embed.shape
        ), f"Shape mismatch: x ({x.shape}) and player_embed ({player_embed.shape})"
        assert (
            x.shape == action_embed.shape
        ), f"Shape mismatch: x ({x.shape}) and action_embed ({action_embed.shape})"
        assert (
            x.shape == position_embed.shape
        ), f"Shape mismatch: x ({x.shape}) and position_embed ({position_embed.shape})"

        # Add embeddings
        x = x + player_embed + action_embed + position_embed

        # Apply projection and normalization
        x = self.layer_norm(self.projection(x))

        return x
