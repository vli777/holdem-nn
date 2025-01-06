import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, hidden_dim]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        x = x + self.pe[:, : x.size(1), :]
        return x


class PokerTransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        seq_len,
        num_heads,
        num_layers,
        num_players=6,
        max_positions=10,
        num_actions=3,
        num_strategies=3,
        dropout=0.1,
    ):
        """
        Transformer-based model for predicting poker actions at each round.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Dimension of transformer embeddings.
            output_dim (int): Number of output classes ("fold", "call", "raise").
            seq_len (int): Maximum sequence length.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            num_players (int): Number of players for embedding.
            max_positions (int): Maximum positions for embedding.
            num_actions (int): Number of actions for embedding.
            num_strategies (int): Number of aggression strategies.
            dropout (float): Dropout rate.
        """
        super(PokerTransformerModel, self).__init__()

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Embedding layers
        self.player_embedding = nn.Embedding(num_players, hidden_dim)
        self.position_embedding = nn.Embedding(max_positions, hidden_dim)
        self.action_embedding = nn.Embedding(num_actions, hidden_dim)
        self.strategy_embedding = nn.Embedding(num_strategies, hidden_dim)
        self.bluffing_embedding = nn.Linear(1, hidden_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Action prediction head
        self.policy_head = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x,
        player_ids,
        positions,
        recent_actions,
        strategies,
        bluffing_probabilities,
        mask=None,
    ):
        """
        Forward pass for predicting actions at each round.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].
            player_ids (torch.Tensor): Player IDs of shape [batch_size, seq_len].
            positions (torch.Tensor): Player positions of shape [batch_size, seq_len].
            recent_actions (torch.Tensor): Recent actions of shape [batch_size, seq_len].
            mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Policy logits of shape [batch_size, seq_len, output_dim].
        """
        # Project inputs to hidden dimension
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]

        # Embed player-specific features
        player_embeds = self.player_embedding(
            player_ids
        )  # [batch_size, seq_len, hidden_dim]
        position_embeds = self.position_embedding(
            positions
        )  # [batch_size, seq_len, hidden_dim]
        action_embeds = self.action_embedding(
            recent_actions
        )  # [batch_size, seq_len, hidden_dim]
        strategy_embeds = self.strategy_embedding(
            strategies
        )  # [batch_size, seq_len, hidden_dim]
        bluffing_embeds = self.bluffing_embedding(
            bluffing_probabilities.unsqueeze(-1)
        )  # [batch_size, seq_len, hidden_dim]

        # Combine embeddings
        x = (
            x
            + player_embeds
            + position_embeds
            + action_embeds
            + strategy_embeds
            + bluffing_embeds
        )  # [batch_size, seq_len, hidden_dim]

        # Add positional encoding
        x = self.positional_encoding(x)  # [batch_size, seq_len, hidden_dim]

        # Transformer expects [seq_len, batch_size, hidden_dim]
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, hidden_dim]

        # Create transformer masks
        if mask is not None:
            src_key_padding_mask = ~mask  # [batch_size, seq_len]
        else:
            src_key_padding_mask = None

        # Pass through transformer encoder
        x = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # [seq_len, batch_size, hidden_dim]

        # Transformer output back to [batch_size, seq_len, hidden_dim]
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim]

        # Predict actions at each step
        policy_logits = self.policy_head(x)  # [batch_size, seq_len, output_dim]

        return policy_logits
