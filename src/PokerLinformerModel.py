from linformer import Linformer
import torch.nn as nn
import torch
import math


class PokerLinformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 seq_len, num_heads, num_layers, num_players=6):
        """
        Linformer-based model for poker predictions.
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Dimension of the transformer embeddings.
            output_dim (int): Number of output classes (e.g., "fold", "call", "raise").
            seq_len (int): Sequence length (fixed, e.g., 1 for poker state).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Linformer layers.
            num_players (int): Number of players.
        """
        super().__init__()
        # Projection layer to match Linformer input dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Player Dynamics Attention
        self.player_dynamics = PlayerDynamicsAttention(hidden_dim, num_players)

        # Transformer Encoder Layer for richer positional/contextual
        # information
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=1  # Can be tuned for complexity
        )

        # Linformer encoder
        self.linformer = Linformer(
            dim=hidden_dim,
            seq_len=seq_len,
            depth=num_layers,
            heads=num_heads,
            k=32,  # Compression factor (lower means more efficient)
            one_kv_head=True,  # Share key-value projection
            share_kv=True,     # Share key-value weights across heads
        )

        # Policy (Action probabilities)
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)  # Value (Win probability)

    def forward(self, x, positions=None, player_ids=None, recent_actions=None):
        device = x.device
        if positions is not None:
            positions = positions.to(device)
        if player_ids is not None:
            player_ids = player_ids.to(device)
        if recent_actions is not None:
            recent_actions = recent_actions.to(device)

        # Incorporate additional features into the computation if necessary
        if positions is not None and player_ids is not None and recent_actions is not None:
            x = self.player_dynamics(x, positions, player_ids, recent_actions)

        # Ensure x has shape [batch_size, seq_len, embedding_dim]
        if x.ndim == 2:  # If input is [batch_size, embedding_dim]
            # Add sequence length dimension, making it [batch_size, 1,
            # embedding_dim]
            x = x.unsqueeze(1)

        x = self.input_projection(x)  # Project input_dim -> hidden_dim

        # Process through Transformer Encoder Layer for richer embeddings
        # Transformer expects [seq_len, batch_size, hidden_dim]
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        # Convert back to [batch_size, seq_len, hidden_dim]
        x = x.permute(1, 0, 2)

        # Process through Linformer encoder
        x = self.linformer(x)

        # Remove the sequence length dimension after Linformer
        x = x.squeeze(1)

        # Separate outputs for policy and value
        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value


class PlayerDynamicsAttention(nn.Module):
    def __init__(self, hidden_dim, num_players,
                 num_actions=3, max_positions=10):
        super().__init__()
        self.player_embeddings = nn.Embedding(num_players, hidden_dim)
        self.action_embeddings = nn.Embedding(num_actions, hidden_dim)
        self.position_embeddings = nn.Embedding(max_positions, hidden_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.player_embeddings.weight)
        nn.init.xavier_uniform_(self.action_embeddings.weight)
        nn.init.xavier_uniform_(self.position_embeddings.weight)

    def forward(self, x, player_ids, actions, positions):
        # Clamp positions to valid range
        max_pos = self.position_embeddings.num_embeddings - 1
        positions = torch.clamp(positions, 0, max_pos)

        # Ensure valid indices for actions
        num_actions = self.action_embeddings.num_embeddings
        if actions.min() < 0 or actions.max() >= num_actions:
            raise ValueError(f"Invalid action indices: {actions}")

        # Move tensors to the same device
        device = x.device
        player_ids = player_ids.to(device)
        actions = actions.to(device)
        positions = positions.to(device)

        # Embedding lookups
        player_embed = self.player_embeddings(player_ids)
        action_embed = self.action_embeddings(actions)
        position_embed = self.dropout(self.position_embeddings(positions))

        return x + player_embed + action_embed + position_embed
