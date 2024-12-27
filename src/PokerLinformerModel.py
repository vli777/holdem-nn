from linformer import Linformer
import torch.nn as nn


class PokerLinformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, num_heads, num_layers):
        """
        Linformer-based model for poker predictions.
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Dimension of the transformer embeddings.
            output_dim (int): Number of output classes (e.g., "fold", "call", "raise").
            seq_len (int): Sequence length (fixed, e.g., 1 for poker state).
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Linformer layers.
        """
        super().__init__()

        # Projection layer to match Linformer input dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)

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

        # Classification head
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Ensure x has shape [batch_size, seq_len, embedding_dim]
        if x.ndim == 2:  # If input is [batch_size, embedding_dim]
            # Add sequence length dimension, making it [batch_size, 1, embedding_dim]
            x = x.unsqueeze(1)

        x = self.input_projection(x)  # Project input_dim -> hidden_dim
        x = self.linformer(x)  # [batch_size, seq_len, hidden_dim]
        # Remove the sequence length dimension after Linformer
        x = x.squeeze(1)
        x = self.classifier(x)  # Final linear layer
        return x
