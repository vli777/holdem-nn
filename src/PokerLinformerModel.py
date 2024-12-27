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
        # Linear embedding for input features
        self.embedding = nn.Linear(input_dim, hidden_dim)

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
        # x: [batch_size, input_dim]
        x = self.embedding(x)  # [batch_size, input_dim] -> [batch_size, hidden_dim]
        x = self.linformer(x)  # [batch_size, hidden_dim] -> [batch_size, hidden_dim]
        x = self.classifier(x.mean(dim=1))  # Aggregate across sequence (if seq_len > 1)
        return x
