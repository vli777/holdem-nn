import torch
import torch.nn as nn


class PokerTransformerModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_heads, num_layers, max_seq_len=1
    ):
        """
        Transformer-based model for predicting poker actions, scalable for multi-round data.
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Dimension of transformer embeddings.
            output_dim (int): Number of output classes ("fold", "call", "raise").
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            max_seq_len (int): Length of the input sequence (can be >1 for multi-round data).
        """
        super(PokerTransformerModel, self).__init__()

        # Input embedding
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Positional encoding for sequence data
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for the PokerTransformerModel.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        # [batch_size, seq_len, input_dim] -> [batch_size, seq_len, hidden_dim]
        x = self.embedding(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer expects [seq_len, batch_size, hidden_dim]
        # [batch_size, seq_len, hidden_dim] -> [seq_len, batch_size, hidden_dim]
        x = x.permute(1, 0, 2)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)

        # Aggregate sequence output (mean pooling)
        # [seq_len, batch_size, hidden_dim] -> [batch_size, hidden_dim]
        x = x.mean(dim=0)

        # Classification
        # [batch_size, hidden_dim] -> [batch_size, output_dim]
        x = self.classifier(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=5000):
        """
        Positional encoding for Transformer models.
        Args:
            hidden_dim (int): Dimension of the embeddings.
            max_len (int): Maximum sequence length to precompute encodings for.
        """
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension
        self.register_buffer("positional_encoding", self.encoding)

    def forward(self, x):
        # Add positional encoding to input
        return x + self.positional_encoding[:, : x.size(1), :]
