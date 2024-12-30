import torch
import torch.nn as nn


class PokerTransformerModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_heads, num_layers, max_seq_len=1
    ):
        """
        Transformer-based model for predicting poker actions.
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Dimension of transformer embeddings.
            output_dim (int): Number of output classes ("fold", "call", "raise").
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            max_seq_len (int): Length of the input sequence (set to 1 for this dataset).
        """
        super(PokerTransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
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
        # x: [batch_size, input_dim]
        # Embed the input and add positional encoding
        # [batch_size, input_dim] -> [batch_size, hidden_dim]
        x = self.embedding(x)
        x = self.positional_encoding(x)  # Add positional encoding

        # Add a sequence dimension for transformer encoder
        # [batch_size, hidden_dim] -> [batch_size, seq_len=1, hidden_dim]
        x = x.unsqueeze(1)

        # Transformer expects [seq_len, batch_size, hidden_dim]
        # [batch_size, seq_len, hidden_dim] -> [seq_len, batch_size, hidden_dim]
        x = x.permute(1, 0, 2)

        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # [seq_len, batch_size, hidden_dim]

        # Aggregate sequence output (since seq_len=1, this is straightforward)
        # [seq_len=1, batch_size, hidden_dim] -> [batch_size, hidden_dim]
        x = x.mean(dim=0)

        # Pass through classifier
        # [batch_size, hidden_dim] -> [batch_size, output_dim]
        x = self.classifier(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=1):
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
        return x + self.positional_encoding[:, : x.size(1), :]
