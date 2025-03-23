import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput


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
        Transformer-based model for predicting poker actions at each round using Hugging Face transformers.

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

        # Configure Hugging Face transformer
        config = AutoConfig.from_pretrained(
            "bert-base-uncased",
            num_labels=output_dim,
            hidden_size=hidden_dim,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=seq_len,
            type_vocab_size=1,
        )
        
        # Initialize the transformer model
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            config=config,
            ignore_mismatched_sizes=True
        )

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
            strategies (torch.Tensor): Player strategies of shape [batch_size, seq_len].
            bluffing_probabilities (torch.Tensor): Bluffing probabilities of shape [batch_size, seq_len].
            mask (torch.Tensor, optional): Attention mask of shape [batch_size, seq_len].

        Returns:
            torch.Tensor: Policy logits of shape [batch_size, seq_len, output_dim].
        """
        # Project inputs to hidden dimension
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]

        # Embed player-specific features
        player_embeds = self.player_embedding(player_ids)  # [batch_size, seq_len, hidden_dim]
        position_embeds = self.position_embedding(positions)  # [batch_size, seq_len, hidden_dim]
        action_embeds = self.action_embedding(recent_actions)  # [batch_size, seq_len, hidden_dim]
        strategy_embeds = self.strategy_embedding(strategies)  # [batch_size, seq_len, hidden_dim]
        bluffing_embeds = self.bluffing_embedding(bluffing_probabilities.unsqueeze(-1))  # [batch_size, seq_len, hidden_dim]

        # Combine embeddings
        x = x + player_embeds + position_embeds + action_embeds + strategy_embeds + bluffing_embeds

        # Create attention mask for transformer
        attention_mask = mask if mask is not None else torch.ones_like(player_ids, dtype=torch.bool)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]

        # Pass through transformer
        outputs = self.transformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Get logits from the transformer output
        logits = outputs.logits  # [batch_size, seq_len, output_dim]

        return logits
