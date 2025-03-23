import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple


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
        input_dim: int = 52,  # 52 cards
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_position_embeddings: int = 100,
        num_players: int = 6,
        num_positions: int = 6,
        num_actions: int = 3,
        num_strategies: int = 3,
        output_dim: int = 3,  # fold, call, raise
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
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.num_players = num_players
        self.num_positions = num_positions
        self.num_actions = num_actions
        self.num_strategies = num_strategies
        self.output_dim = output_dim

        # Input projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Embedding layers
        self.player_embedding = nn.Embedding(num_players, hidden_dim)
        self.position_embedding = nn.Embedding(num_positions, hidden_dim)
        self.action_embedding = nn.Embedding(num_actions, hidden_dim)
        self.strategy_embedding = nn.Embedding(num_strategies, hidden_dim)
        
        # Layer normalization for embeddings
        self.embed_norm = nn.LayerNorm(hidden_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_position_embeddings, hidden_dim))
        
        # Initialize Hugging Face transformer
        config = AutoConfig.from_pretrained(
            "bert-base-uncased",
            hidden_size=hidden_dim,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=1,
            vocab_size=1,
        )
        self.transformer = AutoModelForSequenceClassification.from_config(config)
        
        # Policy head (action prediction)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Value head (state value prediction)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Bound value predictions
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if "weight" in name and "norm" not in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        player_ids: torch.Tensor,
        positions: torch.Tensor,
        recent_actions: torch.Tensor,
        strategies: torch.Tensor,
        bluffing_probabilities: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            player_ids: Player ID indices of shape [batch_size, seq_len]
            positions: Position indices of shape [batch_size, seq_len]
            recent_actions: Recent action indices of shape [batch_size, seq_len]
            strategies: Strategy indices of shape [batch_size, seq_len]
            bluffing_probabilities: Bluffing probabilities of shape [batch_size, seq_len]
            mask: Attention mask of shape [batch_size, seq_len]

        Returns:
            Tuple of (policy_logits, value_pred)
        """
        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_proj(x)

        # Get embeddings
        player_embeds = self.player_embedding(player_ids)
        position_embeds = self.position_embedding(positions)
        action_embeds = self.action_embedding(recent_actions)
        strategy_embeds = self.strategy_embedding(strategies)

        # Combine embeddings with input
        x = x + player_embeds + position_embeds + action_embeds + strategy_embeds
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Normalize embeddings
        x = self.embed_norm(x)

        # Create attention mask for transformer
        if mask is None:
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=x.device)
        
        # Get transformer outputs
        outputs = self.transformer(
            inputs_embeds=x,
            attention_mask=mask,
            output_hidden_states=True,
        )
        
        # Get the last hidden state
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        
        # Get policy logits
        policy_logits = self.policy_head(hidden_states)  # [batch_size, seq_len, output_dim]
        
        # Get value predictions (using [CLS] token)
        value_pred = self.value_head(hidden_states[:, 0, :])  # [batch_size, 1]

        return policy_logits, value_pred

    def compute_loss(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        action_targets: torch.Tensor,
        value_targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the combined loss for policy and value predictions.

        Args:
            policy_logits: Policy logits of shape [batch_size, seq_len, output_dim]
            value_pred: Value predictions of shape [batch_size, 1]
            action_targets: Action targets of shape [batch_size, seq_len]
            value_targets: Value targets of shape [batch_size, 1]
            mask: Optional mask of shape [batch_size, seq_len]

        Returns:
            Tuple of (total_loss, policy_loss, value_loss)
        """
        if mask is None:
            mask = torch.ones_like(action_targets, dtype=torch.bool)

        # Policy loss (cross entropy)
        policy_loss = F.cross_entropy(
            policy_logits.view(-1, self.output_dim),
            action_targets.view(-1),
            reduction='none'
        )
        policy_loss = (policy_loss * mask.view(-1)).mean()

        # Value loss (MSE)
        value_loss = F.mse_loss(value_pred, value_targets)

        # Combined loss (weighted sum)
        total_loss = policy_loss + 0.5 * value_loss

        return total_loss, policy_loss, value_loss
