import torch
import torch.nn as nn
from typing import Tuple

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """
        Initializes the Self Attention Block.
        
        Args:
            embed_dim (int): Dimensionality of the input embeddings.
            num_heads (int): Number of parallel attention heads.
            dropout (float): Dropout rate applied after the attention mechanism.
        """
        super(SelfAttentionBlock, self).__init__()
        
        # Multihead Attention layer
        # batch_first=True ensures input shape is (Batch, Seq_Len, Embed_Dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the attention block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, embed_dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, embed_dim).
        """
        # Apply MultiheadAttention
        # query, key, value are all 'x' for self-attention
        attn_output, _ = self.attention(query=x, key=x, value=x)
        
        # Apply dropout to the attention output
        attn_output = self.dropout(attn_output)
        
        # Add residual connection (input x + dropout output)
        residual = x + attn_output
        
        # Apply Layer Normalization
        output = self.layer_norm(residual)
        
        return output
    
# Assuming SelfAttentionBlock is defined in the same scope or imported from the previous step
# If not, ensure the definition provided earlier is available.

class ProcessTransformer(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        embed_dim: int = 64, 
        num_heads: int = 4, 
        num_layers: int = 2, 
        num_activities: int = 10, 
        dropout: float = 0.1, 
        max_len: int = 50
    ):
        """
        Initializes the Process Transformer model.
        
        Args:
            vocab_size (int): Size of the vocabulary (excluding padding index).
            embed_dim (int): Dimensionality of the embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            num_activities (int): Number of unique activity classes for classification.
            dropout (float): Dropout rate.
            max_len (int): Maximum sequence length (used for positional encoding if added, 
                           though not explicitly requested here, it's stored for context).
        """
        super(ProcessTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_activities = num_activities
        
        # Embedding layer: (vocab_size + 1) to account for padding index 0
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size + 1, 
            embedding_dim=embed_dim, 
            padding_idx=0
        )
        
        # Stack of SelfAttentionBlocks
        self.layers = nn.ModuleList([
            SelfAttentionBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                dropout=dropout
            ) 
            for _ in range(num_layers)
        ])
        
        # Activity prediction head: Linear from embed_dim to num_activities
        self.activity_head = nn.Linear(embed_dim, num_activities)
        
        # Remaining time prediction head: Linear from embed_dim to 1
        self.time_head = nn.Linear(embed_dim, 1)
        
        # Store max_len for potential future use (e.g., positional encodings)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len) containing token indices.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - activity_logits: Tensor of shape (batch, num_activities)
                - time_prediction: Tensor of shape (batch, 1)
        """
        batch_size, seq_len = x.shape
        
        # 1. Apply Embedding
        # Output shape: (batch, seq_len, embed_dim)
        x_embed = self.embedding(x)
        
        # 2. Pass through SelfAttention Blocks sequentially
        for layer in self.layers:
            x_embed = layer(x_embed)
        
        # 3. Global Pooling: Mean over the sequence dimension
        # Reduce dim 1 (seq_len) to get a single vector per sample
        # Shape becomes: (batch, embed_dim)
        pooled_output = x_embed.mean(dim=1)
        
        # 4. Pass through Output Heads
        # Activity Logits
        activity_logits = self.activity_head(pooled_output)  # Shape: (batch, num_activities)
        
        # Time Prediction
        time_prediction = self.time_head(pooled_output)      # Shape: (batch, 1)
        
        return activity_logits, time_prediction