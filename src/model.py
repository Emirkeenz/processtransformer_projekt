import torch
import torch.nn as nn

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