import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceActivityPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, time_hidden_dim=1):
        super(SequenceActivityPredictor, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer for activity indices with padding_idx=0
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        
        # Custom GRU cell with update gate, reset gate, and candidate hidden state
        self.W_z = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        
        # Output layers
        self.activity_output = nn.Linear(hidden_dim, vocab_size)
        self.time_output = nn.Linear(hidden_dim, 1)
    
    def forward(self, activity_indices):
        """
        Forward pass processing sequence step by step through GRU.
        
        Args:
            activity_indices: LongTensor of shape (batch_size, seq_len)
            
        Returns:
            activity_logits: FloatTensor of shape (batch_size, vocab_size)
            time_prediction: FloatTensor of shape (batch_size, 1)
        """
        batch_size, seq_len = activity_indices.shape
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=activity_indices.device)
        
        # Get embeddings for all sequences
        embedded = self.embedding(activity_indices)  # (batch_size, seq_len, embed_dim)
        
        # Process sequence step by step through GRU
        for t in range(seq_len):
            x_t = embedded[:, t, :]  # (batch_size, embed_dim)
            
            # Concatenate current input with previous hidden state
            concat = torch.cat([x_t, h], dim=1)  # (batch_size, embed_dim + hidden_dim)
            
            # Update gate
            z = torch.sigmoid(self.W_z(concat))
            
            # Reset gate
            r = torch.sigmoid(self.W_r(concat))
            
            # Candidate hidden state
            r_concat = torch.cat([r * x_t, h], dim=1)  # (batch_size, embed_dim + hidden_dim)
            h_tilde = torch.tanh(self.W_h(r_concat))
            
            # New hidden state
            h = z * h + (1 - z) * h_tilde
        
        # Use final hidden state for predictions
        activity_logits = self.activity_output(h)  # (batch_size, vocab_size)
        time_prediction = self.time_output(h)  # (batch_size, 1)
        
        return activity_logits, time_prediction