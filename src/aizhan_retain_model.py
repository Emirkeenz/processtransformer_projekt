import torch
import torch.nn as nn


class RETAINModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128, dropout=0.1):
        super(RETAINModel, self).__init__()

        # Embedding layer with padding_idx=0
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        # First GRU for alpha weights (scalar attention)
        self.gru_alpha = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, batch_first=True)
        self.linear_alpha = nn.Linear(hidden_size, 1)

        # Second GRU for beta weights (vector attention)
        self.gru_beta = nn.GRU(input_size=embed_dim, hidden_size=hidden_size, batch_first=True)
        self.linear_beta = nn.Linear(hidden_size, embed_dim)

        # Dropout for context vector
        self.dropout = nn.Dropout(dropout)

        # Output heads
        self.fc_activity = nn.Linear(embed_dim, vocab_size)
        self.fc_time = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len) containing integer activity indices

        Returns:
            activity_logits: tensor of shape (batch_size, vocab_size)
            time_prediction: tensor of shape (batch_size, 1)
        """
        batch_size, seq_len = x.shape

        # Step 1: Embed the input
        embedded = self.embedding(x)  # Shape: (batch_size, seq_len, embed_dim)

        # Step 2: Reverse the sequence along the time dimension
        embedded_reversed = embedded.flip(dims=[1])  # Shape: (batch_size, seq_len, embed_dim)

        # Step 3: Process with first GRU for alpha weights
        gru_alpha_out, _ = self.gru_alpha(embedded_reversed)  # Shape: (batch_size, seq_len, hidden_size)
        alpha_scores = self.linear_alpha(gru_alpha_out)  # Shape: (batch_size, seq_len, 1)
        alpha_weights = torch.softmax(alpha_scores, dim=1)  # Shape: (batch_size, seq_len, 1)

        # Step 4: Process with second GRU for beta weights
        gru_beta_out, _ = self.gru_beta(embedded_reversed)  # Shape: (batch_size, seq_len, hidden_size)
        beta_vectors = torch.tanh(self.linear_beta(gru_beta_out))  # Shape: (batch_size, seq_len, embed_dim)

        # Step 5: Compute context vector
        # We need to multiply alpha_weights (b, s, 1) with beta_vectors (b, s, d) and sum over time
        # First, expand alpha_weights to match beta_vectors dimensions for element-wise multiplication
        weighted_beta = alpha_weights * beta_vectors  # Broadcasting: (batch_size, seq_len, embed_dim)
        context_vector = torch.sum(weighted_beta, dim=1)  # Shape: (batch_size, embed_dim)

        # Step 6: Apply dropout
        context_vector = self.dropout(context_vector)

        # Step 7: Parallel output heads
        activity_logits = self.fc_activity(context_vector)  # Shape: (batch_size, vocab_size)
        time_prediction = self.fc_time(context_vector)  # Shape: (batch_size, 1)

        return activity_logits, time_prediction