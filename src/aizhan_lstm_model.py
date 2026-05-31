import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128, num_layers=2, num_activities=None, dropout=0.2):
        super(LSTMModel, self).__init__()

        # 1. Embedding layer with padding_idx=0
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)

        # 2. Recurrent layer (LSTM) - handles input, forget, and output gates internally
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 3 & 4. Dropout after extracting last time step
        self.dropout = nn.Dropout(dropout)

        # 5. Two parallel output heads
        self.activity_head = nn.Linear(hidden_size, num_activities)
        self.time_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length)

        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_dim)

        # LSTM processing
        lstm_out, _ = self.lstm(embedded)  # lstm_out: (batch_size, seq_length, hidden_size)

        # Take only the last time step
        last_step = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Apply dropout
        last_step = self.dropout(last_step)

        # Parallel heads
        activity_logits = self.activity_head(last_step)  # (batch_size, num_activities)
        time_pred = self.time_head(last_step)  # (batch_size, 1)

        # Squeeze time_pred to (batch_size,)
        time_pred = time_pred.squeeze(-1)

        return activity_logits, time_pred