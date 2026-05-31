import torch
import torch.nn as nn

class MyktbekModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_filters=128, num_activities=10, dropout=0.1):
        super(MyktbekModel, self).__init__()
        
        # Embedding layer with padding_idx=0 and num_embeddings=vocab_size+1
        self.embedding = nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=embed_dim, padding_idx=0)
        
        # Three parallel branches with sliding window convolutions of sizes 3, 5, and 7
        # Input to conv: (batch, embed_dim, seq_len) after embedding
        # We use Conv1d with kernel_size for sliding window along time axis
        self.conv3 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=7, padding=3)
        
        # Batch normalization for each branch
        self.bn3 = nn.BatchNorm1d(num_filters)
        self.bn5 = nn.BatchNorm1d(num_filters)
        self.bn7 = nn.BatchNorm1d(num_filters)
        
        # Nonlinear activation (ReLU)
        self.relu = nn.ReLU()
        
        # Global max-pooling across time axis for each branch
        # After convolution, shape is (batch, num_filters, seq_len)
        # We'll use adaptive max pooling to get fixed size output regardless of sequence length
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers for outputs
        # After concatenation: 3 * num_filters features
        self.fc_activity = nn.Linear(3 * num_filters, num_activities)
        self.fc_time = nn.Linear(3 * num_filters, 1)
    
    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch, seq_len) containing encoded activity integers
        
        Returns:
            activity_logits: tensor of shape (batch, num_activities) for activity classification
            time_prediction: tensor of shape (batch, 1) for remaining time regression
        """
        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # Transpose to (batch, embed_dim, seq_len) for Conv1d
        embedded = embedded.transpose(1, 2)
        
        # Branch 1: kernel size 3
        out3 = self.conv3(embedded)
        out3 = self.bn3(out3)
        out3 = self.relu(out3)
        out3 = self.pool(out3)  # (batch, num_filters, 1)
        out3 = out3.squeeze(-1)  # (batch, num_filters)
        
        # Branch 2: kernel size 5
        out5 = self.conv5(embedded)
        out5 = self.bn5(out5)
        out5 = self.relu(out5)
        out5 = self.pool(out5)  # (batch, num_filters, 1)
        out5 = out5.squeeze(-1)  # (batch, num_filters)
        
        # Branch 3: kernel size 7
        out7 = self.conv7(embedded)
        out7 = self.bn7(out7)
        out7 = self.relu(out7)
        out7 = self.pool(out7)  # (batch, num_filters, 1)
        out7 = out7.squeeze(-1)  # (batch, num_filters)
        
        # Concatenate all three branches: (batch, 3 * num_filters)
        concatenated = torch.cat([out3, out5, out7], dim=1)
        
        # Apply dropout
        dropped = self.dropout(concatenated)
        
        # Output layers
        activity_logits = self.fc_activity(dropped)  # (batch, num_activities)
        time_prediction = self.fc_time(dropped)  # (batch, 1)
        
        return activity_logits, time_prediction