import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomGRUCell(nn.Module):
    """
    Implements the specific recurrent logic requested:
    - Update gate (z) = sigmoid(linear(input) + linear(hidden))
    - Reset gate (r) = sigmoid(linear(input) + linear(hidden))
    - Candidate hidden state (n) = tanh(linear(input) + linear(r * hidden))
    - New hidden state (h) = (1 - z) * h_prev + z * n
    """
    def __init__(self, input_size, hidden_size):
        super(CustomGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Gates parameters
        # Weights for update gate (z)
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Weights for reset gate (r)
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Weights for candidate hidden state (n)
        self.W_n = nn.Linear(input_size, hidden_size)
        self.U_n = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h_prev):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size)
            h_prev: Previous hidden state of shape (batch_size, hidden_size)
        Returns:
            h_new: New hidden state of shape (batch_size, hidden_size)
        """
        # 1. Calculate Update Gate (z)
        # z = sigmoid(W_z * x + U_z * h_prev)
        z = torch.sigmoid(self.W_z(x) + self.U_z(h_prev))
        
        # 2. Calculate Reset Gate (r)
        # r = sigmoid(W_r * x + U_r * h_prev)
        r = torch.sigmoid(self.W_r(x) + self.U_r(h_prev))
        
        # 3. Calculate Candidate Hidden State (n)
        # n = tanh(W_n * x + U_n * (r * h_prev))
        # Note: The reset gate is applied element-wise to the previous hidden state
        reset_h = r * h_prev
        n = torch.tanh(self.W_n(x) + self.U_n(reset_h))
        
        # 4. Calculate New Hidden State (h)
        # h = (1 - z) * h_prev + z * n
        h_new = (1 - z) * h_prev + z * n
        
        return h_new

class SequenceActivityPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_time_features, dropout=0.1):
        super(SequenceActivityPredictor, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_time_features = num_time_features
        
        # 1. Embedding Layer: Converts integer activity IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Custom Recurrent Layer (GRU logic)
        # The input to the RNN will be: [embedded_activity_vector, time_features]
        rnn_input_size = embed_dim + num_time_features
        self.rnn_cell = CustomGRUCell(input_size=rnn_input_size, hidden_size=hidden_size)
        
        # 2. Output Layers
        # Task A: Next Activity Prediction (Classification) -> Logits
        # Input size: hidden_size
        self.fc_activity = nn.Linear(hidden_size, vocab_size)
        
        # Task B: Remaining Time Prediction (Regression) -> Scalar
        # Input size: hidden_size
        self.fc_time = nn.Linear(hidden_size, 1)
        
        # Optional Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, activity_indices, time_features, initial_hidden=None):
        """
        Args:
            activity_indices: Tensor of shape (seq_len, batch_size) containing integer activity IDs.
            time_features: Tensor of shape (seq_len, batch_size, num_time_features).
            initial_hidden: Optional tensor of shape (batch_size, hidden_size). If None, zeros are used.
            
        Returns:
            activity_logits: Tensor of shape (seq_len, batch_size, vocab_size)
            time_predictions: Tensor of shape (seq_len, batch_size, 1)
        """
        seq_len, batch_size = activity_indices.shape
        
        if initial_hidden is None:
            h = torch.zeros(batch_size, self.hidden_size, device=activity_indices.device)
        else:
            h = initial_hidden
            
        activity_logits_list = []
        time_preds_list = []
        
        # Process sequence step-by-step
        for t in range(seq_len):
            # Get current inputs
            act_idx = activity_indices[t]  # (batch_size,)
            time_feat = time_features[t]   # (batch_size, num_time_features)
            
            # Embed activity ID
            embedded_act = self.embedding(act_idx)  # (batch_size, embed_dim)
            
            # Concatenate embedding and time features
            rnn_input = torch.cat([embedded_act, time_feat], dim=1)  # (batch_size, embed_dim + num_time_features)
            
            # Apply Dropout before RNN input (optional, common practice)
            rnn_input = self.dropout(rnn_input)
            
            # Update hidden state using custom GRU cell
            h = self.rnn_cell(rnn_input, h)
            
            # Generate outputs based on the new hidden state
            # We predict the NEXT activity and next time value based on current state
            out_activity = self.fc_activity(h)      # (batch_size, vocab_size)
            out_time = self.fc_time(h)              # (batch_size, 1)
            
            activity_logits_list.append(out_activity)
            time_preds_list.append(out_time)
            
        # Stack results along sequence dimension
        activity_logits = torch.stack(activity_logits_list, dim=0)  # (seq_len, batch_size, vocab_size)
        time_predictions = torch.stack(time_preds_list, dim=0)      # (seq_len, batch_size, 1)
        
        return activity_logits, time_predictions

# Example Usage / Test
if __name__ == "__main__":
    # Configuration
    VOCAB_SIZE = 100       # Number of unique activity IDs
    EMBED_DIM = 64         # Dimension of embedding vector
    HIDDEN_SIZE = 128      # Size of hidden state
    NUM_TIME_FEATURES = 5  # e.g., hour, day_of_week, duration, etc.
    BATCH_SIZE = 32
    SEQ_LEN = 10
    
    # Initialize Model
    model = SequenceActivityPredictor(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_size=HIDDEN_SIZE,
        num_time_features=NUM_TIME_FEATURES
    )
    
    # Create Dummy Inputs
    # activity_indices: Random integers between 0 and VOCAB_SIZE-1
    activity_indices = torch.randint(0, VOCAB_SIZE, (SEQ_LEN, BATCH_SIZE))
    # time_features: Random floats representing time features
    time_features = torch.randn(SEQ_LEN, BATCH_SIZE, NUM_TIME_FEATURES)
    
    # Forward Pass
    logits, time_pred = model(activity_indices, time_features)
    
    print(f"Input Activity Indices Shape: {activity_indices.shape}")
    print(f"Input Time Features Shape: {time_features.shape}")
    print(f"Output Activity Logits Shape: {logits.shape}")
    print(f"Output Time Predictions Shape: {time_pred.shape}")
    
    # Verify shapes match expectations
    assert logits.shape == (SEQ_LEN, BATCH_SIZE, VOCAB_SIZE), "Activity logit shape mismatch"
    assert time_pred.shape == (SEQ_LEN, BATCH_SIZE, 1), "Time prediction shape mismatch"
    
    print("\nModel structure:")
    print(model)