import torch
import torch.nn as nn
from typing import Tuple, Optional

def train_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    device: Optional[str] = None
) -> nn.Module:
    """
    Trains the SequenceActivityPredictor model using Adam optimizer.
    
    Args:
        model (nn.Module): The SequenceActivityPredictor instance to train.
        dataloader (DataLoader): DataLoader yielding batches with:
            - activity_indices: (seq_len, batch_size), LongTensor
            - time_features: (seq_len, batch_size, num_time_features), FloatTensor
            - activity_labels: (seq_len, batch_size), LongTensor
            - time_labels: (seq_len, batch_size), FloatTensor
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the Adam optimizer.
        device (str): Device to run training on ('cpu' or 'cuda'). Defaults to auto-detect.
        
    Returns:
        nn.Module: The trained model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Move model to device
    model = model.to(device)
    
    # Define Loss Functions
    # CrossEntropyLoss expects raw logits (unnormalized) and class indices
    criterion_activity = nn.CrossEntropyLoss()
    # MSELoss for regression
    criterion_time = nn.MSELoss()
    
    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training on {device}...")
    print(f"Epochs: {epochs}, Learning Rate: {learning_rate}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            # Expected keys based on prompt description, handling both dict and tuple unpacking
            if isinstance(batch, dict):
                activity_indices = batch['activity_indices']
                time_features = batch['time_features']
                activity_labels = batch['activity_labels']
                time_labels = batch['time_labels']
            else:
                # Assuming order: activity_indices, time_features, activity_labels, time_labels
                activity_indices, time_features, activity_labels, time_labels = batch
            
            # Move data to device
            activity_indices = activity_indices.to(device)
            time_features = time_features.to(device)
            activity_labels = activity_labels.to(device)
            time_labels = time_labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            # Outputs: (logits_activity, time_preds)
            # logits_activity shape: (seq_len, batch_size, vocab_size)
            # time_preds shape: (seq_len, batch_size, 1)
            logits_activity, time_preds = model(activity_indices, time_features)
            
            # Reshape targets for loss calculation
            # CrossEntropyLoss expects (N, C) where N is total number of elements (seq_len * batch_size)
            # We flatten seq_len and batch_size dimensions
            act_labels_flat = activity_labels.view(-1)          # (seq_len * batch_size,)
            logit_flat = logits_activity.view(-1, logits_activity.size(-1)) # (seq_len * batch_size, vocab_size)
            
            time_labels_flat = time_labels.view(-1, 1)          # (seq_len * batch_size, 1)
            time_pred_flat = time_preds.view(-1, 1)             # (seq_len * batch_size, 1)
            
            # Calculate losses
            loss_act = criterion_activity(logit_flat, act_labels_flat)
            loss_time = criterion_time(time_pred_flat, time_labels_flat)
            
            # Combined loss (equal weighting, can be adjusted)
            loss = loss_act + loss_time
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent exploding gradients (optional but recommended for RNNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
        # Calculate average loss for the epoch
        avg_loss = total_loss / batch_count
        print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_loss:.4f} (Act: {loss_act.item():.4f}, Time: {loss_time.item():.4f})")
        
    return model