import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple, Any
from typing import Dict, Tuple

def train_model(
    model: nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    num_epochs: int, 
    learning_rate: float, 
    device: str
) -> List[float]:
    """
    Trains the ProcessTransformer model for a specified number of epochs.
    
    Args:
        model (nn.Module): The ProcessTransformer model to train.
        train_loader (DataLoader): DataLoader yielding batches of (x, y_activity, y_time).
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the Adam optimizer.
        device (str): Device string (e.g., 'cpu' or 'cuda').
        
    Returns:
        List[float]: A list containing the average combined loss for each epoch.
    """
    # Set up Optimizer and Loss Functions
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Activity loss: CrossEntropyLoss (expects raw logits and class indices)
    criterion_activity = nn.CrossEntropyLoss()
    
    # Time loss: MSELoss
    criterion_time = nn.MSELoss()
    
    # Move model to device
    model.to(device)
    
    epoch_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Unpack batch: (x, y_activity, y_time)
            x, y_activity, y_time = batch
            
            # Move tensors to device
            x = x.to(device)
            y_activity = y_activity.to(device)
            y_time = y_time.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            activity_logits, time_prediction = model(x)
            
            # Compute losses
            # activity_logits shape: (batch, num_activities)
            # y_activity shape: (batch,) - integer class indices
            loss_activity = criterion_activity(activity_logits, y_activity.long())
            
            # time_prediction shape: (batch, 1)
            # y_time shape: (batch, 1) or (batch,) depending on loader, ensure match
            # If y_time is (batch,), unsqueeze to (batch, 1) for MSELoss consistency if needed
            if y_time.dim() == 1:
                y_time = y_time.unsqueeze(1)
                
            loss_time = criterion_time(time_prediction, y_time.float())
            
            # Combined loss (equal weight)
            loss = loss_activity + loss_time
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss for the epoch
        avg_epoch_loss = total_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Average Combined Loss: {avg_epoch_loss:.4f}")
        
    return epoch_losses

def create_data_loader(
    prefixes: pd.DataFrame, 
    encoded_x: torch.Tensor, 
    vocab: Dict[str, int], 
    batch_size: int = 32, 
    shuffle: bool = True
) -> Tuple[DataLoader, float, float]:
    """
    Creates a DataLoader for training, including normalization of the target 'remaining_time'.
    
    Args:
        prefixes (pd.DataFrame): DataFrame containing 'next_activity' and 'remaining_time' columns.
        encoded_x (torch.Tensor): Encoded input sequences of shape (N, max_len).
        vocab (dict): Vocabulary mapping activity names to indices.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        
    Returns:
        Tuple[DataLoader, float, float]: 
            - DataLoader yielding batches of (x, y_activity, y_time_normalized).
            - Mean of the original remaining_time values.
            - Standard deviation of the original remaining_time values.
    """
    # 1. Create y_activity (Next Activity Labels)
    next_activities = prefixes['next_activity'].apply(lambda x: vocab.get(x, 0))
    y_activity = torch.tensor(next_activities.values, dtype=torch.long)
    
    # 2. Prepare y_time for Normalization
    raw_times = prefixes['remaining_time'].values.astype(float)
    
    # Compute Mean and Standard Deviation
    mean_val = raw_times.mean()
    std_val = raw_times.std()
    
    # Handle case where std might be 0 (constant values) to avoid division by zero
    if std_val == 0:
        std_val = 1.0
    
    # Normalize: (value - mean) / std
    normalized_times = (raw_times - mean_val) / std_val
    y_time = torch.tensor(normalized_times, dtype=torch.float32)
    
    # Ensure y_time has shape (N, 1)
    if y_time.dim() == 1:
        y_time = y_time.unsqueeze(1)
    
    # 3. Combine into TensorDataset
    dataset = TensorDataset(encoded_x, y_activity, y_time)
    
    # 4. Create and return DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader, mean_val, std_val