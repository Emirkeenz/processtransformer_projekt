import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_myktbek_model(model, train_loader, num_epochs, learning_rate, device):
    """
    Trains the MyktbekModel for activity prediction and remaining time regression.
    
    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader yielding batches of (x, y_activity, y_time).
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (str): Device to train on (e.g., 'cpu' or 'cuda').
    
    Returns:
        list: Average combined loss per epoch.
    """
    # Move model to device
    model.to(device)
    
    # Initialize Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define loss functions
    criterion_activity = nn.CrossEntropyLoss()
    criterion_time = nn.MSELoss()
    
    # List to store average losses per epoch
    epoch_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y_activity, batch_y_time in train_loader:
            # Move tensors to device
            batch_x = batch_x.to(device)
            batch_y_activity = batch_y_activity.to(device)
            batch_y_time = batch_y_time.to(device)
            
            # Handle y_time shape: if (batch,), unsqueeze to (batch, 1)
            if batch_y_time.dim() == 1:
                batch_y_time = batch_y_time.unsqueeze(1)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            activity_logits, time_prediction = model(batch_x)
            
            # Compute losses
            loss_activity = criterion_activity(activity_logits, batch_y_activity)
            loss_time = criterion_time(time_prediction, batch_y_time)
            
            # Sum losses
            total_batch_loss = loss_activity + loss_time
            
            # Backward pass and optimize
            total_batch_loss.backward()
            optimizer.step()
            
            # Accumulate loss
            total_loss += total_batch_loss.item()
            num_batches += 1
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        
        # Print average loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return epoch_losses