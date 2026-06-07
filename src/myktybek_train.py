import torch
import torch.nn as nn

def train_myktbek_model(model, train_loader, num_epochs, learning_rate, device):
    """
    Train MyktbekModel with activity and time outputs.
    
    Args:
        model: MyktbekModel instance
        train_loader: DataLoader yielding (x, y_activity, y_time) batches
        num_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        device: 'cuda' or 'cpu'
    
    Returns:
        List of average losses per epoch
    """
    # Move model to device
    model.to(device)
    
    # Initialize optimizer and loss functions
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    activity_criterion = nn.CrossEntropyLoss()
    time_criterion = nn.MSELoss()
    
    # Track losses per epoch
    epoch_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for x, y_activity, y_time in train_loader:
            # Move data to device
            x = x.to(device)
            y_activity = y_activity.to(device)
            y_time = y_time.to(device)
            
            # Unsqueeze y_time from (batch,) to (batch, 1) for MSELoss
            if y_time.dim() == 1:
                y_time = y_time.unsqueeze(1)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            activity_logits, remaining_time = model(x)
            
            # Compute losses
            activity_loss = activity_criterion(activity_logits, y_activity)
            time_loss = time_criterion(remaining_time, y_time)
            
            # Combined loss
            loss = activity_loss + time_loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss for epoch
        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    return epoch_losses