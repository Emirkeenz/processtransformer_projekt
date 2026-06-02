import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def train_retain(model, train_loader, num_epochs, lr, device):
    """
    Train the RETAINModel.

    Args:
        model: RETAINModel instance
        train_loader: DataLoader yielding (x, y_activity, y_time) batches
        num_epochs: Number of training epochs
        lr: Learning rate for Adam optimizer
        device: Device to run training on ('cpu' or 'cuda')

    Returns:
        list of average losses per epoch
    """
    model.to(device)
    criterion_activity = nn.CrossEntropyLoss()
    criterion_time = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

            # Forward pass
            activity_logits, time_pred = model(x)

            # Calculate losses
            loss_activity = criterion_activity(activity_logits, y_activity)
            loss_time = criterion_time(time_pred, y_time)

            # Combined loss (equal weighting)
            loss = loss_activity + loss_time

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    return epoch_losses


def predict_retain(model, data_loader, device, time_mean, time_std):
    """
    Run inference with the trained RETAINModel.

    Args:
        model: Trained RETAINModel instance
        data_loader: DataLoader yielding (x, y_activity, y_time) batches
        device: Device to run inference on
        time_mean: Mean value used for normalizing time targets during training
        time_std: Standard deviation used for normalizing time targets during training

    Returns:
        predicted_activities: numpy array of predicted activity indices (shape: total_samples,)
        predicted_times: numpy array of denormalized time predictions in seconds (shape: total_samples,)
    """
    model.to(device)
    model.eval()

    all_activities = []
    all_times = []

    with torch.no_grad():
        for x, _, _ in data_loader:  # We only need x for prediction
            x = x.to(device)

            # Forward pass
            activity_logits, time_pred = model(x)

            # Get predicted activity indices via argmax
            pred_activities = torch.argmax(activity_logits, dim=1)

            # Denormalize time predictions: pred * std + mean
            pred_times = time_pred.squeeze(1) * time_std + time_mean

            # Convert to numpy
            all_activities.extend(pred_activities.cpu().numpy())
            all_times.extend(pred_times.cpu().numpy())

    return np.array(all_activities), np.array(all_times)