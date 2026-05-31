import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


def train_lstm(model, train_loader, num_epochs=20, lr=0.002, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion_activity = nn.CrossEntropyLoss()
    criterion_time = nn.MSELoss()

    losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for batch_x, batch_y_activities, batch_y_times in train_loader:
            batch_x = batch_x.to(device)
            batch_y_activities = batch_y_activities.to(device)
            batch_y_times = batch_y_times.to(device)

            optimizer.zero_grad()

            activity_logits, time_pred = model(batch_x)

            loss_activity = criterion_activity(activity_logits, batch_y_activities)
            loss_time = criterion_time(time_pred, batch_y_times)

            total_loss_batch = loss_activity + loss_time
            total_loss_batch.backward()

            optimizer.step()

            total_loss += total_loss_batch.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1}/{num_epochs} — Loss: {avg_loss:.4f}")
        losses.append(avg_loss)

    return losses


def predict_lstm(model, data_loader, device, time_mean, time_std):
    model.eval()
    all_activities = []
    all_times = []

    with torch.no_grad():
        for batch_x, _, _ in data_loader:
            batch_x = batch_x.to(device)

            activity_logits, time_pred = model(batch_x)

            pred_activities = torch.argmax(activity_logits, dim=1).cpu().numpy()
            all_activities.append(pred_activities)

            # Denormalize time predictions: pred * std + mean
            denormalized_times = (time_pred.squeeze(-1).cpu().numpy() * time_std) + time_mean
            all_times.append(denormalized_times)

    if not all_activities:
        return np.array([]), np.array([])

    pred_activities_numpy = np.concatenate(all_activities, axis=0)
    pred_times_numpy = np.concatenate(all_times, axis=0)

    return pred_activities_numpy, pred_times_numpy