import torch
import torch.nn as nn
import torch.optim as optim

def train_epoch(model, dataloader, criterion_activity, criterion_time, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        activity_indices, activity_labels, time_labels = batch
        
        activity_indices = activity_indices.to(device)
        activity_labels = activity_labels.to(device)
        time_labels = time_labels.to(device)
        
        activity_logits, time_prediction = model(activity_indices)
        
        activity_loss = criterion_activity(activity_logits, activity_labels)
        time_loss = criterion_time(time_prediction.squeeze(), time_labels)
        
        loss = activity_loss + time_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, dataloader, criterion_activity, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            activity_indices, activity_labels, time_labels = batch
            
            activity_indices = activity_indices.to(device)
            activity_labels = activity_labels.to(device)
            
            activity_logits, _ = model(activity_indices)
            
            _, predicted = torch.max(activity_logits, dim=1)
            
            total += activity_labels.size(0)
            correct += (predicted == activity_labels).sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy