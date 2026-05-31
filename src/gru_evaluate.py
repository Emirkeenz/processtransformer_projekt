import torch
import torch.nn as nn
from typing import Tuple, Optional

def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[str] = None
) -> Tuple[float, float]:
    """
    Evaluates a trained SequenceActivityPredictor model on test data.
    
    Args:
        model (nn.Module): The trained SequenceActivityPredictor instance.
        dataloader (DataLoader): DataLoader yielding batches with:
            - activity_indices: (seq_len, batch_size), LongTensor
            - time_features: (seq_len, batch_size, num_time_features), FloatTensor
            - activity_labels: (seq_len, batch_size), LongTensor
            - time_labels: (seq_len, batch_size), FloatTensor
        device (str): Device to run evaluation on ('cpu' or 'cuda'). Defaults to auto-detect.
        
    Returns:
        Tuple[float, float]: 
            - accuracy: Percentage of correctly predicted next activities (0.0 to 100.0).
            - mae: Mean Absolute Error for remaining time prediction.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure model is in eval mode and on correct device
    model.eval()
    model = model.to(device)
    
    total_correct = 0
    total_samples = 0
    total_abs_error = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch (handling both dict and tuple formats)
            if isinstance(batch, dict):
                activity_indices = batch['activity_indices']
                time_features = batch['time_features']
                activity_labels = batch['activity_labels']
                time_labels = batch['time_labels']
            else:
                activity_indices, time_features, activity_labels, time_labels = batch
            
            # Move to device
            activity_indices = activity_indices.to(device)
            time_features = time_features.to(device)
            activity_labels = activity_labels.to(device)
            time_labels = time_labels.to(device)
            
            # Forward pass
            logits_activity, time_preds = model(activity_indices, time_features)
            
            # --- Activity Accuracy Calculation ---
            # logits_activity shape: (seq_len, batch_size, vocab_size)
            # Get the index of the highest logit (prediction)
            # argmax over the last dimension (vocab_size)
            predictions = torch.argmax(logits_activity, dim=-1)  # Shape: (seq_len, batch_size)
            
            # Flatten to compare element-wise
            pred_flat = predictions.view(-1)          # (total_elements,)
            label_flat = activity_labels.view(-1)     # (total_elements,)
            
            # Count correct predictions
            correct = (pred_flat == label_flat).sum().item()
            total_correct += correct
            total_samples += label_flat.numel()
            
            # --- Time MAE Calculation ---
            # time_preds shape: (seq_len, batch_size, 1)
            # time_labels shape: (seq_len, batch_size)
            
            # Squeeze time_preds to match labels shape (remove last dim of size 1)
            pred_time = time_preds.squeeze(-1)        # Shape: (seq_len, batch_size)
            
            # Calculate absolute error per element
            abs_errors = torch.abs(pred_time - time_labels)
            total_abs_error += abs_errors.sum().item()
            
    # Calculate final metrics
    accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    mae = (total_abs_error / total_samples) if total_samples > 0 else 0.0
    
    return accuracy, mae