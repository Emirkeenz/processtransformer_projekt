import pandas as pd
from typing import List, Any

def accuracy(y_true: list, y_pred: list) -> float:
    """
    Computes classification accuracy between two lists.
    
    Parameters:
        y_true (list): List of true labels.
        y_pred (list): List of predicted labels.
        
    Returns:
        float: Accuracy score (number of correct predictions / total predictions).
               Returns 0.0 if lists are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    if not y_true:
        return 0.0
    
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def mae(y_true: list, y_pred: list) -> float:
    """
    Computes Mean Absolute Error between two lists of floats.
    
    Parameters:
        y_true (list): List of true float values.
        y_pred (list): List of predicted float values.
        
    Returns:
        float: Mean Absolute Error.
               Returns 0.0 if lists are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    if not y_true:
        return 0.0
    
    try:
        errors = [abs(float(t) - float(p)) for t, p in zip(y_true, y_pred)]
        return sum(errors) / len(errors)
    except (TypeError, ValueError):
        raise ValueError("y_true and y_pred must contain numeric values convertible to float.")


def evaluate_by_prefix_length(df: pd.DataFrame, y_pred_activity: list, y_pred_time: list) -> pd.DataFrame:
    """
    Evaluates activity prediction accuracy and remaining time MAE grouped by 'prefix_length'.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing at least 'prefix_length', 'next_activity', 
                           and 'remaining_time' columns.
        y_pred_activity (list): List of predicted next activities corresponding to rows in df.
        y_pred_time (list): List of predicted remaining times corresponding to rows in df.
        
    Returns:
        pd.DataFrame: A DataFrame with columns 'prefix_length', 'accuracy', and 'mae'.
    """
    if len(df) != len(y_pred_activity) or len(df) != len(y_pred_time):
        raise ValueError("Input lists must have the same length as the number of rows in df.")
    
    # Add predictions to the dataframe for easier grouping
    eval_df = df.copy()
    eval_df['pred_activity'] = y_pred_activity
    eval_df['pred_time'] = y_pred_time
    
    results = []
    
    # Group by prefix_length
    grouped = eval_df.groupby('prefix_length')
    
    for prefix_len, group in grouped:
        # Extract ground truth and predictions for this group
        true_acts = group['next_activity'].tolist()
        pred_acts = group['pred_activity'].tolist()
        true_times = group['remaining_time'].tolist()
        pred_times = group['pred_time'].tolist()
        
        # Calculate metrics using the helper functions
        acc_score = accuracy(true_acts, pred_acts)
        mae_score = mae(true_times, pred_times)
        
        results.append({
            'prefix_length': prefix_len,
            'accuracy': acc_score,
            'mae': mae_score
        })
    
    return pd.DataFrame(results).sort_values(by='prefix_length').reset_index(drop=True)