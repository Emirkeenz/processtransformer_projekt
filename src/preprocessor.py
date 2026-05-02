import pandas as pd
import numpy as np
import torch
from typing import List, Dict

def create_prefixes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates all possible prefixes for each case in an event log DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with columns 'case:concept:name', 
                           'concept:name', and 'time:timestamp'. 
                           Must be sorted by case ID and timestamp.
                           
    Returns:
        pd.DataFrame: A DataFrame containing one row per prefix with columns:
                      'case_id', 'prefix_activities', 'prefix_length', 
                      'next_activity', 'remaining_time'.
    """
    # Ensure the input is sorted correctly to maintain sequence integrity
    if not df.empty:
        df = df.sort_values(by=['case:concept:name', 'time:timestamp']).reset_index(drop=True)
    
    results = []
    
    # Group by case ID
    grouped = df.groupby('case:concept:name')
    
    for case_id, group in grouped:
        # Convert group to list of activities and timestamps for easier indexing
        activities = group['concept:name'].tolist()
        timestamps = group['time:timestamp'].tolist()
        
        n_events = len(activities)
        
        # Exclude cases with only one event as per requirements
        if n_events <= 1:
            continue
            
        # Iterate through possible prefix lengths (k from 1 to n_events - 1)
        # A prefix of length k includes events at indices 0 to k-1
        for k in range(1, n_events):
            prefix_acts = activities[:k]
            prefix_len = k
            
            # Determine the next activity (the one immediately following the prefix)
            next_act = activities[k]
            
            # Calculate remaining time
            # The prefix ends at index k-1. The case ends at index n_events-1.
            # We need the time difference between the last event of the full case 
            # and the last event of the current prefix.
            
            prefix_last_ts = timestamps[k-1]
            case_last_ts = timestamps[-1]
            
            # Ensure timestamps are numeric or convert to seconds if they are datetime objects
            if isinstance(prefix_last_ts, pd.Timestamp) or isinstance(case_last_ts, pd.Timestamp):
                diff_seconds = (case_last_ts - prefix_last_ts).total_seconds()
            else:
                # Assuming raw float/int seconds if not Timestamp
                diff_seconds = float(case_last_ts) - float(prefix_last_ts)
            
            results.append({
                'case_id': case_id,
                'prefix_activities': prefix_acts,
                'prefix_length': prefix_len,
                'next_activity': next_act,
                'remaining_time': diff_seconds
            })
    
    if not results:
        return pd.DataFrame(columns=['case_id', 'prefix_activities', 'prefix_length', 'next_activity', 'remaining_time'])
    
    return pd.DataFrame(results)

def build_activity_vocab(df: pd.DataFrame) -> dict:
    """
    Builds a vocabulary dictionary mapping unique activity names to integer indices.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'prefix_activities' column with lists of strings.
        
    Returns:
        dict: A dictionary mapping each unique activity name to a unique integer index.
              Index 0 is reserved for padding and is not included in this mapping.
    """
    # Collect all unique activity names from the lists in the 'prefix_activities' column
    unique_activities = set()
    
    for activity_list in df['prefix_activities']:
        if isinstance(activity_list, list):
            unique_activities.update(activity_list)
    
    # Sort the activities to ensure consistent indexing across runs
    sorted_activities = sorted(list(unique_activities))
    
    # Create the mapping: start index from 1
    vocab = {activity: idx + 1 for idx, activity in enumerate(sorted_activities)}
    
    return vocab

def encode_and_pad(prefixes: List[List[str]], vocab: Dict[str, int], max_len: int) -> torch.Tensor:
    """
    Encodes a list of activity name lists into integer indices and pads them to a fixed length.
    
    Args:
        prefixes (List[List[str]]): A list where each element is a list of activity name strings.
        vocab (Dict[str, int]): A dictionary mapping activity names to integer indices (starting from 1).
        max_len (int): The maximum sequence length. Sequences shorter than this will be padded with 0s on the left.
        
    Returns:
        torch.Tensor: A 2D tensor of shape (len(prefixes), max_len) with dtype torch.long.
                      Padding index 0 is used for left-padding.
    """
    encoded_sequences = []
    
    for prefix in prefixes:
        # Convert string activities to their corresponding integer indices
        # If an activity is not in vocab, it defaults to 0 (though typically vocab should cover all inputs)
        indices = [vocab.get(activity, 0) for activity in prefix]
        
        # Calculate current length
        current_len = len(indices)
        
        if current_len > max_len:
            # Truncate if longer than max_len (taking the last max_len elements usually makes sense for sequences)
            # Alternatively, you might want to take the first max_len. 
            # Given "prefix" context, keeping the end or start depends on specific use case. 
            # Standard practice for variable length padding often truncates the tail or head.
            # Here we truncate the tail to keep the 'prefix' intact if possible, but strictly speaking
            # if it's a "prefix", maybe we keep the beginning? 
            # Let's assume standard truncation to fit max_len (keeping the start is safer for "prefixes").
            indices = indices[:max_len]
            current_len = max_len
            
        elif current_len < max_len:
            # Pad with 0s on the LEFT to reach max_len
            padding_needed = max_len - current_len
            indices = [0] * padding_needed + indices
        
        encoded_sequences.append(indices)
    
    # Convert list of lists to a torch tensor
    tensor = torch.tensor(encoded_sequences, dtype=torch.long)
    
    return tensor