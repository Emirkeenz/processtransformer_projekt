import pandas as pd
import numpy as np

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
