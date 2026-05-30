import pm4py
import pandas as pd

def load_event_log(filepath: str) -> pd.DataFrame:
    """
    Loads an XES event log file using pm4py, converts it to a pandas DataFrame,
    applies a robust filter for 'COMPLETE' lifecycle transitions, and sorts the result.
    
    Parameters:
        filepath (str): Path to the .xes file.
        
    Returns:
        pd.DataFrame: A DataFrame containing 'case:concept:name', 'concept:name', 
                      'time:timestamp', sorted by case ID and timestamp.
    """
    # Load the XES log
    log = pm4py.read_xes(filepath)
    
    # Convert to pandas DataFrame
    df = pm4py.convert_to_dataframe(log)
    
    if 'lifecycle:transition' in df.columns:
        # Step 1: Try exact match first
        filtered_df = df[df['lifecycle:transition'] == 'COMPLETE']
        
        if filtered_df.empty:
            # Step 2: If empty, try case-insensitive match
            # Ensure the column is treated as string for .str accessor
            filtered_df = df[df['lifecycle:transition'].astype(str).str.upper() == 'COMPLETE']
            
            if filtered_df.empty:
                # Step 3: If still empty, skip the filter entirely
                filtered_df = df
        else:
            # Exact match worked, use it
            filtered_df = filtered_df
            
        df = filtered_df
    
    # Ensure required columns exist
    required_cols = ['case:concept:name', 'concept:name', 'time:timestamp']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")
    
    # Sort by case ID and then by timestamp
    df = df.sort_values(by=['case:concept:name', 'time:timestamp']).reset_index(drop=True)
    
    return df