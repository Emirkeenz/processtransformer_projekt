import pm4py
import pandas as pd

def load_event_log(filepath: str) -> pd.DataFrame:
    """
    Loads an XES event log file using pm4py, converts it to a pandas DataFrame,
    filters for 'COMPLETE' lifecycle transitions if available, and sorts the result.
    
    Parameters:
        filepath (str): Path to the .xes file.
        
    Returns:
        pd.DataFrame: A DataFrame containing 'case:concept:name', 'concept:name', 
                      'time:timestamp', and optionally other columns, sorted by case ID and timestamp.
    """
    # Load the XES log
    log = pm4py.read_xes(filepath)
    
    # Convert to pandas DataFrame
    df = pm4py.convert_to_dataframe(log)
    
    # Filter for 'COMPLETE' lifecycle transition if the column exists
    if 'lifecycle:transition' in df.columns:
        df = df[df['lifecycle:transition'] == 'COMPLETE'].reset_index(drop=True)
    
    # Ensure required columns exist (optional safety check, though pm4py usually provides them)
    required_cols = ['case:concept:name', 'concept:name', 'time:timestamp']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame is missing required columns: {required_cols}")
    
    # Sort by case ID and then by timestamp
    df = df.sort_values(by=['case:concept:name', 'time:timestamp'])
    
    return df