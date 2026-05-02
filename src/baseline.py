import pandas as pd
from typing import List, Any, Union

class MostFrequentClassBaseline:
    def __init__(self):
        self.most_frequent = None

    def fit(self, df: pd.DataFrame) -> 'MostFrequentClassBaseline':
        """
        Fits the model by finding the most frequently occurring activity 
        in the 'next_activity' column.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the 'next_activity' column.
            
        Returns:
            MostFrequentClassBaseline: The fitted instance.
        """
        if 'next_activity' not in df.columns:
            raise ValueError("DataFrame must contain a 'next_activity' column.")
        
        # Count values and get the most common one
        # mode() returns a Series; we take the first value if there are ties
        mode_result = df['next_activity'].mode()
        
        if len(mode_result) == 0:
            # Handle case where column is empty or all NaN
            self.most_frequent = None
        else:
            self.most_frequent = mode_result.iloc[0]
            
        return self

    def predict(self, df: pd.DataFrame) -> List[Any]:
        """
        Predicts the most frequent activity for every row in the input DataFrame.
        
        Parameters:
            df (pd.DataFrame): DataFrame to generate predictions for.
            
        Returns:
            List[Any]: A list of predictions, each being the stored most frequent activity.
                       The length of the list equals the number of rows in df.
        """
        if self.most_frequent is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
            
        return [self.most_frequent] * len(df)
    
class MeanRemainingTimeBaseline:
    def __init__(self):
        self.mean_time: Union[float, None] = None

    def fit(self, df: pd.DataFrame) -> 'MeanRemainingTimeBaseline':
        """
        Fits the model by computing the mean of the 'remaining_time' column.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing the 'remaining_time' column with float values.
            
        Returns:
            MeanRemainingTimeBaseline: The fitted instance.
        """
        if 'remaining_time' not in df.columns:
            raise ValueError("DataFrame must contain a 'remaining_time' column.")
        
        # Calculate mean, handling potential NaNs by default behavior of pandas mean()
        # If the column is empty or all NaN, this will return NaN, which is stored.
        self.mean_time = df['remaining_time'].mean()
        
        return self
    
    def predict(self, df: pd.DataFrame) -> List[float]:
        """
        Predicts the mean remaining time for every row in the input DataFrame.
        
        Parameters:
            df (pd.DataFrame): DataFrame to generate predictions for.
            
        Returns:
            List[float]: A list where every value equals self.mean_time. 
                         The length of the list equals the number of rows in df.
        """
        if self.mean_time is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
            
        return [self.mean_time] * len(df)