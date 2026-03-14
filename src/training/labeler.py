import pandas as pd
import numpy as np

class TargetLabeler:
    """
    Offline component for target variable computation.
    Used strictly in training/backtesting, NEVER in the live runtime pipeline.
    """
    def __init__(self, target_minutes: int = 5, threshold: float = 0.0005):
        """
        Args:
            target_minutes: N minutes lookahead for return calculation
            threshold: Minimum log return to classify as Up (1) or Down (-1). 
                       Between is Flat (0).
        """
        self.target_minutes = target_minutes
        self.threshold = threshold

    def compute_labels(self, price_series: pd.Series, timestamps: pd.Series) -> pd.Series:
        """
        Computes forward-looking N-minute log returns and binarizes them.
        
        Args:
            price_series: Series of spot prices (or ATM strikes as proxy)
            timestamps: Series of datetime objects for each price
            
        Returns:
            pd.Series of labels: 1 (Up), 0 (Flat), -1 (Down)
        """
        df = pd.DataFrame({'price': price_series, 'timestamp': pd.to_datetime(timestamps)})
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        
        # Calculate exactly based on time difference, not row index shift
        target_times = df['timestamp'] + pd.Timedelta(minutes=self.target_minutes)
        future_prices = df.merge(
            df[['timestamp', 'price']], 
            left_on=target_times, 
            right_on='timestamp', 
            how='left',
            suffixes=('', '_future')
        )['price_future']
        
        df['future_price'] = future_prices
        
        # Log return
        df['log_return'] = np.log(df['future_price'] / df['price'])
        
        # Binarize/Trinarize
        conditions = [
            df['log_return'] > self.threshold,
            df['log_return'] < -self.threshold
        ]
        choices = [1, -1] # 1=Up, -1=Down, 0=Flat (Dead-band)
        
        df['label'] = np.select(conditions, choices, default=0)
        
        # NaN out rows where future_price wasn't available (end of day)
        df.loc[df['future_price'].isna(), 'label'] = np.nan
        
        return df['label']
