"""
Target labeler for the Adaptive Control System.
Offline component for target variable computation.
Used strictly in training/backtesting, NEVER in the live runtime pipeline.
"""
import pandas as pd
import numpy as np
from typing import Optional
import logging

from src.utils.config import get_config

logger = logging.getLogger(__name__)


class TargetLabeler:
    """
    Offline component for target variable computation.
    Used strictly in training/backtesting, NEVER in the live runtime pipeline.
    """
    
    def __init__(self, target_minutes: int = None, threshold: float = None):
        """
        Args:
            target_minutes: N minutes lookahead for return calculation. Uses config if None.
            threshold: Minimum log return to classify as Up (1) or Down (-1).
                      Between is Flat (0). Uses config if None.
        """
        config = get_config()
        
        self.target_minutes = target_minutes if target_minutes is not None else config.labeler.target_minutes
        self.threshold = threshold if threshold is not None else config.labeler.threshold
        
        logger.info(
            f"TargetLabeler initialized: target_minutes={self.target_minutes}, "
            f"threshold={self.threshold}"
        )
    
    def compute_labels(
        self, 
        price_series: pd.Series, 
        timestamps: pd.Series
    ) -> pd.Series:
        """
        Computes forward-looking N-minute log returns and binarizes them.
        
        Args:
            price_series: Series of spot prices (or ATM strikes as proxy)
            timestamps: Series of datetime objects for each price
            
        Returns:
            pd.Series of labels: 1 (Up), 0 (Flat), -1 (Down), np.nan (unavailable)
        """
        df = pd.DataFrame({
            'price': price_series, 
            'timestamp': pd.to_datetime(timestamps)
        })
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        
        # Calculate exactly based on time difference, not row index shift
        target_times = df['timestamp'] + pd.Timedelta(minutes=self.target_minutes)
        
        # Create a lookup for future prices
        price_lookup = df.set_index('timestamp')['price']
        
        # Map target times to future prices
        future_prices = target_times.map(price_lookup)
        
        df['future_price'] = future_prices
        
        # Log return
        with np.errstate(divide='ignore', invalid='ignore'):
            df['log_return'] = np.log(df['future_price'] / df['price'])
        
        # Binarize/Trinarize with proper NaN handling
        conditions = [
            df['log_return'] > self.threshold,
            df['log_return'] < -self.threshold
        ]
        choices = [1, -1]  # 1=Up, -1=Down, 0=Flat (Dead-band)
        
        df['label'] = np.select(conditions, choices, default=0)
        
        # NaN out rows where future_price wasn't available (end of day or data gaps)
        df.loc[df['future_price'].isna(), 'label'] = np.nan
        df.loc[df['log_return'].isna(), 'label'] = np.nan
        
        # Log statistics
        valid_labels = df['label'].dropna()
        if len(valid_labels) > 0:
            up_pct = (valid_labels == 1).mean() * 100
            flat_pct = (valid_labels == 0).mean() * 100
            down_pct = (valid_labels == -1).mean() * 100
            nan_pct = df['label'].isna().mean() * 100
            
            logger.info(
                f"Label distribution: Up={up_pct:.1f}%, Flat={flat_pct:.1f}%, "
                f"Down={down_pct:.1f}%, NaN={nan_pct:.1f}%"
            )
        
        return df['label']
    
    def compute_continuous_labels(
        self, 
        price_series: pd.Series, 
        timestamps: pd.Series
    ) -> pd.Series:
        """
        Computes forward-looking N-minute log returns as continuous values.
        Useful for regression models.
        
        Args:
            price_series: Series of spot prices
            timestamps: Series of datetime objects
            
        Returns:
            pd.Series of log returns (can be NaN for unavailable future prices)
        """
        df = pd.DataFrame({
            'price': price_series, 
            'timestamp': pd.to_datetime(timestamps)
        })
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        
        target_times = df['timestamp'] + pd.Timedelta(minutes=self.target_minutes)
        price_lookup = df.set_index('timestamp')['price']
        future_prices = target_times.map(price_lookup)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_returns = np.log(future_prices / df['price'])
        
        return log_returns
