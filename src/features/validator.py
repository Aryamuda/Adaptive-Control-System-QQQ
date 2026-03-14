import pandas as pd
import numpy as np

class SnapshotValidator:
    """
    Early schema consistency check.
    Rejects corrupted or incomplete snapshots before they propagate downstream.
    """
    
    def __init__(self, expected_strikes=120, expected_columns=None):
        self.expected_strikes = expected_strikes
        self.expected_columns = expected_columns or [
            'timestamp', 'strike', 'call_bid', 'call_ask', 'call_iv', 'call_delta', 'call_gamma', 'call_vega', 'call_theta', 'call_rho', 'call_vanna', 'call_charm', 'call_oi', 'call_volume',
            'put_bid', 'put_ask', 'put_iv', 'put_delta', 'put_gamma', 'put_vega', 'put_theta', 'put_rho', 'put_vanna', 'put_charm', 'put_oi', 'put_volume'
        ]

    def validate(self, snapshot_df: pd.DataFrame) -> bool:
        """
        Validates the snapshot integrity.
        Returns True if valid, False otherwise.
        """
        if snapshot_df.empty:
            return False
            
        # 1. Check for expected columns
        missing_cols = set(self.expected_columns) - set(snapshot_df.columns)
        if missing_cols:
            print(f"Validation failed: Missing columns {missing_cols}")
            return False
            
        # 2. Check for missing strikes (assuming 1 row per strike)
        num_strikes = len(snapshot_df['strike'].unique())
        if num_strikes < self.expected_strikes:
            # We allow slightly more strikes but strictly reject fewer strikes
            print(f"Validation failed: Expected at least {self.expected_strikes} strikes, got {num_strikes}")
            return False
            
        # 3. Check for severe null Greeks (especially near ATM)
        # Assuming we need complete Greek data for basic computation
        critical_cols = ['call_delta', 'call_gamma', 'put_delta', 'put_gamma']
        null_counts = snapshot_df[critical_cols].isnull().sum()
        if null_counts.sum() > (self.expected_strikes * 0.1): # Reject if >10% of critical Greeks are missing
             print(f"Validation failed: High number of missing critical Greeks. \n{null_counts}")
             return False

        # 4. Check for invalid IV
        if (snapshot_df['call_iv'] <= 0).sum() > (self.expected_strikes * 0.1):
             print(f"Validation failed: Call IV contains excessive zeros or negatives.")
             return False

        return True
