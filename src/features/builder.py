"""
Feature builder for the Adaptive Control System.
Transforms 1,300 rows/snapshot into a single row of aggregated features.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from src.features.buffer import WindowBuffer
from src.utils.config import get_config
from src.utils.types import FeatureVector

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Transforms 1,300 rows/snapshot into a single row of aggregated features.
    Relies on WindowBuffer for time-derivative calculations.
    """
    
    def __init__(self, buffer: WindowBuffer):
        self.buffer = buffer
        self._config = get_config()
    
    @property
    def feature_names(self) -> list:
        """Return the list of computed feature names (excluding metadata)."""
        return [
            'net_gex', 'net_vanna', 'pc_oi_ratio', 'wavg_call_iv',
            'max_call_oi_strike', 'max_put_oi_strike', 'delta_gex'
        ]
    
    def apply_deterministic_sort(self, snapshot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts snapshot deterministically to ensure feature stability.
        """
        return snapshot_df.sort_values(by='strike').reset_index(drop=True)
    
    def _get_atm_strike(self, snapshot_df: pd.DataFrame) -> float:
        """
        Approximates ATM strike by finding the strike where call delta is closest to 0.5.
        """
        idx = (snapshot_df['call_delta'] - 0.5).abs().idxmin()
        return snapshot_df.loc[idx, 'strike']
    
    def build_features(self, snapshot_df: pd.DataFrame, add_to_buffer: bool = True) -> FeatureVector:
        """
        Computes the complete feature vector for a given snapshot.
        
        Args:
            snapshot_df: Raw snapshot DataFrame
            add_to_buffer: If True, adds snapshot to buffer for time-derivatives.
                          Set to False if buffer was already updated externally.
                          
        Returns:
            FeatureVector containing computed features and metadata
        """
        # Ensure we always work with deterministically ordered data
        df = self.apply_deterministic_sort(snapshot_df)
        
        # Optionally add to buffer (decoupled to avoid side effects)
        if add_to_buffer:
            self.buffer.add(df)
        
        timestamp = df['timestamp'].iloc[0]
        atm_strike = self._get_atm_strike(df)
        
        features = {}
        
        # --- 1. Aggregate cross-strike features ---
        
        # Spot Proxy (simplification: ATM strike for scaffolding)
        spot_price = atm_strike 
        
        # Net GEX (simplified: Gamma * OI * 100 * Spot^2)
        call_gex = (df['call_gamma'] * df['call_oi'] * 100 * (spot_price ** 2)).sum()
        put_gex  = (df['put_gamma']  * df['put_oi']  * 100 * (spot_price ** 2)).sum()
        features['net_gex'] = call_gex - put_gex
        
        # Net Vanna Exposure (Vanna * OI * 100)
        call_vanna_exp = (df['call_vanna'] * df['call_oi'] * 100).sum()
        put_vanna_exp  = (df['put_vanna']  * df['put_oi']  * 100).sum()
        features['net_vanna'] = call_vanna_exp - put_vanna_exp
        
        # Put/Call OI Ratio - with proper zero handling
        total_call_oi = df['call_oi'].sum()
        total_put_oi  = df['put_oi'].sum()
        
        if total_call_oi > 0:
            features['pc_oi_ratio'] = total_put_oi / total_call_oi
        else:
            logger.warning("Total call OI is zero, setting pc_oi_ratio to 0")
            features['pc_oi_ratio'] = 0.0
        
        # Weighted Avg IV (Calls) - with proper zero handling
        if total_call_oi > 0:
            features['wavg_call_iv'] = (df['call_iv'] * df['call_oi']).sum() / total_call_oi
        else:
            logger.warning("Total call OI is zero for IV calculation, setting to 0")
            features['wavg_call_iv'] = 0.0
        
        # Max OI Strikes
        features['max_call_oi_strike'] = df.loc[df['call_oi'].idxmax(), 'strike']
        features['max_put_oi_strike']  = df.loc[df['put_oi'].idxmax(), 'strike']
        
        # --- 2. Time derivatives (Requires WindowBuffer) ---
        oldest_df = self.buffer.get_oldest()
        if oldest_df is not None and not oldest_df.equals(df):
            # Calculate Delta GEX
            old_oldest_df = self.apply_deterministic_sort(oldest_df)
            old_spot = self._get_atm_strike(old_oldest_df)
            
            old_call_gex = (old_oldest_df['call_gamma'] * old_oldest_df['call_oi'] * 100 * (old_spot ** 2)).sum()
            old_put_gex  = (old_oldest_df['put_gamma']  * old_oldest_df['put_oi']  * 100 * (old_spot ** 2)).sum()
            old_net_gex = old_call_gex - old_put_gex
            
            features['delta_gex'] = features['net_gex'] - old_net_gex
        else:
            features['delta_gex'] = 0.0  # First snapshot, no derivative yet
        
        return FeatureVector(
            features=features,
            timestamp=timestamp,
            atm_strike=atm_strike
        )
    
    def get_atm_strike(self, snapshot_df: pd.DataFrame) -> float:
        """
        Public method to get ATM strike (previously private _get_atm_strike).
        """
        df = self.apply_deterministic_sort(snapshot_df)
        return self._get_atm_strike(df)
