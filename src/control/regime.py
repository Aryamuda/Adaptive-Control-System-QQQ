"""
Regime classifier for the Adaptive Control System.
Tags the current market state to allow the Cybernetic loop 
to apply regime-specific PID gains or override signals.
"""
import pandas as pd
from typing import Dict
import logging

from src.utils.config import get_config
from src.utils.types import RegimeState

logger = logging.getLogger(__name__)


class RegimeClassifier:
    """
    Runtime regime classifier.
    Tags the current market state to allow the Cybernetic loop 
    to apply regime-specific PID gains or override signals.
    """
    
    def __init__(self):
        config = get_config()
        self._open_time = pd.Timestamp(config.regime.open_time).time()
        self._close_time = pd.Timestamp(config.regime.close_time).time()
        self._high_vol_threshold = config.feedback.high_vol_threshold
        
        logger.debug(
            f"RegimeClassifier initialized: open={self._open_time}, "
            f"close={self._close_time}, high_vol_threshold={self._high_vol_threshold}"
        )
    
    def classify_time_regime(self, timestamp: pd.Timestamp) -> str:
        """
        Classifies based on time of day (US Eastern Time assumed for market hours).
        
        Args:
            timestamp: The timestamp to classify
            
        Returns:
            'open', 'mid', or 'close'
        """
        # Note: Timestamp should theoretically be localized to ET for real trading.
        # Scaffolding uses exact hour/minute assuming 09:30-16:00 ET.
        time = timestamp.time()
        
        if time < self._open_time:
            return 'pre_open'
        elif time < self._open_time:
            return 'open'
        elif time > self._close_time:
            return 'close'
        else:
            return 'mid'
    
    def classify_vol_regime(self, features: Dict[str, float]) -> str:
        """
        Classifies based on feature vector (e.g., ATM IV or VIX proxy).
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            'normal' or 'high_vol'
        """
        atm_iv = features.get('wavg_call_iv', 0.0)
        
        if atm_iv > self._high_vol_threshold:
            logger.debug(f"High volatility regime detected: IV={atm_iv:.2%}")
            return 'high_vol'
        return 'normal'
    
    def get_current_regime(self, timestamp: pd.Timestamp, features: Dict[str, float]) -> RegimeState:
        """
        Returns a composite tag of the current regime for logging and control logic.
        
        Args:
            timestamp: Current timestamp
            features: Current feature values
            
        Returns:
            RegimeState with time_regime and vol_regime
        """
        return RegimeState(
            time_regime=self.classify_time_regime(timestamp),
            vol_regime=self.classify_vol_regime(features)
        )
