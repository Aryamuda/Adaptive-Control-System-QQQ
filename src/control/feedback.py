"""
Feedback Queue for the Adaptive Control System.
Time-indexed prediction queue that stores predictions at time t 
and matches them with realized outcomes at t+N.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

from src.utils.config import get_config
from src.utils.types import PredictionRecord, ResolvedError, PriceHistory

logger = logging.getLogger(__name__)


class FeedbackQueue:
    """
    Time-indexed Prediction Queue.
    Stores predictions at time t and matches them with realized outcomes at t+N.
    Integrates TTL/expiry logic to handle market closes or data gaps.
    """
    
    def __init__(self, target_minutes: int = None, ttl_minutes: int = None):
        """
        Args:
            target_minutes: The N minutes lookahead for our prediction. Uses config if None.
            ttl_minutes: Maximum time to wait for a match before discarding a prediction.
                        Uses config if None.
        """
        config = get_config()
        
        self.target_minutes = target_minutes if target_minutes is not None else config.pipeline.target_minutes
        self.ttl_minutes = ttl_minutes if ttl_minutes is not None else config.pipeline.ttl_minutes
        self.threshold = config.labeler.threshold
        
        # Each entry stores: {'timestamp': t, 'predicted_prob': p, 'features': {...}, 'attributions': {...}}
        self.queue: List[Dict[str, Any]] = []
        
        logger.info(
            f"FeedbackQueue initialized: target_minutes={self.target_minutes}, "
            f"ttl_minutes={self.ttl_minutes}, threshold={self.threshold}"
        )
    
    def add_prediction(
        self, 
        timestamp: pd.Timestamp, 
        prediction: float, 
        features: Dict[str, float], 
        attributions: Dict[str, float]
    ) -> None:
        """
        Stores a prediction waiting for future resolution.
        
        Args:
            timestamp: When the prediction was made
            prediction: Predicted probability
            features: Raw features used for prediction
            attributions: SHAP attributions for the prediction
        """
        self.queue.append({
            'timestamp': timestamp,
            'prediction': prediction,
            'features': features,
            'attributions': attributions
        })
        logger.debug(f"Added prediction at {timestamp}, queue size: {len(self.queue)}")
    
    def process_outcomes(
        self, 
        current_timestamp: pd.Timestamp, 
        current_price: float, 
        price_history: PriceHistory
    ) -> List[Dict[str, Any]]:
        """
        Given the current time and price, attempts to resolve pending predictions.
        Returns a list of resolved error records.
        Also discards expired predictions.
        
        Args:
            current_timestamp: Current time
            current_price: Current price (used for resolution if target price not in history)
            price_history: A pandas Series with DatetimeIndex of past prices, 
                          used to resolve the exact price at t+N.
                          
        Returns:
            List of resolved error records with 'prediction_time', 'resolution_time', 
            'error', and 'attributions'
        """
        resolved = []
        keep_queue = []
        
        for p in self.queue:
            target_time = p['timestamp'] + pd.Timedelta(minutes=self.target_minutes)
            expiry_time = target_time + pd.Timedelta(minutes=self.ttl_minutes)
            
            if target_time <= current_timestamp:
                # Prediction is mature enough to be resolved.
                # Try to find the exact price at target_time in the history.
                if target_time in price_history.index:
                    resolution_price = price_history[target_time]
                    
                    # Calculate log return from t to t+N
                    # We need the original spot price at time t. We stored atm_strike as a proxy in features.
                    original_price = p['features'].get('atm_strike')
                    
                    if original_price and resolution_price > 0:
                        log_return = np.log(resolution_price / original_price)
                        
                        # Classify: Up if return > threshold, Down if return < -threshold, else Flat
                        if log_return > self.threshold:
                            realized_label = 1
                        elif log_return < -self.threshold:
                            realized_label = -1
                        else:
                            realized_label = 0
                        
                        error = realized_label - p['prediction']
                        
                        resolved.append({
                            'prediction_time': p['timestamp'],
                            'resolution_time': target_time,
                            'error': error,
                            'attributions': p['attributions']
                        })
                        
                        logger.debug(
                            f"Resolved prediction from {p['timestamp']}: "
                            f"log_return={log_return:.6f}, realized={realized_label}, "
                            f"error={error:.4f}"
                        )
                        continue  # Successfully resolved, don't keep in queue
                    else:
                        logger.warning(
                            f"Missing original_price ({original_price}) or resolution_price ({resolution_price}) "
                            f"for prediction at {p['timestamp']}"
                        )
            
            # If we reach here, it's either not mature, or we couldn't find a price.
            # Check TTL
            if current_timestamp <= expiry_time:
                keep_queue.append(p)
            else:
                # Expired due to gap or market close. Discard.
                logger.warning(f"Prediction from {p['timestamp']} expired without resolution")
        
        self.queue = keep_queue
        
        if resolved:
            logger.info(f"Resolved {len(resolved)} predictions, {len(self.queue)} remaining in queue")
        
        return resolved
    
    def get_queue_size(self) -> int:
        """Return current queue size."""
        return len(self.queue)
    
    def clear(self) -> None:
        """Clear all pending predictions (e.g., at market open)."""
        self.queue.clear()
        logger.info("Feedback queue cleared")
