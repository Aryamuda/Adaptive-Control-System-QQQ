import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

class FeedbackQueue:
    """
    Time-indexed Prediction Queue.
    Stores predictions at time t and matches them with realized outcomes at t+N.
    Integrates TTL/expiry logic to handle market closes or data gaps.
    """
    
    def __init__(self, target_minutes: int = 5, ttl_minutes: int = 15):
        """
        Args:
            target_minutes: The N minutes lookahead for our prediction.
            ttl_minutes: Maximum time to wait for a match before discarding a prediction.
        """
        self.target_minutes = target_minutes
        self.ttl_minutes = ttl_minutes
        
        # type: List[Dict[str, Any]]
        # Each entry stores: {'timestamp': t, 'predicted_prob': p, 'features': {...}, 'attributions': {...}}
        self.queue = []
        
    def add_prediction(self, timestamp: pd.Timestamp, prediction: float, 
                       features: Dict[str, float], attributions: Dict[str, float]) -> None:
        """
        Stores a prediction waiting for future resolution.
        """
        self.queue.append({
            'timestamp': timestamp,
            'prediction': prediction,
            'features': features,
            'attributions': attributions
        })
        
    def process_outcomes(self, current_timestamp: pd.Timestamp, current_price: float, price_history: pd.Series) -> List[Dict[str, Any]]:
        """
        Given the current time and price, attempts to resolve pending predictions.
        Returns a list of resolved error records.
        Also discards expired predictions.
        
        Args:
            price_history: A pandas Series with DatetimeIndex of past prices, 
                           used to resolve the exact price at t+N.
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
                    
                    if original_price:
                        log_return = np.log(resolution_price / original_price)
                        realized_label = 1 if log_return > 0.0005 else (-1 if log_return < -0.0005 else 0)
                        
                        error = realized_label - p['prediction'] # Simple error for scaffolding
                        
                        resolved.append({
                            'prediction_time': p['timestamp'],
                            'resolution_time': target_time,
                            'error': error,
                            'attributions': p['attributions']
                        })
                        continue # Successfully resolved, don't keep in queue
                
            # If we reach here, it's either not mature, or we couldn't find a price.
            # Check TTL
            if current_timestamp <= expiry_time:
                keep_queue.append(p)
            else:
                # Expired due to gap or market close. Discard.
                print(f"Prediction from {p['timestamp']} expired without resolution.")
                
        self.queue = keep_queue
        return resolved
