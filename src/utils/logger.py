import json
import logging
from typing import Dict, Any

class SignalLogger:
    """
    Structured logger for the Cybernetic Loop outputs.
    Logs confidence scalars, regime tags, and feature weights for offline evaluation
    and later execution engine integration.
    """
    
    def __init__(self, log_file: str = "signals.log"):
        self.logger = logging.getLogger("Cybernetics")
        self.logger.setLevel(logging.INFO)
        
        # Ensure we don't add multiple handlers if re-instantiated
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(message)s') # Just raw JSON lines
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
    def log_signal(self, timestamp: str, raw_prediction: float, confidence: float, 
                   regime: Dict[str, str], current_weights: Dict[str, float]) -> None:
        """
        Logs a single inference event's complete state.
        """
        record = {
            "timestamp": timestamp,
            "raw_prediction": raw_prediction,
            "confidence_scalar": confidence,
            "regime": regime,
            "feature_weights": current_weights
        }
        
        self.logger.info(json.dumps(record))
