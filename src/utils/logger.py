"""
Structured logger for the Adaptive Control System.
Logs confidence scalars, regime tags, and feature weights for offline evaluation
and later execution engine integration.
"""
import json
import logging
import sys
from typing import Dict, Any, Optional
from pathlib import Path

from src.utils.config import get_config
from src.utils.types import SignalRecord


class SignalLogger:
    """
    Structured logger for the Cybernetic Loop outputs.
    Logs confidence scalars, regime tags, and feature weights for offline evaluation
    and later execution engine integration.
    """
    
    def __init__(self, log_file: Optional[str] = None, level: Optional[str] = None):
        """
        Initialize the logger.
        
        Args:
            log_file: Path to log file. Uses config if None.
            level: Logging level. Uses config if None.
        """
        config = get_config()
        
        self.log_file = log_file or config.logging.log_file
        self.level = getattr(logging, level or config.logging.level)
        
        # Set up root logger
        self.logger = logging.getLogger("Cybernetics")
        self.logger.setLevel(self.level)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        try:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.level)
            # Just raw JSON lines for file
            file_formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        except (IOError, OSError) as e:
            self.logger.error(f"Could not create log file {self.log_file}: {e}")
    
    def log_signal(
        self, 
        timestamp: str, 
        raw_prediction: float, 
        confidence: float, 
        regime: Dict[str, str], 
        current_weights: Dict[str, float]
    ) -> None:
        """
        Logs a single inference event's complete state.
        
        Args:
            timestamp: ISO format timestamp
            raw_prediction: Raw model prediction
            confidence: Confidence scalar
            regime: Regime state dictionary
            current_weights: Current feature weights
        """
        record = SignalRecord(
            timestamp=timestamp,
            raw_prediction=raw_prediction,
            confidence=confidence,
            regime=regime,
            feature_weights=current_weights
        )
        
        # Log as JSON to file
        self.logger.info(json.dumps(record.to_dict()))
        
        # Also log structured info for debugging
        self.logger.debug(
            f"Signal: pred={raw_prediction:.3f}, conf={confidence:.3f}, "
            f"regime={regime['time_regime']}/{regime['vol_regime']}"
        )
    
    def log_event(self, level: str, message: str, **kwargs: Any) -> None:
        """
        Log a generic event with additional context.
        
        Args:
            level: Log level (debug, info, warning, error)
            message: Log message
            **kwargs: Additional context to include
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        
        if kwargs:
            extra = " ".join([f"{k}={v}" for k, v in kwargs.items()])
            log_method(f"{message} | {extra}")
        else:
            log_method(message)
    
    def log_prediction_added(self, timestamp: str, queue_size: int) -> None:
        """Log that a prediction was added to the queue."""
        self.logger.debug(f"Prediction added at {timestamp}, queue size: {queue_size}")
    
    def log_prediction_resolved(
        self, 
        prediction_time: str, 
        error: float, 
        resolved_count: int,
        remaining: int
    ) -> None:
        """Log that predictions were resolved."""
        self.logger.info(
            f"Resolved {resolved_count} predictions (error={error:.4f}), "
            f"{remaining} remaining in queue"
        )
    
    def log_prediction_expired(self, prediction_time: str) -> None:
        """Log that a prediction expired."""
        self.logger.warning(f"Prediction from {prediction_time} expired without resolution")
    
    def log_validation_failure(self, reason: str, details: str = "") -> None:
        """Log validation failure."""
        self.logger.warning(f"Validation failed: {reason} {details}")
    
    def log_weight_update(
        self, 
        feature: str, 
        old_weight: float, 
        new_weight: float
    ) -> None:
        """Log feature weight update."""
        self.logger.debug(
            f"Weight update: {feature} {old_weight:.4f} -> {new_weight:.4f}"
        )


# Module-level convenience function
_default_logger: Optional[SignalLogger] = None


def get_logger() -> SignalLogger:
    """Get the default logger instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = SignalLogger()
    return _default_logger
