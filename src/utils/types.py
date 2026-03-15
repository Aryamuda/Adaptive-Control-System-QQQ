"""
Type definitions and dataclasses for the Adaptive Control System.
Provides type-safe DTOs for cross-module communication.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np


@dataclass
class PredictionRecord:
    """
    Represents a stored prediction waiting for future resolution.
    """
    timestamp: pd.Timestamp
    prediction: float
    features: Dict[str, float]
    attributions: Dict[str, float]
    
    def __post_init__(self):
        if not isinstance(self.timestamp, pd.Timestamp):
            self.timestamp = pd.to_datetime(self.timestamp)


@dataclass
class ResolvedError:
    """
    Represents a resolved prediction with its error and attribution.
    """
    prediction_time: pd.Timestamp
    resolution_time: pd.Timestamp
    error: float
    attributions: Dict[str, float]
    
    def __post_init__(self):
        if not isinstance(self.prediction_time, pd.Timestamp):
            self.prediction_time = pd.to_datetime(self.prediction_time)
        if not isinstance(self.resolution_time, pd.Timestamp):
            self.resolution_time = pd.to_datetime(self.resolution_time)


@dataclass
class RegimeState:
    """
    Represents the current market regime.
    """
    time_regime: str  # 'open', 'mid', 'close'
    vol_regime: str   # 'normal', 'high_vol'
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'time_regime': self.time_regime,
            'vol_regime': self.vol_regime
        }


@dataclass
class FeatureVector:
    """
    Represents a computed feature vector with metadata.
    """
    features: Dict[str, float]
    timestamp: Optional[pd.Timestamp] = None
    atm_strike: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        result = self.features.copy()
        if self.timestamp is not None:
            result['timestamp'] = self.timestamp
        if self.atm_strike is not None:
            result['atm_strike'] = self.atm_strike
        return result
    
    def get_feature(self, name: str, default: float = 0.0) -> float:
        """Get a feature value with a default."""
        return self.features.get(name, default)


@dataclass
class SignalRecord:
    """
    Represents a complete signal output for logging.
    """
    timestamp: str
    raw_prediction: float
    confidence: float
    regime: Dict[str, str]
    feature_weights: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "raw_prediction": self.raw_prediction,
            "confidence_scalar": self.confidence,
            "regime": self.regime,
            "feature_weights": self.feature_weights
        }
    
    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict())


@dataclass
class ValidationResult:
    """
    Result of snapshot validation.
    """
    is_valid: bool
    missing_columns: List[str] = field(default_factory=list)
    null_greek_count: int = 0
    zero_iv_count: int = 0
    strike_count: int = 0
    error_message: Optional[str] = None
    
    def __bool__(self) -> bool:
        return self.is_valid


@dataclass
class PIDState:
    """
    Internal state of the PID controller.
    """
    integral_error: float = 0.0
    prev_error: float = 0.0
    last_output: float = 0.0
    
    def reset(self) -> None:
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.last_output = 0.0


# Type aliases for clarity
FeatureDict = Dict[str, float]
AttributionDict = Dict[str, float]
WeightDict = Dict[str, float]
Timestamp = pd.Timestamp
PriceHistory = pd.Series  # DatetimeIndex -> float prices