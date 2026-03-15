"""
Cybernetic Control Loop for the Adaptive Control System.
Applies dynamic feature scaling based on delayed errors processed by PID.
"""
from typing import Dict, Any, List
import pandas as pd
import logging

from src.control.pid import PIDController
from src.control.regime import RegimeClassifier
from src.utils.config import get_config
from src.utils.types import ResolvedError, RegimeState, WeightDict

logger = logging.getLogger(__name__)


class CyberneticLoop:
    """
    Central orchestrator for the cybernetic feedback cycle.
    Applies dynamic feature scaling based on delayed errors processed by PID.
    """
    
    def __init__(self, feature_names: List[str], alpha: float = None):
        """
        Args:
            feature_names: List of all features to track weights for.
            alpha: Learning rate for weight decay. If None, uses config value.
        """
        config = get_config()
        
        self.alpha = alpha if alpha is not None else config.control.alpha
        self.min_weight = config.control.min_weight
        self.max_weight = config.control.max_weight
        
        # Initialize all feature weights to 1.0
        self.feature_weights: WeightDict = {f: 1.0 for f in feature_names}
        
        # We need a central PID to calculate the error magnitude correction
        self.pid = PIDController(
            kp=config.control.kp,
            ki=config.control.ki,
            kd=config.control.kd
        )
        self.regime_classifier = RegimeClassifier()
        
        logger.info(
            f"CyberneticLoop initialized with {len(feature_names)} features, "
            f"alpha={self.alpha}, PID(kp={config.control.kp}, ki={config.control.ki}, kd={config.control.kd})"
        )
    
    def adjust_features(self, raw_features: Dict[str, float]) -> Dict[str, float]:
        """
        Applies current cybernetic weights to the raw features before inference.
        
        Args:
            raw_features: Dictionary of raw feature values
            
        Returns:
            Dictionary of adjusted feature values
        """
        adjusted = {}
        for f_name, value in raw_features.items():
            if f_name in self.feature_weights:
                adjusted[f_name] = value * self.feature_weights[f_name]
            else:
                adjusted[f_name] = value  # e.g. timestamp does not get scaled
                logger.debug(f"Feature '{f_name}' not in weight tracking, using raw value")
        
        return adjusted
    
    def generate_confidence(self, prediction: float, regime: Dict[str, str]) -> float:
        """
        Calculates a confidence scalar [0, 1] based on current prediction and regime.
        In this scaffolding, it scales down confidence in high_vol regimes unless
        the raw prediction is extremely strong.
        
        Args:
            prediction: Raw model prediction (probability)
            regime: Dictionary with 'time_regime' and 'vol_regime'
            
        Returns:
            Confidence value between 0 and 1
        """
        # Base confidence is how far the prediction is from 0.5
        base_confidence = abs(prediction - 0.5) * 2.0
        
        # Regime override
        if regime.get('vol_regime') == 'high_vol':
            base_confidence *= 0.5  # Less confident in chaotic regimes
            logger.debug("High volatility regime - reducing confidence by 50%")
        
        return max(0.0, min(1.0, base_confidence))
    
    def process_feedback(self, resolved_errors: List[Dict[str, Any]]) -> None:
        """
        Takes retroactive matched errors and applies PID adjustments to feature weights.
        
        The weight adjustment logic:
        - error = realized_label - predicted_probability
        - contribution = SHAP_value * error
        - If contribution is positive (feature pushed in wrong direction), decrease weight
        - If contribution is negative (feature helped), increase weight
        - weight_multiplier = 1 + (alpha * PID_output * SHAP_value)
        
        Args:
            resolved_errors: List of resolved error records with 'error' and 'attributions'
        """
        if not resolved_errors:
            return
        
        for record in resolved_errors:
            error = record['error']
            attributions = record['attributions']
            
            # 1. Get PID adjustment magnitude
            # Note: error is (realized - predicted).
            pid_adjustment = self.pid.update(error)
            
            logger.debug(
                f"Processing feedback: error={error:.4f}, pid_adjustment={pid_adjustment:.4f}"
            )
            
            # 2. Distribute error correction across features based on their responsibility (SHAP)
            for f_name, shap_val in attributions.items():
                if f_name not in self.feature_weights:
                    logger.warning(
                        f"Feature '{f_name}' in attributions but not in weight tracking"
                    )
                    continue
                
                # Did this feature contribute to the error?
                # contribution = shap_val * error
                # - Positive contribution: feature pushed model in wrong direction
                # - Negative contribution: feature helped the prediction
                contribution = shap_val * error
                
                # Weight adjustment: 
                # If contribution is negative (feature helped), we want to increase weight
                # If contribution is positive (feature hurt), we want to decrease weight
                # The formula: weight_multiplier = 1 + (alpha * pid_adjustment * shap_val)
                # - When error is negative (under-predicted) and shap is positive (pushed up),
                #   contribution is negative -> pid is negative -> multiplier < 1 (decay)
                weight_multiplier = 1.0 + (self.alpha * pid_adjustment * shap_val)
                
                old_weight = self.feature_weights[f_name]
                new_weight = old_weight * weight_multiplier
                
                # Apply bounds
                self.feature_weights[f_name] = max(
                    self.min_weight, 
                    min(self.max_weight, new_weight)
                )
                
                if abs(new_weight - old_weight) > 0.01:
                    logger.debug(
                        f"Feature '{f_name}': weight {old_weight:.4f} -> {self.feature_weights[f_name]:.4f} "
                        f"(contribution={contribution:.4f}, multiplier={weight_multiplier:.4f})"
                    )
    
    def get_weights(self) -> WeightDict:
        """Return current feature weights."""
        return self.feature_weights.copy()
    
    def reset_weights(self) -> None:
        """Reset all weights to 1.0 (e.g., at market open)."""
        for f_name in self.feature_weights:
            self.feature_weights[f_name] = 1.0
        self.pid.reset()
        logger.info("Feature weights reset to 1.0")
