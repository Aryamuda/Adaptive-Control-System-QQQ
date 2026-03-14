from typing import Dict, Any, List
import pandas as pd
from src.control.pid import PIDController
from src.control.regime import RegimeClassifier

class CyberneticLoop:
    """
    Central orchestrator for the cybernetic feedback cycle.
    Applies dynamic feature scaling based on delayed errors processed by PID.
    """
    
    def __init__(self, feature_names: List[str], alpha: float = 0.05):
        """
        Args:
            feature_names: List of all features to track weights for.
            alpha: Learning rate for weight decay.
        """
        self.alpha = alpha
        
        # Initialize all feature weights to 1.0
        self.feature_weights = {f: 1.0 for f in feature_names}
        
        # We need a central PID to calculate the error magnitude correction
        self.pid = PIDController(kp=0.5, ki=0.1, kd=0.0)
        self.regime_classifier = RegimeClassifier()
        
    def adjust_features(self, raw_features: Dict[str, float]) -> Dict[str, float]:
        """
        Applies current cybernetic weights to the raw features before inference.
        """
        adjusted = {}
        for f_name, value in raw_features.items():
            if f_name in self.feature_weights:
                adjusted[f_name] = value * self.feature_weights[f_name]
            else:
                adjusted[f_name] = value # e.g. timestamp does not get scaled
                
        return adjusted
        
    def generate_confidence(self, prediction: float, regime: Dict[str, str]) -> float:
        """
        Calculates a confidence scalar [0, 1] based on current prediction and regime.
        In this scaffolding, it scales down confidence in high_vol regimes unless
        the raw prediction is extremely strong.
        """
        # Base confidence is how far the prediction is from 0.5
        base_confidence = abs(prediction - 0.5) * 2.0 
        
        # Regime override
        if regime['vol_regime'] == 'high_vol':
            base_confidence *= 0.5 # Less confident in chaotic regimes
            
        return max(0.0, min(1.0, base_confidence))

    def process_feedback(self, resolved_errors: List[Dict[str, Any]]) -> None:
        """
        Takes retroactive matched errors and applies PID adjustments to feature weights.
        """
        for record in resolved_errors:
            error = record['error']
            attributions = record['attributions']
            
            # 1. Get PID adjustment magnitude
            # Note: error is (realized - predicted).
            pid_adjustment = self.pid.update(error)
            
            # 2. Distribute error correction across features based on their responsibility (SHAP)
            # If error is positive (model underpredicted UP), and feature X pushed us DOWN (negative SHAP),
            # then feature X was wrong. We should decay its weight.
            # Simplified heuristic for scaffolding:
            
            for f_name, shap_val in attributions.items():
                if f_name not in self.feature_weights:
                    continue
                    
                # Did this feature contribute to the error?
                # e.g., error is -1.8 (predicted too high), shap_val is +0.5 (pulled model higher)
                # Then shap_val * error is negative (-0.9). This means it contributed negatively
                # to the accuracy. We want the decay factor to be < 1.0 for negative contributions.
                contribution = shap_val * error
                
                # Weight update: + contribution -> increase weight, - contribution -> decrease weight
                # Because PID can be negative (if error is negative), pid_adjustment * shap_val
                # correctly aligns the sign.
                
                # We use (1 + alpha * sign(contribution) * |pid|)
                # If error = -1.8 (neg), shap = 0.5 (pos) -> contribution is negative -> we want decay.
                # pid_adjustment will be roughly -0.9.
                # (-0.9) * 0.5 = -0.45.
                # 1 + (0.1 * -0.45) = 1 - 0.045 = 0.955 (Decay! Correct form)
                
                decay_factor = 1.0 + (self.alpha * pid_adjustment * shap_val)
                
                new_weight = self.feature_weights[f_name] * decay_factor
                self.feature_weights[f_name] = max(0.1, min(2.0, new_weight))
