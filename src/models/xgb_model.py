"""
XGBoost baseline model for the Adaptive Control System.
Handles training and inference with proper time-series validation.
"""
import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from src.utils.config import get_config
from src.utils.types import FeatureDict

logger = logging.getLogger(__name__)


class XGBoostBaseline:
    """
    Baseline XGBoost classifier for inference.
    Handles inference of pre-computed feature vectors.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initializes the model. If a path is provided, loads the saved weights.
        
        Args:
            model_path: Optional path to load pre-trained model weights
        """
        self.model: Optional[xgb.XGBClassifier] = None
        self.is_trained = False
        self._feature_names: Optional[list] = None
        
        config = get_config()
        self._default_params = {
            'n_estimators': config.model.n_estimators,
            'max_depth': config.model.max_depth,
            'learning_rate': config.model.learning_rate,
            'subsample': config.model.subsample,
            'colsample_bytree': config.model.colsample_bytree,
            'early_stopping_rounds': config.model.early_stopping_rounds,
            'eval_metric': config.model.eval_metric
        }
        
        if model_path:
            self.load_model(model_path)
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_eval: Optional[pd.DataFrame] = None,
        y_eval: Optional[pd.Series] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Trains the XGBoost baseline.
        
        IMPORTANT: For time-series data, NEVER shuffle. Use a strict time-split
        for train/eval to avoid look-ahead bias.
        
        Args:
            X_train: Training features (must be time-ordered, no shuffling)
            y_train: Training labels
            X_eval: Optional evaluation features (should be later in time than train)
            y_eval: Optional evaluation labels
            params: Optional override parameters
            
        Raises:
            ValueError: If eval set not provided or if data appears shuffled
        """
        # Validate time-series integrity
        self._validate_time_series(X_train)
        
        # Build parameters
        train_params = self._default_params.copy()
        if params:
            train_params.update(params)
        
        # Handle class imbalance
        if 'scale_pos_weight' not in train_params and y_train is not None:
            neg_count = (y_train == -1).sum()
            pos_count = (y_train == 1).sum()
            if pos_count > 0 and neg_count > 0:
                train_params['scale_pos_weight'] = neg_count / pos_count
        
        # Create eval set - REQUIRED for proper early stopping
        if X_eval is None or y_eval is None:
            # Fallback: use last 20% of training data as eval (time-based split)
            split_idx = int(len(X_train) * 0.8)
            X_train_split = X_train.iloc[:split_idx]
            y_train_split = y_train.iloc[:split_idx]
            X_eval = X_train.iloc[split_idx:]
            y_eval = y_train.iloc[split_idx:]
            logger.warning(
                "No eval set provided. Using time-based split (80/20) from training data. "
                "For production, provide explicit time-split eval set."
            )
        
        self._validate_time_series(X_eval, is_eval=True)
        
        # Store feature names for inference
        self._feature_names = [c for c in X_train.columns if c != 'timestamp']
        
        # Initialize model
        self.model = xgb.XGBClassifier(**train_params)
        
        # Train with proper eval set
        self.model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_eval, y_eval)],
            verbose=False
        )
        
        self.is_trained = True
        logger.info(f"Model trained on {len(X_train_split)} samples, eval on {len(X_eval)} samples")
    
    def _validate_time_series(self, X: pd.DataFrame, is_eval: bool = False) -> None:
        """
        Validate that data appears to be time-ordered.
        
        Args:
            X: Feature DataFrame
            is_eval: Whether this is eval data
            
        Raises:
            ValueError: If data appears shuffled or has issues
        """
        if 'timestamp' not in X.columns:
            if not is_eval:
                logger.warning("No timestamp column found - cannot verify time-series order")
            return
        
        timestamps = pd.to_datetime(X['timestamp'])
        if not timestamps.is_monotonic_increasing:
            logger.warning(
                "Data timestamps are not monotonically increasing - "
                "data may be shuffled! This causes look-ahead bias."
            )
    
    def predict_proba(self, features: FeatureDict) -> float:
        """
        Runtime Inference: Given a single feature vector dictionary, 
        returns the probability of the positive class.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Probability of the positive class (Up)
            
        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning default probability 0.51")
            return 0.51
        
        # Convert single dict back to 1-row DataFrame for XGBoost
        feature_df = pd.DataFrame([features])
        
        # Ensure we have the same columns as training
        if self._feature_names:
            for col in self._feature_names:
                if col not in feature_df.columns:
                    feature_df[col] = 0.0
            feature_df = feature_df[self._feature_names]
        
        # Remove non-feature columns
        feature_df = feature_df.drop(columns=['timestamp'], errors='ignore')
        
        # Get probability of class '1' (Up)
        prob = self.model.predict_proba(feature_df)[0][1]
        return float(prob)
