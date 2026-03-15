"""
Main pipeline orchestrator for the Adaptive Control System.
Runs the complete cybernetic scaffolding end-to-end.
"""
import pandas as pd
import numpy as np
import logging

from src.features.validator import SnapshotValidator
from src.features.buffer import WindowBuffer
from src.features.builder import FeatureBuilder

from src.models.xgb_model import XGBoostBaseline
from src.models.explainer import SHAPExplainerWrapper

from src.control.feedback import FeedbackQueue
from src.control.loop import CyberneticLoop
from src.control.regime import RegimeClassifier

from src.utils.logger import SignalLogger
from src.utils.config import get_config
from src.utils.types import FeatureVector

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Main controller running the complete cybernetic scaffolding end-to-end.
    """
    
    def __init__(self, target_minutes: int = None):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            target_minutes: Override for target_minutes. Uses config if None.
        """
        config = get_config()
        
        # 1. Features & Validation
        self.validator = SnapshotValidator(
            expected_strikes=config.features.expected_strikes,
            expected_columns=config.features.expected_columns
        )
        self.buffer = WindowBuffer(capacity=config.features.buffer_capacity)
        self.builder = FeatureBuilder(self.buffer)
        
        # 2. Models
        self.model = XGBoostBaseline()
        self.explainer = SHAPExplainerWrapper(self.model)
        
        # 3. Control Loop
        self.feedback_queue = FeedbackQueue(
            target_minutes=target_minutes,
            ttl_minutes=config.pipeline.ttl_minutes
        )
        
        # Get feature names from builder (not hardcoded!)
        self._feature_names = self.builder.feature_names
        self.loop = CyberneticLoop(feature_names=self._feature_names)
        
        # 4. Logger & History
        self.logger = SignalLogger()
        self.price_history = pd.Series(dtype=float)  # Stores atm_strike for resolution
        
        logger.info(
            f"PipelineOrchestrator initialized: target_minutes={target_minutes or config.pipeline.target_minutes}, "
            f"buffer_capacity={config.features.buffer_capacity}, "
            f"features={self._feature_names}"
        )
    
    @property
    def feature_names(self) -> list:
        """Return the list of feature names."""
        return self._feature_names
    
    def process_snapshot(self, snapshot_df: pd.DataFrame) -> None:
        """
        Executes a single step of the pipeline for a new 1-minute snapshot.
        
        Args:
            snapshot_df: DataFrame containing the snapshot data
        """
        # --- A. Validation ---
        is_valid_data = self.validator.validate(snapshot_df)
        
        if not is_valid_data:
            # We don't skip processing entirely if it fails validation,
            # because we still need to process old feedback queue items against the new time.
            timestamp = pd.Timestamp.now()  # Fallback if snapshot corrupt
            current_price = None
            logger.warning("Snapshot validation failed, using fallback timestamp")
        else:
            # Assumes snapshot is ordered by time, take first row for meta
            timestamp = pd.to_datetime(snapshot_df['timestamp'].iloc[0])
        
        # --- B. Retroactive Feedback Processing ---
        if is_valid_data:
            # We use atm_strike as the proxy for current price for resolving prediction errors
            atm_strike = self.builder.get_atm_strike(snapshot_df)
            self.price_history[timestamp] = atm_strike
            
            # Resolve matured predictions and update PID/Weights
            resolved_errors = self.feedback_queue.process_outcomes(
                current_timestamp=timestamp, 
                current_price=atm_strike, 
                price_history=self.price_history
            )
            
            if resolved_errors:
                self.loop.process_feedback(resolved_errors)
                logger.debug(f"Processed {len(resolved_errors)} resolved errors")
        else:
            # Still process timeouts even if current snapshot is bad
            # Pass None for price to indicate invalid data
            self.feedback_queue.process_outcomes(
                timestamp, 
                0.0,  # Placeholder - won't be used for resolution
                self.price_history
            )
        
        # --- C. Live Inference & Cybernetic Control ---
        if is_valid_data:
            # 1. Build Raw Features (1300 rows -> 1 row)
            # Note: buffer.add() is called inside build_features by default
            feature_vector = self.builder.build_features(snapshot_df)
            raw_features = feature_vector.to_dict()
            
            # 2. Get Regime
            regime_state = self.loop.regime_classifier.get_current_regime(
                timestamp, 
                raw_features
            )
            regime = regime_state.to_dict()
            
            # 3. Apply Adaptive Weights
            adjusted_features = self.loop.adjust_features(raw_features)
            
            # 4. XGBoost Inference (Baseline)
            # The model predicts using the raw features (in a real setup, models
            # shouldn't be trained on dynamically scaled features unless re-trained constantly)
            # BUT the scaffolding allows using the scaled features either way.
            # Using adjusted features here to show end-to-end effect.
            raw_prediction = self.model.predict_proba(adjusted_features)
            
            # 5. Extract SHAP Attribution
            attributions = self.explainer.get_local_attribution(adjusted_features)
            
            # 6. Apply Control Logic (Scale down confidence based on regime/prediction strength)
            confidence = self.loop.generate_confidence(raw_prediction, regime)
            
            # --- D. Post-Inference Storage ---
            # Enqueue prediction for future evaluation at t+N
            self.feedback_queue.add_prediction(
                timestamp=timestamp,
                prediction=raw_prediction,
                features=raw_features,
                attributions=attributions
            )
            
            # 7. Log output
            self.logger.log_signal(
                timestamp=str(timestamp),
                raw_prediction=raw_prediction,
                confidence=confidence,
                regime=regime,
                current_weights=self.loop.get_weights()
            )
            
            logger.info(
                f"[{timestamp}] Signal processed. Conf: {confidence:.2f} | "
                f"Regime: {regime['time_regime']}/{regime['vol_regime']}"
            )
    
    def get_status(self) -> dict:
        """Get current pipeline status."""
        return {
            'queue_size': self.feedback_queue.get_queue_size(),
            'feature_weights': self.loop.get_weights(),
            'price_history_len': len(self.price_history)
        }
    
    def reset(self) -> None:
        """Reset the pipeline for a new trading day."""
        self.buffer.clear()
        self.feedback_queue.clear()
        self.loop.reset_weights()
        self.price_history = pd.Series(dtype=float)
        logger.info("Pipeline reset for new trading day")
