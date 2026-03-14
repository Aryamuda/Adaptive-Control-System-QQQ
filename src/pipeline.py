import pandas as pd
import numpy as np

from src.features.validator import SnapshotValidator
from src.features.buffer import WindowBuffer
from src.features.builder import FeatureBuilder

from src.models.xgb_model import XGBoostBaseline
from src.models.explainer import SHAPExplainerWrapper

from src.control.feedback import FeedbackQueue
from src.control.loop import CyberneticLoop
from src.control.regime import RegimeClassifier

from src.utils.logger import SignalLogger

class PipelineOrchestrator:
    """
    Main controller running the complete cybernetic scaffolding end-to-end.
    """
    
    def __init__(self, target_minutes=5):
        # 1. Features & Validation
        self.validator = SnapshotValidator()
        self.buffer = WindowBuffer(capacity=5)
        self.builder = FeatureBuilder(self.buffer)
        
        # 2. Models
        self.model = XGBoostBaseline()
        self.explainer = SHAPExplainerWrapper(self.model)
        
        # 3. Control Loop
        self.feedback_queue = FeedbackQueue(target_minutes=target_minutes, ttl_minutes=15)
        # Assuming we know the features we calculate in builder.py
        feature_names = ['net_gex', 'net_vanna', 'pc_oi_ratio', 'wavg_call_iv', 'max_call_oi_strike', 'max_put_oi_strike', 'delta_gex']
        self.loop = CyberneticLoop(feature_names=feature_names)
        
        # 4. Logger & History
        self.logger = SignalLogger()
        self.price_history = pd.Series(dtype=float) # Stores atm_strike for resolution

    def process_snapshot(self, snapshot_df: pd.DataFrame) -> None:
        """
        Executes a single step of the pipeline for a new 1-minute snapshot.
        """
        # --- A. Validation ---
        if not self.validator.validate(snapshot_df):
            # We don't skip processing entirely if it fails validation,
            # because we still need to process old feedback queue items against the new time.
            timestamp = pd.Timestamp.now() # Fallback if snapshot corrupt
            current_price = None
            is_valid_data = False
        else:
            # Assumes snapshot is ordered by time, take first row for meta
            timestamp = pd.to_datetime(snapshot_df['timestamp'].iloc[0])
            is_valid_data = True

        # --- B. Retroactive Feedback Processing ---
        if is_valid_data:
            # We use atm_strike as the proxy for current price for resolving prediction errors
            atm_strike = self.builder._get_atm_strike(snapshot_df) 
            self.price_history[timestamp] = atm_strike
            
            # Resolve matured predictions and update PID/Weights
            resolved_errors = self.feedback_queue.process_outcomes(
                current_timestamp=timestamp, 
                current_price=atm_strike, 
                price_history=self.price_history
            )
            
            if resolved_errors:
                self.loop.process_feedback(resolved_errors)
        else:
            # Still process timeouts even if current snapshot is bad
            self.feedback_queue.process_outcomes(timestamp, 0.0, self.price_history)


        # --- C. Live Inference & Cybernetic Control ---
        if is_valid_data:
            # 1. Build Raw Features limit 1300 rows -> 1 row
            raw_features = self.builder.build_features(snapshot_df)
            
            # 2. Get Regime
            regime = self.loop.regime_classifier.get_current_regime(timestamp, raw_features)
            
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
                current_weights=self.loop.feature_weights
            )
            
            print(f"[{timestamp}] Signal processed. Conf: {confidence:.2f} | Regime: {regime['time_regime']}/{regime['vol_regime']}")
