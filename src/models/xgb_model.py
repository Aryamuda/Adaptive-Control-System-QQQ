import xgboost as xgb
import pandas as pd
from typing import Dict, Any

class XGBoostBaseline:
    """
    Baseline XGBoost classifier for inference.
    Handles inference of pre-computed feature vectors.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initializes the model. If a path is provided, loads the saved weights.
        """
        self.model = xgb.XGBClassifier()
        self.is_trained = False
        
        if model_path:
            self.load_model(model_path)
            
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, params: Dict[str, Any] = None) -> None:
        """
        Trains the XGBoost baseline. (Offline process)
        Never shuffles time-series for train/test split.
        """
        default_params = {
            'n_estimators': 500,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 50,
            'eval_metric': 'auc'
        }
        
        if params:
            default_params.update(params)
            
        # Ensure scale_pos_weight is handled if classes are imbalanced
        if 'scale_pos_weight' not in default_params and y_train is not None:
            neg_count = (y_train == -1).sum()
            pos_count = (y_train == 1).sum()
            if pos_count > 0:
                default_params['scale_pos_weight'] = neg_count / pos_count
                
        # Re-initialize with new params
        self.model = xgb.XGBClassifier(**default_params)
        
        # We need an eval set for early stopping. For scaffolding, we just use train set.
        # In real usage, this should be a strict time-split holdout.
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
        self.is_trained = True
        
    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        Runtime Inference: Given a single feature vector dictionary, 
        returns the probability of the positive class.
        """
        if not self.is_trained:
            # For scaffolding/testing without a trained model, return a dummy probability
            return 0.51
            
        # Convert single dict back to 1-row DataFrame for XGBoost
        # Excluding non-feature metadata like timestamp
        feature_df = pd.DataFrame([features]).drop(columns=['timestamp'], errors='ignore')
        
        # [:, 1] gets the probability of class '1' (Up)
        prob = self.model.predict_proba(feature_df)[0][1]
        return float(prob)

    def save_model(self, path: str) -> None:
        self.model.save_model(path)
        
    def load_model(self, path: str) -> None:
        self.model.load_model(path)
        self.is_trained = True
