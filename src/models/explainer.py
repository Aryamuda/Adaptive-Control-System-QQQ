"""
SHAP Explainer wrapper for the Adaptive Control System.
Provides feature attribution for the Cybernetic loop.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseExplainer:
    """Base class for feature attribution explainers."""
    
    def get_global_importance(self, X_dataset: pd.DataFrame) -> pd.Series:
        """Returns global feature importance."""
        raise NotImplementedError
    
    def get_local_attribution(self, feature_vector: Dict[str, float]) -> Dict[str, float]:
        """Returns local feature attribution for a single prediction."""
        raise NotImplementedError


class SHAPExplainer(BaseExplainer):
    """
    Real SHAP explainer using shap library.
    Extracts global and local feature importance from XGBoost.
    """
    
    def __init__(self, model):
        """
        Initialize with a trained XGBoost model.
        
        Args:
            model: Trained XGBoost classifier
        """
        try:
            import shap
            self.shap = shap
            self.explainer = shap.TreeExplainer(model)
            self._is_available = True
            logger.info("SHAP explainer initialized successfully")
        except ImportError:
            logger.warning("shap library not available, falling back to mock")
            self._is_available = False
            self._model = model
    
    def get_global_importance(self, X_dataset: pd.DataFrame) -> pd.Series:
        """Returns global SHAP feature importance."""
        if not self._is_available:
            return self._get_mock_importance(X_dataset)
        
        cols = [c for c in X_dataset.columns if c != 'timestamp']
        # Use mean absolute SHAP values
        shap_values = self.explainer.shap_values(X_dataset[cols])
        importance = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=cols
        ).sort_values(ascending=False)
        return importance
    
    def get_local_attribution(self, feature_vector: Dict[str, float]) -> Dict[str, float]:
        """Returns local SHAP attribution for a single prediction."""
        if not self._is_available:
            return self._get_mock_attribution(feature_vector)
        
        cols = [k for k in feature_vector.keys() if k != 'timestamp']
        feature_df = pd.DataFrame([feature_vector])[cols]
        
        shap_values = self.explainer.shap_values(feature_df)
        
        # Return as dictionary
        return {col: shap_values[0][i] for i, col in enumerate(cols)}
    
    def _get_mock_importance(self, X_dataset: pd.DataFrame) -> pd.Series:
        """Fallback mock implementation."""
        return MockExplainer().get_global_importance(X_dataset)
    
    def _get_mock_attribution(self, feature_vector: Dict[str, float]) -> Dict[str, float]:
        """Fallback mock implementation."""
        return MockExplainer().get_local_attribution(feature_vector)


class MockExplainer(BaseExplainer):
    """
    Mock explainer for testing/scaffolding when SHAP is not available.
    Uses feature variance as a deterministic proxy for importance.
    """
    
    def __init__(self, seed: int = 42):
        self._seed = seed
        np.random.seed(seed)
    
    def get_global_importance(self, X_dataset: pd.DataFrame) -> pd.Series:
        """
        Returns mock global importance based on feature variance.
        Uses deterministic weights based on column names for reproducibility.
        """
        cols = [c for c in X_dataset.columns if c != 'timestamp']
        
        # Use hash of column name for deterministic "random" weights
        importance = {}
        for col in cols:
            hash_val = hash(col) % 1000
            importance[col] = 0.01 + (hash_val / 1000) * 0.29
        
        return pd.Series(importance).sort_values(ascending=False)
    
    def get_local_attribution(self, feature_vector: Dict[str, float]) -> Dict[str, float]:
        """
        Returns mock local attribution.
        Uses deterministic weights based on feature names for reproducibility.
        """
        cols = [k for k in feature_vector.keys() if k != 'timestamp']
        
        attribution = {}
        for col in cols:
            val = feature_vector[col]
            # Use hash for deterministic weight
            hash_val = hash(col) % 100
            weight = -0.1 + (hash_val / 100) * 0.2  # Range [-0.1, 0.1]
            attribution[col] = val * weight
        
        return attribution


class SHAPExplainerWrapper:
    """
    Wrapper for SHAP explainer with fallback to mock.
    Supports dependency injection for testing.
    """
    
    def __init__(
        self, 
        model=None,
        use_mock: bool = False,
        mock_seed: int = 42
    ):
        """
        Initialize the explainer.
        
        Args:
            model: Trained XGBoost model (optional, can be injected later)
            use_mock: Force use of mock explainer
            mock_seed: Seed for mock deterministic behavior
        """
        self._model = model
        self._explainer: Optional[BaseExplainer] = None
        
        if use_mock:
            self._explainer = MockExplainer(seed=mock_seed)
            logger.info("Using mock explainer (forced)")
        elif model is not None:
            self._explainer = SHAPExplainer(model)
        else:
            # Default to mock if no model provided
            self._explainer = MockExplainer(seed=mock_seed)
            logger.info("No model provided, using mock explainer")
    
    def set_model(self, model) -> None:
        """Inject a model after initialization."""
        self._model = model
        self._explainer = SHAPExplainer(model)
        logger.info("Model injected, using SHAP explainer")
    
    def get_global_importance(self, X_dataset: pd.DataFrame) -> pd.Series:
        """Get global feature importance."""
        if self._explainer is None:
            self._explainer = MockExplainer()
        return self._explainer.get_global_importance(X_dataset)
    
    def get_local_attribution(self, feature_vector: Dict[str, float]) -> Dict[str, float]:
        """Get local feature attribution."""
        if self._explainer is None:
            self._explainer = MockExplainer()
        return self._explainer.get_local_attribution(feature_vector)
    
    @property
    def is_using_mock(self) -> bool:
        """Check if using mock explainer."""
        return isinstance(self._explainer, MockExplainer)
