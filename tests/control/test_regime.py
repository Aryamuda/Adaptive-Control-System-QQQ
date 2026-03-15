"""
Tests for RegimeClassifier.
"""
import pytest
import pandas as pd
from src.control.regime import RegimeClassifier


class TestRegimeClassifier:
    """Test cases for RegimeClassifier."""
    
    def test_open_regime(self):
        """Test classification during market open."""
        classifier = RegimeClassifier()
        
        # 09:45 - should be 'open'
        timestamp = pd.Timestamp("2026-03-14 09:45:00")
        regime = classifier.classify_time_regime(timestamp)
        assert regime == 'open'
    
    def test_mid_regime(self):
        """Test classification during mid-day."""
        classifier = RegimeClassifier()
        
        # 12:00 - should be 'mid'
        timestamp = pd.Timestamp("2026-03-14 12:00:00")
        regime = classifier.classify_time_regime(timestamp)
        assert regime == 'mid'
    
    def test_close_regime(self):
        """Test classification during market close."""
        classifier = RegimeClassifier()
        
        # 15:30 - should be 'close'
        timestamp = pd.Timestamp("2026-03-14 15:30:00")
        regime = classifier.classify_time_regime(timestamp)
        assert regime == 'close'
    
    def test_pre_open_regime(self):
        """Test classification before market open."""
        classifier = RegimeClassifier()
        
        # 08:00 - should be 'pre_open'
        timestamp = pd.Timestamp("2026-03-14 08:00:00")
        regime = classifier.classify_time_regime(timestamp)
        assert regime == 'pre_open'
    
    def test_normal_vol_regime(self):
        """Test normal volatility regime classification."""
        classifier = RegimeClassifier()
        
        features = {'wavg_call_iv': 0.15}
        regime = classifier.classify_vol_regime(features)
        assert regime == 'normal'
    
    def test_high_vol_regime(self):
        """Test high volatility regime classification."""
        classifier = RegimeClassifier()
        
        features = {'wavg_call_iv': 0.35}
        regime = classifier.classify_vol_regime(features)
        assert regime == 'high_vol'
    
    def test_boundary_vol_regime(self):
        """Test boundary case for volatility regime."""
        classifier = RegimeClassifier()
        
        # Exactly at threshold - should be normal (not high)
        features = {'wavg_call_iv': 0.30}
        regime = classifier.classify_vol_regime(features)
        assert regime == 'normal'
    
    def test_missing_iv_feature(self):
        """Test handling of missing IV feature."""
        classifier = RegimeClassifier()
        
        features = {}
        regime = classifier.classify_vol_regime(features)
        assert regime == 'normal'  # Defaults to normal
    
    def test_get_current_regime(self):
        """Test combined regime classification."""
        classifier = RegimeClassifier()
        
        timestamp = pd.Timestamp("2026-03-14 10:00:00")
        features = {'wavg_call_iv': 0.20}
        
        regime_state = classifier.get_current_regime(timestamp, features)
        
        assert regime_state.time_regime == 'open'
        assert regime_state.vol_regime == 'normal'
    
    def test_get_current_regime_high_vol(self):
        """Test combined regime with high volatility."""
        classifier = RegimeClassifier()
        
        timestamp = pd.Timestamp("2026-03-14 14:00:00")
        features = {'wavg_call_iv': 0.50}
        
        regime_state = classifier.get_current_regime(timestamp, features)
        
        assert regime_state.time_regime == 'mid'
        assert regime_state.vol_regime == 'high_vol'
    
    def test_regime_to_dict(self):
        """Test RegimeState to_dict method."""
        classifier = RegimeClassifier()
        
        timestamp = pd.Timestamp("2026-03-14 10:00:00")
        features = {'wavg_call_iv': 0.20}
        
        regime_state = classifier.get_current_regime(timestamp, features)
        regime_dict = regime_state.to_dict()
        
        assert isinstance(regime_dict, dict)
        assert 'time_regime' in regime_dict
        assert 'vol_regime' in regime_dict
        assert regime_dict['time_regime'] == 'open'
        assert regime_dict['vol_regime'] == 'normal'