import pytest
import pandas as pd
from src.control.loop import CyberneticLoop
from src.control.regime import RegimeClassifier

def test_regime_classification():
    rc = RegimeClassifier()
    
    t_open = pd.Timestamp("2026-03-14 09:35:00")
    t_mid = pd.Timestamp("2026-03-14 11:30:00")
    t_close = pd.Timestamp("2026-03-14 15:45:00")
    
    assert rc.classify_time_regime(t_open) == 'open'
    assert rc.classify_time_regime(t_mid) == 'mid'
    assert rc.classify_time_regime(t_close) == 'close'
    
    features_normal = {'wavg_call_iv': 0.15}
    features_high_vol = {'wavg_call_iv': 0.35}
    
    assert rc.classify_vol_regime(features_normal) == 'normal'
    assert rc.classify_vol_regime(features_high_vol) == 'high_vol'

def test_cybernetic_loop_decay():
    f_names = ['feat A', 'feat B']
    cl = CyberneticLoop(feature_names=f_names, alpha=0.1)
    
    # 1. Initial weights should be 1.0
    assert cl.feature_weights['feat A'] == 1.0
    
    # Simulate an error.
    # Prediction: 0.8 (Up), Realized: -1.0 (Down).
    # Error (realized - predicted) = -1.8
    # Feat A pulled prediction UP (Sharpe +0.5). Wrong! Should decay.
    # Feat B pushed prediction DOWN (Sharpe -0.2). Right! Output should strengthen slightly.
    
    mock_errors = [{
        'prediction_time': pd.Timestamp.now(),
        'resolution_time': pd.Timestamp.now(),
        'error': -1.8, # -1.8
        'attributions': {'feat A': 0.5, 'feat B': -0.2}
    }]
    
    # This also tests integration with the PID term natively processing the error
    cl.process_feedback(mock_errors)
    
    # Feat A contributed negatively to accuracy (pushed opposite of realized)
    # Expected weight < 1.0
    assert cl.feature_weights['feat A'] < 1.0
    
    # Feat B contributed positively to accuracy (pushed towards realized)
    # Expected weight > 1.0
    assert cl.feature_weights['feat B'] > 1.0
