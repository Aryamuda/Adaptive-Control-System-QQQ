import pytest
import pandas as pd
import numpy as np
from src.features.buffer import WindowBuffer
from src.features.builder import FeatureBuilder

@pytest.fixture
def valid_snapshot():
    # create sample data
    strikes = np.linspace(300, 420, 120)
    data = {
        'timestamp': ['2026-03-14 09:30:00'] * 120,
        'strike': strikes,
        'call_bid': [1.0] * 120, 'call_ask': [1.1] * 120,
        'call_iv': [0.2] * 120, 'call_delta': np.linspace(1.0, 0.0, 120), 'call_gamma': [0.01]*120,
        'call_vega': [0.1]*120, 'call_theta': [-0.05]*120, 'call_rho': [0.02]*120,
        'call_vanna': [0.001]*120, 'call_charm': [-0.001]*120, 'call_oi': [100]*120, 'call_volume': [10]*120,
        'put_bid': [1.0]*120, 'put_ask': [1.1]*120, 
        'put_iv': [0.2]*120, 'put_delta': np.linspace(0.0, -1.0, 120), 'put_gamma': [0.01]*120,
        'put_vega': [0.1]*120, 'put_theta': [-0.05]*120, 'put_rho': [-0.02]*120,
        'put_vanna': [0.001]*120, 'put_charm': [-0.001]*120, 'put_oi': [100]*120, 'put_volume': [10]*120
    }
    return pd.DataFrame(data)

def test_feature_stability(valid_snapshot):
    """
    Ensures that processing the same snapshot twice (even if shuffled)
    yields identical deterministic output.
    """
    buffer1 = WindowBuffer(capacity=5)
    builder1 = FeatureBuilder(buffer1)
    
    buffer2 = WindowBuffer(capacity=5)
    builder2 = FeatureBuilder(buffer2)
    
    # Process original
    features1 = builder1.build_features(valid_snapshot)
    
    # Process randomly shuffled variant
    shuffled_snapshot = valid_snapshot.sample(frac=1.0, random_state=42).reset_index(drop=True)
    features2 = builder2.build_features(shuffled_snapshot)
    
    assert features1 == features2

def test_delta_gex_calculation(valid_snapshot):
    """
    Tests if delta GEX correctly computes the difference between the newest
    and oldest snapshot in the buffer.
    """
    buffer = WindowBuffer(capacity=2)
    builder = FeatureBuilder(buffer)
    
    # T1
    f1 = builder.build_features(valid_snapshot)
    assert f1['delta_gex'] == 0.0 # First snapshot
    
    # T2 (same data, delta should be 0 again)
    f2 = builder.build_features(valid_snapshot)
    assert f2['delta_gex'] == 0.0
    
    # T3 (higher gamma -> higher GEX)
    higher_gamma = valid_snapshot.copy()
    higher_gamma['call_gamma'] = 0.02
    
    f3 = builder.build_features(higher_gamma)
    assert f3['delta_gex'] > 0.0 # GEX increased since T2
