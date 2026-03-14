import pytest
import pandas as pd
import numpy as np

from src.features.validator import SnapshotValidator

@pytest.fixture
def valid_snapshot():
    strikes = np.linspace(300, 420, 120)
    data = {
        'timestamp': ['2026-03-14 09:30:00'] * 120,
        'strike': strikes,
        'call_bid': [1.0] * 120, 'call_ask': [1.1] * 120,
        'call_iv': [0.2] * 120, 'call_delta': [0.5] * 120, 'call_gamma': [0.01]*120,
        'call_vega': [0.1]*120, 'call_theta': [-0.05]*120, 'call_rho': [0.02]*120,
        'call_vanna': [0.001]*120, 'call_charm': [-0.001]*120, 'call_oi': [100]*120, 'call_volume': [10]*120,
        'put_bid': [1.0]*120, 'put_ask': [1.1]*120, 
        'put_iv': [0.2]*120, 'put_delta': [-0.5]*120, 'put_gamma': [0.01]*120,
        'put_vega': [0.1]*120, 'put_theta': [-0.05]*120, 'put_rho': [-0.02]*120,
        'put_vanna': [0.001]*120, 'put_charm': [-0.001]*120, 'put_oi': [100]*120, 'put_volume': [10]*120
    }
    return pd.DataFrame(data)

def test_valid_snapshot(valid_snapshot):
    validator = SnapshotValidator()
    assert validator.validate(valid_snapshot) is True

def test_missing_column(valid_snapshot):
    invalid = valid_snapshot.drop(columns=['call_gamma'])
    validator = SnapshotValidator()
    assert validator.validate(invalid) is False

def test_missing_strikes(valid_snapshot):
    invalid = valid_snapshot.head(100) # Only 100 strikes
    validator = SnapshotValidator()
    assert validator.validate(invalid) is False

def test_null_greeks(valid_snapshot):
    invalid = valid_snapshot.copy()
    invalid.loc[:30, 'call_delta'] = np.nan # 31 nulls out of 120 > 10%
    validator = SnapshotValidator()
    assert validator.validate(invalid) is False

def test_empty_snapshot():
    validator = SnapshotValidator()
    assert validator.validate(pd.DataFrame()) is False
