import pytest
import pandas as pd
import numpy as np
from src.training.labeler import TargetLabeler

def test_target_labeler_up():
    price_series = pd.Series([100, 100.06])
    timestamps = pd.Series(pd.to_datetime(['2026-03-14 09:30:00', '2026-03-14 09:35:00']))
    
    labeler = TargetLabeler(target_minutes=5, threshold=0.0005)
    labels = labeler.compute_labels(price_series, timestamps)
    
    # log(100.06 / 100) = ~0.0006 > 0.0005 -> Class 1 (Up)
    assert labels[0] == 1
    assert pd.isna(labels[1]) # No future data for the last timestamp

def test_target_labeler_down():
    price_series = pd.Series([100, 99.94])
    timestamps = pd.Series(pd.to_datetime(['2026-03-14 09:30:00', '2026-03-14 09:35:00']))
    
    labeler = TargetLabeler(target_minutes=5, threshold=0.0005)
    labels = labeler.compute_labels(price_series, timestamps)
    
    # log(99.94 / 100) = ~-0.0006 < -0.0005 -> Class -1 (Down)
    assert labels[0] == -1
    
def test_target_labeler_flat():
    price_series = pd.Series([100, 100.01]) # Only moved 1 cent
    timestamps = pd.Series(pd.to_datetime(['2026-03-14 09:30:00', '2026-03-14 09:35:00']))
    
    labeler = TargetLabeler(target_minutes=5, threshold=0.0005)
    labels = labeler.compute_labels(price_series, timestamps)
    
    # log(100.01 / 100) = 0.0001 (Between -0.0005 and +0.0005) -> Class 0 (Flat)
    assert labels[0] == 0
