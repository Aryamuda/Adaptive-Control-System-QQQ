import pytest
import pandas as pd
import numpy as np

# A real module would separate this out, simulating it for test structure
from src.control.feedback import FeedbackQueue

def test_feedback_queue_matching():
    queue = FeedbackQueue(target_minutes=5, ttl_minutes=15)
    
    # 1. Add a prediction at 9:30
    t0 = pd.Timestamp('2026-03-14 09:30:00')
    queue.add_prediction(
        timestamp=t0,
        prediction=0.6,
        features={'atm_strike': 100.0},
        attributions={'feat1': 0.1, 'feat2': -0.05}
    )
    
    assert len(queue.queue) == 1
    
    # 2. Simulate 9:31, no match yet
    # Price history is a series with DatetimeIndex
    t1 = pd.Timestamp('2026-03-14 09:31:00')
    history_931 = pd.Series([100.0, 100.01], index=[t0, t1])
    
    resolved = queue.process_outcomes(t1, 100.01, history_931)
    
    assert len(resolved) == 0
    assert len(queue.queue) == 1
    
    # 3. Simulate 9:35, match found!
    t5 = pd.Timestamp('2026-03-14 09:35:00')
    history_935 = pd.Series([100.0, 100.01, 100.06], index=[t0, t1, t5])
    
    resolved = queue.process_outcomes(t5, 100.06, history_935)
    
    assert len(resolved) == 1
    assert len(queue.queue) == 0 # Queue cleared
    
    # Check error calc (log(100.06/100) > 0.0005 -> Class 1 -> Error = 1 - 0.6 = +0.4)
    assert np.isclose(resolved[0]['error'], 0.4)
    assert resolved[0]['attributions']['feat1'] == 0.1

def test_feedback_queue_expiry():
    queue = FeedbackQueue(target_minutes=5, ttl_minutes=15)
    
    # 1. Add prediction at 9:30
    t0 = pd.Timestamp('2026-03-14 09:30:00')
    queue.add_prediction(t0, 0.6, {'atm_strike': 100.0}, {})
    
    # 2. Simulate 9:40, missing the 9:35 resolution price
    t10 = pd.Timestamp('2026-03-14 09:40:00')
    # No price at 9:35 in history!
    history_940 = pd.Series([100.0, 100.10], index=[t0, t10])
    
    resolved = queue.process_outcomes(t10, 100.10, history_940)
    
    assert len(resolved) == 0
    assert len(queue.queue) == 1 # Still hanging around waiting for grace period
    
    # 3. Simulate 9:55, past TTL (9:30 + 5 + 15 = 9:50)
    t25 = pd.Timestamp('2026-03-14 09:55:00')
    history_955 = pd.Series([100.0, 100.10, 100.20], index=[t0, t10, t25])
    
    resolved = queue.process_outcomes(t25, 100.20, history_955)
    
    assert len(resolved) == 0
    assert len(queue.queue) == 0 # Trashed!
