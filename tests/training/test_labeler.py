"""
Tests for TargetLabeler.
"""
import pytest
import pandas as pd
import numpy as np
from src.training.labeler import TargetLabeler


class TestTargetLabeler:
    """Test cases for TargetLabeler."""
    
    def test_basic_up_label(self):
        """Test labeling of upward movement."""
        labeler = TargetLabeler(target_minutes=5, threshold=0.0005)
        
        prices = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 101.0])
        timestamps = pd.Series(pd.date_range("2026-03-14 09:30", periods=6, freq="1min"))
        
        labels = labeler.compute_labels(prices, timestamps)
        
        # First 5 should be NaN (no future price available)
        assert pd.isna(labels.iloc[0])
        assert pd.isna(labels.iloc[1])
        assert pd.isna(labels.iloc[2])
        assert pd.isna(labels.iloc[3])
        assert pd.isna(labels.iloc[4])
        
        # Last one should be labeled Up (log(101/100) > 0.0005)
        assert labels.iloc[5] == 1
    
    def test_basic_down_label(self):
        """Test labeling of downward movement."""
        labeler = TargetLabeler(target_minutes=5, threshold=0.0005)
        
        prices = pd.Series([101.0, 101.0, 101.0, 101.0, 101.0, 100.0])
        timestamps = pd.Series(pd.date_range("2026-03-14 09:30", periods=6, freq="1min"))
        
        labels = labeler.compute_labels(prices, timestamps)
        
        # Last one should be labeled Down
        assert labels.iloc[5] == -1
    
    def test_flat_label(self):
        """Test labeling of flat movement (within threshold)."""
        labeler = TargetLabeler(target_minutes=5, threshold=0.0005)
        
        # Small movement: log(100.04/100) = 0.0004 < 0.0005
        prices = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 100.04])
        timestamps = pd.Series(pd.date_range("2026-03-14 09:30", periods=6, freq="1min"))
        
        labels = labeler.compute_labels(prices, timestamps)
        
        # Should be flat
        assert labels.iloc[5] == 0
    
    def test_boundary_up(self):
        """Test boundary case for upward label."""
        labeler = TargetLabeler(target_minutes=5, threshold=0.0005)
        
        # Exactly at threshold: log(100.05/100) ≈ 0.0005
        prices = pd.Series([100.0] * 6 + [100.050001])
        timestamps = pd.Series(pd.date_range("2026-03-14 09:30", periods=7, freq="1min"))
        
        labels = labeler.compute_labels(prices, timestamps)
        
        # Should be Up (>= threshold)
        assert labels.iloc[6] == 1
    
    def test_boundary_down(self):
        """Test boundary case for downward label."""
        labeler = TargetLabeler(target_minutes=5, threshold=0.0005)
        
        # Exactly at negative threshold: log(99.95/100) ≈ -0.0005
        prices = pd.Series([100.0] * 6 + [99.95])
        timestamps = pd.Series(pd.date_range("2026-03-14 09:30", periods=7, freq="1min"))
        
        labels = labeler.compute_labels(prices, timestamps)
        
        # Should be Down (<= -threshold)
        assert labels.iloc[6] == -1
    
    def test_continuous_labels(self):
        """Test continuous label generation."""
        labeler = TargetLabeler(target_minutes=5, threshold=0.0005)
        
        prices = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 101.0])
        timestamps = pd.Series(pd.date_range("2026-03-14 09:30", periods=6, freq="1min"))
        
        continuous = labeler.compute_continuous_labels(prices, timestamps)
        
        # First 5 should be NaN
        assert pd.isna(continuous.iloc[0])
        
        # Last one should have log return
        assert continuous.iloc[5] == pytest.approx(np.log(101.0 / 100.0))
    
    def test_zero_price(self):
        """Test handling of zero price (edge case)."""
        labeler = TargetLabeler(target_minutes=5, threshold=0.0005)
        
        prices = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 100.0])
        timestamps = pd.Series(pd.date_range("2026-03-14 09:30", periods=6, freq="1min"))
        
        labels = labeler.compute_labels(prices, timestamps)
        
        # Should handle division by zero gracefully
        assert pd.isna(labels.iloc[5])  # log(0) is -inf
    
    def test_negative_price(self):
        """Test handling of negative price (shouldn't happen but test robustness)."""
        labeler = TargetLabeler(target_minutes=5, threshold=0.0005)
        
        prices = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, -1.0])
        timestamps = pd.Series(pd.date_range("2026-03-14 09:30", periods=6, freq="1min"))
        
        labels = labeler.compute_labels(prices, timestamps)
        
        # Should handle invalid price gracefully
        assert pd.isna(labels.iloc[5])
    
    def test_unsorted_timestamps(self):
        """Test that labels work with unsorted timestamps."""
        labeler = TargetLabeler(target_minutes=5, threshold=0.0005)
        
        # Prices that would be different if not sorted
        prices = pd.Series([100.0, 101.0, 100.0, 101.0, 100.0, 101.0])
        timestamps = pd.Series([
            "2026-03-14 09:35",
            "2026-03-14 09:30",
            "2026-03-14 09:36",
            "2026-03-14 09:31",
            "2026-03-14 09:37",
            "2026-03-14 09:32"
        ])
        
        labels = labeler.compute_labels(prices, timestamps)
        
        # Should sort by timestamp and produce valid labels
        assert len(labels) == 6
    
    def test_all_same_prices(self):
        """Test with all same prices (flat market)."""
        labeler = TargetLabeler(target_minutes=5, threshold=0.0005)
        
        prices = pd.Series([100.0] * 10)
        timestamps = pd.Series(pd.date_range("2026-03-14 09:30", periods=10, freq="1min"))
        
        labels = labeler.compute_labels(prices, timestamps)
        
        # All valid labels should be 0 (flat)
        valid_labels = labels.dropna()
        assert (valid_labels == 0).all()
    
    def test_large_target_minutes(self):
        """Test with target_minutes larger than data."""
        labeler = TargetLabeler(target_minutes=60, threshold=0.0005)
        
        prices = pd.Series([100.0] * 5)
        timestamps = pd.Series(pd.date_range("2026-03-14 09:30", periods=5, freq="1min"))
        
        labels = labeler.compute_labels(prices, timestamps)
        
        # All should be NaN because target is beyond data
        assert labels.isna().all()
