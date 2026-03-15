"""
Tests for PID Controller.
"""
import pytest
import numpy as np
from src.control.pid import PIDController


class TestPIDController:
    """Test cases for PIDController."""
    
    def test_proportional_only(self):
        """Test P-only controller."""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        
        # Error of 1.0 should give output of 1.0
        output = pid.update(error=1.0)
        assert output == pytest.approx(1.0)
    
    def test_proportional_integral(self):
        """Test PI controller."""
        pid = PIDController(kp=1.0, ki=0.5, kd=0.0)
        
        # First error
        out1 = pid.update(error=1.0)
        assert out1 == pytest.approx(1.0)  # P term only
        
        # Second error - should accumulate integral
        out2 = pid.update(error=1.0)
        # P=1.0 + I=0.5*1.0 = 1.5
        assert out2 == pytest.approx(1.5)
    
    def test_derivative_term(self):
        """Test PID with derivative term."""
        pid = PIDController(kp=0.0, ki=0.0, kd=1.0)
        
        # First call - no derivative yet
        out1 = pid.update(error=1.0)
        assert out1 == pytest.approx(0.0)
        
        # Second call - derivative = (2.0 - 1.0) / 1.0 = 1.0
        out2 = pid.update(error=2.0)
        assert out2 == pytest.approx(1.0)
    
    def test_full_pid(self):
        """Test full PID controller."""
        pid = PIDController(kp=1.0, ki=0.5, kd=0.5)
        
        # First call
        out1 = pid.update(error=1.0)
        # P=1.0, I=0.5, D=0 (no prev error)
        assert out1 == pytest.approx(1.0)
        
        # Second call
        out2 = pid.update(error=2.0)
        # P=2.0, I=1.5 (0.5*2.0), D=0.5 ((2-1)*0.5)
        assert out2 == pytest.approx(4.0)
    
    def test_reset(self):
        """Test PID reset clears integral and derivative state."""
        pid = PIDController(kp=1.0, ki=0.5, kd=0.0)
        
        # Accumulate integral
        pid.update(error=1.0)
        pid.update(error=1.0)
        
        # Reset
        pid.reset()
        
        # After reset, output should be just proportional
        out = pid.update(error=1.0)
        assert out == pytest.approx(1.0)
    
    def test_negative_error(self):
        """Test handling of negative errors."""
        pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        
        output = pid.update(error=-1.0)
        assert output == pytest.approx(-1.0)
    
    def test_zero_error(self):
        """Test handling of zero error."""
        pid = PIDController(kp=1.0, ki=0.5, kd=0.0)
        
        # Accumulate some integral
        pid.update(error=1.0)
        
        # Zero error should still have integral component
        output = pid.update(error=0.0)
        assert output == pytest.approx(0.5)  # Just integral
    
    def test_custom_dt(self):
        """Test custom time step."""
        pid = PIDController(kp=1.0, ki=0.5, kd=0.0)
        
        # With dt=2.0, integral should be doubled
        output = pid.update(error=1.0, dt=2.0)
        # P=1.0, I=0.5*2.0=1.0
        assert output == pytest.approx(2.0)
    
    def test_integral_windup(self):
        """Test that integral can grow unbounded (no anti-windup in base class)."""
        pid = PIDController(kp=0.0, ki=0.1, kd=0.0)
        
        # Accumulate integral over many steps
        for _ in range(100):
            pid.update(error=1.0)
        
        # Integral should be large
        output = pid.update(error=1.0)
        assert output > 10.0  # At least 10.0 from integral