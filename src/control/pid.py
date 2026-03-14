class PIDController:
    """
    Proportional-Integral-Derivative Controller.
    Used to dynamically adjust feature weights based on sequential prediction errors.
    """
    
    def __init__(self, kp: float = 0.5, ki: float = 0.0, kd: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral_error = 0.0
        self.prev_error = 0.0
        
    def update(self, error: float, dt: float = 1.0) -> float:
        """
        Calculates the PID output based on the current error.
        
        Args:
            error: The current error signal (realized - predicted)
            dt: Time step (default 1 for discrete minute steps)
            
        Returns:
            The control adjustment value.
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (accumulated error over time to correct bias)
        self.integral_error += error * dt
        i_term = self.ki * self.integral_error
        
        # Derivative term (rate of error change to dampen overshoot/catch shifts)
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        # Save state
        self.prev_error = error
        
        return p_term + i_term + d_term

    def reset(self) -> None:
        """
        Resets the integral and derivative state. Useful across days or regime breaks.
        """
        self.integral_error = 0.0
        self.prev_error = 0.0
