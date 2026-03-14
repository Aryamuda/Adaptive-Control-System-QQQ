import pandas as pd
from typing import List, Optional
from collections import deque

class WindowBuffer:
    """
    In-memory N-snapshot buffer.
    Maintains a rolling window of the last N valid snapshots to support 
    time-derivative features (e.g. delta GEX, gamma acceleration)
    without hitting disk I/O.
    """
    
    def __init__(self, capacity: int = 5):
        """
        Args:
            capacity: The number of snapshots to keep in memory (e.g. 5 for a 5-minute rolling window)
        """
        self.capacity = capacity
        # Storing data in a deque for O(1) append/pop left
        self.buffer: deque = deque(maxlen=capacity)
        
    def add(self, snapshot_df: pd.DataFrame) -> None:
        """
        Adds a new snapshot to the rolling buffer.
        """
        if snapshot_df.empty:
            raise ValueError("Cannot add an empty snapshot to the buffer.")
            
        # Ensure timestamp exists so we can sort/validate if needed
        if 'timestamp' not in snapshot_df.columns:
            raise ValueError("Snapshot is missing 'timestamp' column.")
            
        self.buffer.append(snapshot_df)

    def is_full(self) -> bool:
        """
        Returns True if the buffer has accumulated `capacity` snapshots.
        """
        return len(self.buffer) == self.capacity
        
    def get_latest(self) -> Optional[pd.DataFrame]:
        """
        Returns the most recent snapshot.
        """
        if not self.buffer:
            return None
        return self.buffer[-1]
        
    def get_oldest(self) -> Optional[pd.DataFrame]:
        """
        Returns the oldest snapshot currently in the window.
        """
        if not self.buffer:
            return None
        return self.buffer[0]
        
    def get_all(self) -> List[pd.DataFrame]:
        """
        Returns a list of all snapshots currently in memory (oldest to newest).
        """
        return list(self.buffer)

    def clear(self) -> None:
        """
        Clears the current buffer. Useful for day transitions.
        """
        self.buffer.clear()
