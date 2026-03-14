import pytest
import pandas as pd
from src.features.buffer import WindowBuffer

def test_buffer_add_and_get():
    buffer = WindowBuffer(capacity=3)
    
    df1 = pd.DataFrame({'timestamp': ['T1'], 'data': [1]})
    df2 = pd.DataFrame({'timestamp': ['T2'], 'data': [2]})
    
    buffer.add(df1)
    buffer.add(df2)
    
    assert not buffer.is_full()
    
    latest = buffer.get_latest()
    assert latest.iloc[0]['timestamp'] == 'T2'
    
    oldest = buffer.get_oldest()
    assert oldest.iloc[0]['timestamp'] == 'T1'

def test_buffer_capacity():
    buffer = WindowBuffer(capacity=2)
    
    df1 = pd.DataFrame({'timestamp': ['T1']})
    df2 = pd.DataFrame({'timestamp': ['T2']})
    df3 = pd.DataFrame({'timestamp': ['T3']})
    
    buffer.add(df1)
    buffer.add(df2)
    assert buffer.is_full()
    
    buffer.add(df3) # Should evict df1
    
    assert buffer.is_full()
    assert buffer.get_oldest().iloc[0]['timestamp'] == 'T2'
    assert buffer.get_latest().iloc[0]['timestamp'] == 'T3'
    assert len(buffer.get_all()) == 2

def test_buffer_empty_add():
    buffer = WindowBuffer()
    with pytest.raises(ValueError):
        buffer.add(pd.DataFrame())

def test_buffer_missing_timestamp():
    buffer = WindowBuffer()
    with pytest.raises(ValueError):
         buffer.add(pd.DataFrame({'data': [1]}))
