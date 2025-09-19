import time
from cosmosdr.signal_acquisition import SignalStreamer
import numpy as np


def test_basic_init():
    streamer = SignalStreamer()

    assert streamer.can_start_acquisition()
    assert streamer.enabled is False
    assert streamer.thread is None
    assert isinstance(streamer.current_signal, np.ndarray)
    assert streamer.current_signal.size == 0

    streamer.start_stream(102.7, 2.4e6)
    time.sleep(2.5)  # Give it a moment to start
    assert streamer.enabled is True
    assert streamer.thread is not None
    assert isinstance(streamer.current_signal, np.ndarray)
    assert streamer.current_signal.size > 0

    streamer.stop_stream()
