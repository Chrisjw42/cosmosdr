"""
3 main stages:
    1. signal_acquisition << you are here
    2. signal_processing
    3. presentation
"""

import time
import structlog
import numpy as np
from rtlsdr import RtlSdr
from threading import Thread

logger = structlog.get_logger()


class SignalStream:
    """
    Used for a signal acquisition loop.

    This object stores the current signal for reference by other logic.

    At any time, the .enabled bool can be set to false, to stop processing.
    """

    def __init__(self):
        self.enabled = False
        self.current_signal = np.array([])


def get_sdr(center_freq=102.7e6, sample_rate=1e6, sdr_gain="auto"):
    """
    SDR setup, based on example code from rtlsdr
    """
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq
    sdr.freq_correction = 60  # PPM
    sdr.gain = sdr_gain
    return sdr


async def acquire_signal(center_freq=102.7e6, sample_rate=1e6):
    try:
        sdr = get_sdr(center_freq=center_freq, sample_rate=sample_rate)

        # Bin the first 2048 samples, they are just padding
        sdr.read_samples(2048)

        for i in range(100):
            logger.info(i)
            s = sdr.read_samples(4096)
            logger.info(s[:16])
    except Exception as e:
        raise e
    finally:
        # Always ensure the sdr connection is closed, otherwise there can be issues with USB ports remaining busy
        sdr.close()


def signal_acquisition_loop(signal_stream, center_freq=102.7e6, sample_rate=1e6):
    """
    Blocking acquisition loop, intended to be run in a background thread.

    Continuously updates signal_stream.current_signal.
    """
    sdr = get_sdr(center_freq=center_freq, sample_rate=sample_rate)
    try:
        # Discard initial padding samples
        sdr.read_samples(2048)

        logger.info("Acquisition loop started")
        while signal_stream.enabled:
            signal_stream.current_signal = sdr.read_samples(4096)
    except Exception as e:
        logger.exception("Error in acquisition loop")
        raise e
    finally:
        sdr.close()
        logger.info("SDR closed, acquisition loop ended")


if __name__ == "__main__":
    # Create the shared state object
    signal_stream = SignalStream()
    signal_stream.enabled = True

    # Start acquisition loop in background thread
    thread = Thread(
        target=signal_acquisition_loop,
        args=(signal_stream,),
        daemon=True,  # stops automatically if main thread exits
    )
    thread.start()

    # Let it run for a few seconds, simulating main thread doing other work
    for i in range(5):
        time.sleep(1)
        logger.info(f"Snapshot {i}", signal=signal_stream.current_signal[:16])

    # Stop the acquisition
    signal_stream.enabled = False
    thread.join()  # wait for clean shutdown
    logger.info("Final signal snapshot", signal=signal_stream.current_signal[:16])
