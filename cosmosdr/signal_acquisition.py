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


class SignalStreamer:
    """
    A Tool for establishing streams of data from an RTL-SDR note that:
    - this takes a second or so
    - only one connection can be open at a time
    - if the connection is not closed properly, the USB port can remain busy

    So, most of this architecture is built around these constraints, we only recreate the connection when we need to.

    This object stores the current signal for reference by other logic.

    At any time, the .enabled bool can be set to false, to stop processing.

    Can only stream one signal at a time, because we can only hold one SDR signal open at once.
    """

    def can_start_acquisition(self):
        return self.thread is None

    def __init__(self):
        # If thread is none, then no acquisition loop is running, and we can start one
        self.enabled: bool = False
        self.thread: Thread | None = None
        self.current_signal = np.array([])

    def start_stream(self, center_freq: float, sample_rate: float, sdr_gain="auto"):
        """
        Tell the SignalStreamer to begin streaming with a givne set of parameters.

        Creates a thread to run the acquisition loop in the background.
        """
        if not self.can_start_acquisition():
            logger.warning("SignalStreamer: Acquisition loop already running")

        self.enabled = True

        # Start acquisition loop in background thread
        thread = Thread(
            target=self.signal_acquisition_loop,
            args=(center_freq, sample_rate, sdr_gain),
            daemon=True,  # stops automatically if main thread exits
        )
        thread.start()
        self.thread = thread

    def stop_stream(self):
        """Tell the SignalStreamer to stop streaming, if there is an acquisition loop running."""
        if self.thread is None:
            logger.warning("SignalStreamer: Acquisition loop not running")
            return

        # Signal the acquisition loop to stop
        self.enabled = False

        # Wait for clean shutdown
        self.thread.join()
        del self.thread
        self.thread = None
        logger.info("SignalStreamer: Acquisition loop stopped")

    def signal_acquisition_loop(
        self, center_freq: float, sample_rate: float, sdr_gain="auto"
    ):
        """
        Efficiently handle the SDR connection and begin acquiring samples.

        This is intended to be run in a background thread.

        It will continuously run until self.enabled is set to False.

        It will Continuously updates signal_stream.current_signal.
        """
        sdr = get_sdr(
            center_freq=center_freq, sample_rate=sample_rate, sdr_gain=sdr_gain
        )
        try:
            # Discard initial padding samples
            sdr.read_samples(2048)

            logger.info("Acquisition loop started")
            while self.enabled:
                self.current_signal = sdr.read_samples(2048)
                # slight delay to avoid overwhelming the CPU, this allows 40FPS
                time.sleep(0.025)

        except Exception as e:
            logger.exception("Error in acquisition loop")
            raise e
        finally:
            sdr.close()
            logger.info("SDR closed, acquisition loop ended")


def get_sdr(center_freq=102.7e6, sample_rate=1e6, sdr_gain="auto"):
    """
    Establish a connection to the RTL-SDR
    """
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq
    sdr.freq_correction = 60  # PPM
    sdr.gain = sdr_gain

    # Bin the first 2048 samples, they are just padding
    sdr.read_samples(2048)
    return sdr


def acquire_signal(sdr, n_reads: int, n_samples=2048) -> np.ndarray:
    """_summary_

    Args:
        sdr (_type_): sdr object.
        n_reads (int): how many times to read sammples from the SDR
        n_samples (int, optional): passthrough to SDR. Defaults to 2048.
    Returns:
        np.ndarray: array of shape (n_reads, n_samples) containing complex samples
    """
    t0 = time.time()
    try:
        # Preallocate memory
        reads = np.zeros([n_reads, n_samples], dtype=np.complex64)
        for i in range(n_reads):
            reads[i, :] = sdr.read_samples(num_samples=n_samples)
        t1 = time.time()

        print(f"Acquired {n_reads} reads of {n_samples} samples in {t1 - t0:.2f}s")
    except Exception as e:
        # Always ensure the sdr connection is closed, otherwise there can be issues with USB ports remaining busy
        sdr.close()
        raise e

    return reads
    # finally:


signal_streamer = SignalStreamer()

if __name__ == "__main__":
    # Example of how to use the SignalStreamer

    # Create the shared state object
    streamer = SignalStreamer()
    signal_streamer.start_stream(center_freq=102.7e6, sample_rate=2.4e6)

    # Print the output to demonstrate it is being updated
    for i in range(5):
        time.sleep(1)
        logger.info(f"Snapshot {i}", signal=signal_streamer.current_signal[:16])

    # Stop the acquisition
    signal_streamer.stop_stream()

    logger.info("Final signal snapshot", signal=signal_streamer.current_signal[:16])
