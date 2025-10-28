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


class SDRStore:
    sdr: RtlSdr | None = None


class SignalStreamer:
    """
    A Tool for establishing streams of data from an RTL-SDR note that:
    - The SDR connection takes a second or so to start up
    - Only one connection can be open at a time
    - if the connection is not closed properly, the USB port can remain busy

    So, most of this architecture is built around these constraints, we only recreate the connection when we need to.

    This object stores the current signal for reference by other logic.

    At any time, the stop_stream() function can be used to stop processing.

    We can only stream one signal at a time, because we can only hold one SDR signal open at once.
    """

    def can_start_acquisition(self):
        return self.thread is None

    def __init__(self):
        self.enabled: bool = False
        # If thread is none, then no acquisition loop is running, and we can start one
        self.thread: Thread | None = None
        self.current_signal = np.array([])
        self.read_is_available = False

    def start_stream(
        self,
        center_freq: float,
        sample_rate: float,
        n_reads_per_acquisition=1,
        n_samples_per_read=2048,
        sleep_length_s=0.025,
        sdr_gain="auto",
    ):
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
            args=(
                center_freq,
                sample_rate,
                n_reads_per_acquisition,
                n_samples_per_read,
                sleep_length_s,
                sdr_gain,
            ),
            daemon=True,  # stops automatically if main thread exits
        )
        thread.start()
        logger.info(
            "SignalStreamer: Acquisition loop started with params",
            center_freq=center_freq,
            sample_rate=sample_rate,
            n_reads_per_acquisition=n_reads_per_acquisition,
            n_samples_per_read=n_samples_per_read,
            sleep_length_s=sleep_length_s,
            sdr_gain=sdr_gain,
        )
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

    def get_current_signal(self) -> np.ndarray:
        """
        Return the current signal, if it is available.

        If a read is in progress, return an empty array.

        We need this method to manage access to the current_signal variable, because it is being updated in a background thread.
        """
        while True:
            if self.read_is_available:
                break
            time.sleep(0.001)  # wait 1ms before checking again

        return self.current_signal

    def signal_acquisition_loop(
        self,
        center_freq: float,
        sample_rate: float,
        n_reads_per_acquisition=1,
        n_samples_per_read=2048,
        sleep_length_s=0.025,
        sdr_gain="auto",
    ):
        """
        Efficiently handle the SDR connection and begin acquiring samples.

        default parameters (n_reads_per_acquisition=1, n_samples_per_read=2048: int, sleep_length_s=0.025) allows 40 reads per second.

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

            # Preallocate memory
            self.current_signal = np.zeros(
                [n_reads_per_acquisition, n_samples_per_read], dtype=np.complex64
            )

            while self.enabled:
                # Disallow reading while data is being written into the current_signal store
                self.read_is_available = False

                for i in range(n_reads_per_acquisition):
                    self.current_signal[i, :] = sdr.read_samples(
                        num_samples=n_samples_per_read
                    )

                # Allow reading during sleep
                self.read_is_available = True
                time.sleep(sleep_length_s)

        except Exception as e:
            logger.exception("Error in acquisition loop")
            raise e
        finally:
            sdr.close()
            logger.info("SDR closed, acquisition loop ended")


streamer = SignalStreamer()
sdrStore = SDRStore()


def get_sdr(center_freq=102.7e6, sample_rate=1e6, sdr_gain="auto"):
    """
    Establish a connection to the RTL-SDR

    """
    # TODO update the values if get() is called with different params, rather than instantiating a new one
    if sdrStore.sdr is not None:
        logger.warning(
            "SDR instance already exists, closing it before creating a new one"
        )
        # We already have an SDR instance, just update the params
        sdrStore.sdr.close()
        sdrStore.sdr = None

    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq
    sdr.freq_correction = 60  # PPM
    sdr.gain = sdr_gain

    # Bin the first 2048 samples, they are just padding
    sdr.read_samples(2048)

    sdrStore.sdr = sdr

    return sdr


def acquire_signal(sdr, n_reads: int, n_samples=2048) -> np.ndarray:
    """Example function to demonstrate repeated signal acquisition over a chunk of time.

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
    # finally:, close sdr


def get_indices_of_highest_peaks(s, verbose=False, n=1):
    """
    Assumes the input is the result of acquire_signal(), meaning it is an (n, m) ndarray, with n reads

    For each read, find the 10th highest signal pulse received. This is more resillient to outlier pulses, if we received an ADSB from an aircraft, there wil be dozens of high strength pulses at roughly the same strength.

    Returns the index(es) with the highest peak, essentially giving you an index at which there is very likely a signal of some kind.
    """
    tenth_highest_pulse_per_read = np.sort(np.abs(s))[:, -10]
    # Get indices of n highest peaks in descending order
    indices = np.argsort(tenth_highest_pulse_per_read)[-n:][::-1]
    if verbose:
        logger.info("Indices of sampling batches with the highest peaks: %s", indices)
    return (
        indices if n > 1 else indices[0]
    )  # Return single value if n=1 for backward compatibility


if __name__ == "__main__":
    # Example of how to use the SignalStreamer

    # Create the shared state object
    streamer.start_stream(center_freq=102.7e6, sample_rate=2.4e6)

    # Print the output to demonstrate it is being updated
    for i in range(5):
        time.sleep(1)
        logger.info(f"Snapshot {i}", signal=streamer.current_signal[:16])

    # Stop the acquisition
    streamer.stop_stream()

    logger.info("Final signal snapshot", signal=streamer.current_signal[:16])
