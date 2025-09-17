import structlog
import numpy as np
import pandas as pd
from rtlsdr import RtlSdr

logger = structlog.get_logger()


def get_sdr(center_freq=102.7e6, sample_rate=1e6, sdr_gain="auto"):
    """
    Based on example code from rtlsdr
    """
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate  # Hz
    sdr.center_freq = center_freq  # Hz, ~100MHz
    sdr.freq_correction = 60  # PPM

    sdr.gain = sdr_gain

    return sdr


def get_frequency_space_df(s, center_freq, sample_rate):
    # The output x axis ranges from -(x/2) to (x/2), where x is the sample rate, and 0 is centered on the target frequency
    # In this case, it is 102.2 MHz +/- ~2 MHz

    # fft the input, and center it on 0
    s_fft = np.fft.fftshift(np.fft.fft(s))

    dfft = pd.DataFrame(s_fft, columns=["signal_fft"])

    # Calculate magnitude and angle of the complex signal
    dfft["signal_mag_fft"] = dfft["signal_fft"].apply(np.abs)
    dfft["signal_angle_fft"] = dfft["signal_fft"].apply(np.angle)

    # Estimate frequencies
    dfft["frequency"] = np.linspace(
        center_freq - (sample_rate / 2), center_freq + (sample_rate / 2), dfft.shape[0]
    )


def main():
    print("Hello from cosmosdr!")

    center_freq = 102.7e6
    sample_rate = 1e6

    try:
        sdr = get_sdr(center_freq=center_freq, sample_rate=sample_rate)

        # Bin the first 2048 samples, they are just padding
        sdr.read_samples(2048)

        for i in range(100):
            logger.info(i)
            s = sdr.read_samples(4096)
            logger.info(s[:16])

            # Mock the signal processing work to give a sense of speed
            get_frequency_space_df(s, center_freq=center_freq, sample_rate=sample_rate)

    except Exception as e:
        raise e
    finally:
        # Always ensure the sdr connection is closed, otherwise there can be issues with USB ports remaining busy
        sdr.close()


if __name__ == "__main__":
    main()
