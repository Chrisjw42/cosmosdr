"""
3 main stages:
    1. signal_acquisition
    2. signal_processing << you are here
    3. presentation
"""

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from fractions import Fraction


def get_frequency_space_np(s, center_freq, sample_rate):
    """Process the signal only, pass back a numpy object"""
    # The output x axis ranges from -(x/2) to (x/2), where x is the sample rate, and 0 is centered on the target frequency
    # In this case, it is 102.2 MHz +/- ~2 MHz

    # apply hamming window to ensure the first and last samples align
    s = s * np.hamming(s.shape[0])

    # fft the input, and center it on 0
    s_fft = np.fft.fftshift(np.fft.fft(s))

    # Calculate magnitude and angle of the complex signal
    s_mag_fft = np.abs(s_fft)
    s_angle_fft = np.angle(s_fft)

    # Estimate frequencies
    freqs = np.linspace(
        center_freq - (sample_rate / 2), center_freq + (sample_rate / 2), s.shape[0]
    )

    return freqs, s_fft, s_mag_fft, s_angle_fft


def get_frequency_space_df(s, center_freq, sample_rate):
    """Process the signal, and assemble into a pandas dataframe for easy plotting"""
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


def resample_to_target(s: np.ndarray, orig_sr: float, target_sr: float):
    """Try to rebuild the underlying analog waveform that would have produced your samples.

    This makes the 'bandlimited reconstruction assumption', which says that, 'I will construct
    the smoothest possible signal I can, on the assumption that there is no received signal above
    the nyquist of the sample rate.

    Principally, we make an assumption that the frequency is related to the sample rate, which of course may not be
    true, but we know it is in our case.

    Generates the mathematically correct reconstruction of what the signal would look like between samples (assuming
    the original SDR samples met Nyquist(which is 1/2 sample rate) )

    Nyquist is the maximum frequency you can capture without ambiguity at a given sample rate..
    """

    # TODO some input validation on the divisibility of orig_sr and target_sr

    # find integer up/down using Fraction
    ratio = Fraction(int(target_sr), int(orig_sr)).limit_denominator()
    up, down = ratio.numerator, ratio.denominator
    return resample_poly(s, up, down), target_sr


def iq_to_envelope(iq):
    """assumes `iq` is a 1D complex64 numpy array from your SDR, and sr is sample_rate (Hz)"""

    return np.abs(iq)
