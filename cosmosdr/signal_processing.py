"""
3 main stages:
    1. signal_acquisition
    2. signal_processing << you are here
    3. presentation
"""

import numpy as np
import pandas as pd


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
