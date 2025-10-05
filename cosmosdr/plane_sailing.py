"""Precursor module with aircraft processing logic, that will eventually be moved into a standalone package"""

from copy import copy

import numpy as np
import pandas as pd
import structlog

import cosmosdr.signal_acquisition as s_acq
import cosmosdr.signal_processing as s_proc

logger = structlog.get_logger()


def get_example_dataset(
    center_freq=1090.0, sample_rate=2.4e6, n_reads=32, n_samples=4096
):
    try:
        sdr = s_acq.get_sdr(center_freq=center_freq, sample_rate=sample_rate)

        s = s_acq.acquire_signal(sdr, n_reads=n_reads, n_samples=n_samples)

        # Grab the read with the highest peak, which is likely an ADSB pulse
        highest_peak_read = s_acq.get_index_of_highest_peak(s)
        iq = s[highest_peak_read]

        return iq
    finally:
        sdr.close()


def trim_iq_around_peak(iq_mag, n_either_side=1000):
    """Trim the IQ data around the peak. Essentially assuming that there is a peak in the data, which is a single
    pulse, and we want to center on it.
    """
    idx = iq_mag.argmax()
    return iq_mag[idx - n_either_side : idx + n_either_side]


def shift_to_optimal_phase(iq_mag, samples_per_us):
    """Given that the signal is almost certainly not aligned perfectly with the initial sampling start time, and our
    eventual aim of bucketing these samples into 0.5us buckets, we want to try and center the aircraft pulses well
    onto the buckets.

    e.g. If we have oversampled and have 12x the samples, there are 12 possible 'start points' or phase positions.

    We assess each phase position, scoring them based on how much difference there is between neighboring 6-sample
    blocks (i.e. 0.5us blocks). The phase position with the highest average difference between neighboring blocks
    is likely to be the best position.

    # TODO parametrise this to different sample rates and whatnot
    """
    iq_mag = copy(iq_mag)

    max_score = -1
    max_score_phase = None

    # We check each of the first n starting points, after a while we just get back to the start of the cycle again so we can stop checking
    steps_to_check = int(samples_per_us)
    scores = {}

    # Step through the possible starting points
    for phase in range(0, steps_to_check - 1):
        data_phase = iq_mag[phase:]

        # 0,0,0,0,0,0,1,1,1....
        indices = np.arange(len(data_phase)) // steps_to_check

        data_phase = pd.Series(data_phase, index=indices)

        block_averages = data_phase.groupby(data_phase.index).mean()

        # calculate the differences between the blocks
        deltas = (block_averages - block_averages.shift(1)).abs()
        score = deltas.mean()
        scores[phase] = score

        # The score will oscillate naturally, so we only update if the score is significantly higher than the previous max
        if score > max_score * 1.01:
            max_score = score
            max_score_phase = phase

    # TODO do this in numpy, dict is lame
    logger.info("----------")
    logger.info("max_score: %s", max_score)
    logger.info("max_score_phase: %s", max_score_phase)

    iq_mag = iq_mag[max_score_phase:]
    return iq_mag


def downsample_to_buckets(iq_mag, samples_per_us):
    """
    Downsample the data back to the convenient bucketing (1 bucket per 0.5us)

    Uses mean to downsample, but we could also take the max from the bucket.
    """
    indices = np.arange(len(iq_mag)) // (samples_per_us / 2)
    data_downsampled = pd.DataFrame(iq_mag).groupby(indices).mean()
    return data_downsampled[0].to_numpy()


def preprocess_iq(iq, orig_sr, target_sr):
    """Resample the IQ data to a target sample rate, and extract the envelope

    Then, Find the optimal phase starting position
    - We don't know the exact timing of the pulses
    - It could be anywhere from n, ..., n+11
    - We can define the most optimal position as that which has the highest difference between neighboring 6-width blocks
    """
    iq_resampled, sr = s_proc.resample_to_target(iq, orig_sr, target_sr)
    iq_mag = s_proc.iq_to_envelope(iq_resampled)

    iq_mag = trim_iq_around_peak(iq_mag)

    samples_per_us = int(round(target_sr / 1e6))  # 12

    iq_mag = shift_to_optimal_phase(iq_mag, samples_per_us=samples_per_us)

    return iq_mag
