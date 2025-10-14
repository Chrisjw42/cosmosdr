"""Precursor module with aircraft processing logic, that will eventually be moved into a standalone package"""

import time
from copy import copy

import numpy as np
import pandas as pd
import structlog
import rs1090

import cosmosdr.signal_acquisition as s_acq
import cosmosdr.signal_processing as s_proc


logger = structlog.get_logger()

ADSB_FREQUENCY = 1090e6
# Total number of bits in a single ADSB message
ADSB_BITS = 112
# One microsecond timeslot for one bit, 'on' if signal is within the first half of this slot
# ADSB_SLOT_LENGTH = 1 / 1e6


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


def shift_to_optimal_phase(iq_mag, samples_per_us, verbose=False):
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
    if verbose:
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
    # 0,0,0,0,0,0,1,1,1,1,1...
    # We want 0.5us buckets, so we need to half the samples_per_us
    indices = np.arange(len(iq_mag)) // (samples_per_us / 2)

    data_downsampled = pd.DataFrame(iq_mag).groupby(indices).mean()
    return data_downsampled[0].to_numpy()


def threshold_elbow(s, upper_pulses_to_ignore=20):
    """
    Identify the largest jump in values, after they have been sorted. Works well in cases where the
    pulses are well above the background noise.

    If we received an ADSB from an aircraft, there wil be dozens of high strength pulses at roughly the same strength,
    so we ignore the top N pulses.
    """
    x = np.sort(s)[:-upper_pulses_to_ignore]
    diffs = np.diff(x)

    if len(diffs) == 0:
        raise ValueError("Not enough unique values to determine a threshold")

    # grab the location of largest jump
    gap_idx = np.argmax(diffs)
    # Return the value half-way between the largest jump
    return np.mean([x[gap_idx], x[gap_idx + 1]])  # threshold is value at gap


def convert_to_binary(iq_mag):
    """
    Use an elbow analysis to identify a threshold, then convert the signal to binary based on this threshold.

    The returned signal is a binary array indicating the presence of pulses, it is NOT yet decoded, or represented as binary
    """

    threshold = threshold_elbow(iq_mag)
    iq_mag_bin = (iq_mag > threshold).astype(int)
    return iq_mag_bin


def preprocess_iq(iq, orig_sr, target_sr, verbose=False):
    """Resample the IQ data to a target sample rate, and optimise to highlight the presence of pulses

    The returned data is a 1D array of magnitudes, with one value per 0.5us bucket

    Then, Find the optimal phase starting position
    - We don't know the exact timing of the pulses
    - It could be anywhere from n, ..., n+11
    - We can define the most optimal position as that which has the highest difference between neighboring 6-width blocks
    """
    iq_resampled, sr = s_proc.resample_to_target(iq, orig_sr, target_sr)
    iq_mag = s_proc.iq_to_envelope(iq_resampled)

    iq_mag = trim_iq_around_peak(iq_mag)

    samples_per_us = int(round(target_sr / 1e6))  # 12

    iq_mag = shift_to_optimal_phase(
        iq_mag, samples_per_us=samples_per_us, verbose=verbose
    )

    iq_mag = downsample_to_buckets(iq_mag, samples_per_us=samples_per_us)

    return iq_mag


def decode_to_adsb(s_bin, verbose=False):
    """
    Take a binary pulse signal, and convert it to ADSB 1s and 0s.
    """
    # mode S preamble, 8us
    preamble = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    len_preamble = len(preamble)
    assert len_preamble == 16

    # Find where the preamble starts, if at all
    for i in range(len(s_bin)):
        candidate = s_bin[i : i + len_preamble]
        if np.array_equal(candidate, preamble):
            logger.info("ADSB preamble identified @ idx: %s, damn!", i)
            break
        if i >= (len(s_bin) - len_preamble):
            if verbose:
                logger.info("No ADSB preamble found, exiting")
            return None

    # the index of the last piece of the preamble
    final_preamble_idx = i + len_preamble

    # Drop the preamble off the front of the burst
    adsb = s_bin[final_preamble_idx:]

    # Note, the last part of the signal may indeed be one or many zeros, if this doens't fit into a standard msg length, that's likely why
    adsb = np.trim_zeros(adsb)

    # If we end up with exactly one less bit than was expected, then assume we just chopped off a bit
    if (len(adsb) + 1) == ADSB_BITS * 2:
        adsb = np.append(adsb, 0)

    # Take every other value (a.k.a. take the value in the first half of each bucket
    # This is the same as 'if there is a pule in the first half, then 1.'
    adsb = adsb[::2]
    return adsb


def convert_adsb_binary_to_hex(adsb):
    """Convert a binary array of 1s and 0s to a hex string, as is required by rs1090.decode()"""
    # Step 1: pack bits into bytes
    packed = np.packbits(adsb)

    # Step 2: convert to hex string
    hex_string = packed.tobytes().hex()

    return hex_string


def stream_and_decode_adsb(
    n,
    n_reads_per_acquisition=32,
    n_samples_per_read=4096,
    sleep_time=0.1,
    sample_rate=2.4e6,
    target_sr=12e6,
    verbose=False,
):
    """Start the signal streamer, then repeatedly grab the current signal, process it, and try to decode any ADSB messages.

    n: number of reads to attempt
    n_reads_per_acquisition: number of reads to acquire in each acquisition loop iteration
    n_samples_per_read: number of samples per read in each acquisition loop iteration
    sleep_time: time to wait between reads, purely for managing compute load
    sample_rate: sample rate to use for the SDR, needs to be adjusted in-line with target_sr, see process_iq()
    target_sr: target sample rate to resample to for processing, see process_iq()
    verbose: if True, log more information
    """
    hits = 0
    # Ensure any existing stream is stopped
    s_acq.streamer.stop_stream()

    # Kick off a stream
    s_acq.streamer.start_stream(
        center_freq=1090e6,
        sample_rate=sample_rate,
        n_reads_per_acquisition=n_reads_per_acquisition,
        n_samples_per_read=n_samples_per_read,
        sleep_length_s=sleep_time
        / 2,  # Ensure the signal stream is updated more frequently than the reads happen
        sdr_gain="auto",
    )
    time.sleep(3)  # Give the streamer a moment to start up

    aircraft_info = []

    for i in range(n):
        try:
            # Sleep a bit to allow the signal streamer to fill the buffer, and to not hammer CPU
            time.sleep(sleep_time)
            signal = s_acq.streamer.get_current_signal()

            # Pick the read with the highest peak, which is likely an ADSB pulse
            index_of_strongest_signal = s_acq.get_index_of_highest_peak(
                signal, verbose=verbose
            )
            iq = signal[index_of_strongest_signal]

            # Only do the expensive signal processing on the strongest read
            iq_mag = preprocess_iq(
                iq, orig_sr=sample_rate, target_sr=target_sr, verbose=verbose
            )
            iq_mag_bin = convert_to_binary(iq_mag)
            adsb = decode_to_adsb(iq_mag_bin, verbose=verbose)

            if adsb is not None:
                # We have what looks like an ADSB message, try to decode it
                if verbose:
                    logger.info("ADSB message identified, attempting to decode")
                hex_string = convert_adsb_binary_to_hex(adsb)
                decoded = rs1090.decode(hex_string)
                if decoded:
                    logger.info("ADSB message decoded: %s", decoded)
                    hits += 1
                    aircraft_info.append(decoded)
                else:
                    decoded = None

        except Exception as e:
            logger.warning("Error processing read, moving on to next read", error=e)
            # Don't worry if one read fails, just move on to the next read
            # We expect to receive chopped off signals and imperfect data
    if verbose:
        logger.info("Hit rate: %s\t/\t%s", hits, n)
    return aircraft_info


def main():
    """Example pipeline to acquire and process an ADSB signal from an SDR"""

    # reccomended upper limit of sample rate. Fast enough to oversample
    sample_rate = 2.4e6
    n_reads = 32
    n_samples = 4096

    # choose integer samples per microsecond: 12 samples/us
    # This is a good choice, because 6x2 = 12, and 5*2.4=12, so we can sample to 12, then subsample back to 1 block per 0.5us
    target_sr = 12e6

    # samples_per_us_after_upsampling = int(round(target_sr / 1e6))  # 12

    iq = get_example_dataset(
        center_freq=ADSB_FREQUENCY,
        sample_rate=sample_rate,
        n_reads=n_reads,
        n_samples=n_samples,
    )

    iq_mag = preprocess_iq(iq, orig_sr=sample_rate, target_sr=target_sr)

    iq_mag_bin = convert_to_binary(iq_mag)

    adsb = decode_to_adsb(iq_mag_bin)

    return adsb
