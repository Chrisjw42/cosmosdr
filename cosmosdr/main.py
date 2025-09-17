from rtlsdr import RtlSdr


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


def main():
    print("Hello from cosmosdr!")

    try:
        sdr = get_sdr()

    except Exception as e:
        raise e
    finally:
        # Always ensure the sdr connection is closed, otherwise there can be issues with USB ports remaining busy
        sdr.close()


if __name__ == "__main__":
    main()
