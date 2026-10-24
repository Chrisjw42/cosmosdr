"""Tests for radioviz-support additions, using a mocked RtlSdr (no hardware needed)."""

import threading
import time

import numpy as np
import pytest

from cosmosdr import signal_acquisition
from cosmosdr.streaming import (
    ContinuousStreamer,
    FrequencySweeper,
    RingBuffer,
    StreamConfig,
    SweepConfig,
    averaged_spectrum,
    u8_to_complex64,
)


class LibUSBError(Exception):
    pass


class FakeSdr:
    """Mimics the parts of rtlsdr.RtlSdr we use. Emits a tone at +tone_hz from the centre."""

    def __init__(self, serial=None, tone_hz=100e3):
        self.serial = serial
        self.sample_rate = 2.4e6
        self.center_freq = 100e6
        self._ppm = 0
        self.gain = "auto"
        self.closed = False
        self.tone_hz = tone_hz
        self._cancel = threading.Event()
        self._phase = 0
        self.set_ppm_calls = []

    # librtlsdr refuses to set the ppm it already holds
    @property
    def freq_correction(self):
        return self._ppm

    @freq_correction.setter
    def freq_correction(self, v):
        self.set_ppm_calls.append(v)
        if v == self._ppm:
            raise LibUSBError("Could not set freq. offset")
        self._ppm = v

    def _make_bytes(self, n_samples):
        t = (self._phase + np.arange(n_samples)) / self.sample_rate
        self._phase += n_samples
        iq = 0.5 * np.exp(2j * np.pi * self.tone_hz * t)
        raw = np.empty(2 * n_samples, dtype=np.uint8)
        raw[0::2] = np.clip(np.round(iq.real * 127.5 + 127.5), 0, 255)
        raw[1::2] = np.clip(np.round(iq.imag * 127.5 + 127.5), 0, 255)
        return raw

    def read_bytes(self, num_bytes):
        return self._make_bytes(num_bytes // 2).tobytes()

    def read_samples(self, n):
        return u8_to_complex64(self._make_bytes(n))

    def read_bytes_async(self, callback, num_bytes):
        period = (num_bytes / 2) / self.sample_rate
        while not self._cancel.is_set():
            callback(bytearray(self._make_bytes(num_bytes // 2).tobytes()), self)
            time.sleep(period)

    def cancel_read_async(self):
        self._cancel.set()

    def close(self):
        self.closed = True


def test_u8_conversion():
    raw = np.array([0, 255, 127, 128], dtype=np.uint8)
    iq = u8_to_complex64(raw)
    assert iq.dtype == np.complex64
    assert iq[0].real == pytest.approx(-1.0) and iq[0].imag == pytest.approx(1.0)
    assert abs(iq[1]) < 0.01


def test_ring_buffer_wrap_and_overrun():
    rb = RingBuffer(10)
    rb.write(np.arange(7, dtype=np.complex64))
    rb.write(np.arange(7, 14, dtype=np.complex64))
    latest, idx = rb.read_latest(5)
    assert idx == 9
    assert np.array_equal(latest.real, np.arange(9, 14))
    got, nxt, overrun = rb.read_since(0)
    assert overrun == 4 and nxt == 14
    assert np.array_equal(got.real, np.arange(4, 14))
    got, nxt, overrun = rb.read_since(12, max_n=1)
    assert np.array_equal(got.real, [12]) and nxt == 13 and overrun == 0


def test_ring_buffer_oversized_write():
    rb = RingBuffer(4)
    rb.write(np.arange(10, dtype=np.complex64))
    latest, idx = rb.read_latest(4)
    assert np.array_equal(latest.real, [6, 7, 8, 9]) and idx == 6


def test_get_sdr_ppm_default_unchanged_and_zero_ok(monkeypatch):
    fakes = []

    def factory(serial=None):
        f = FakeSdr(serial)
        fakes.append(f)
        return f

    monkeypatch.setattr(
        signal_acquisition, "RtlSdr", lambda **kw: factory(kw.get("serial_number"))
    )
    signal_acquisition.sdrStore.sdr = None
    sdr = signal_acquisition.get_sdr()
    assert sdr.freq_correction == 60  # plane-sailing behaviour preserved
    sdr = signal_acquisition.get_sdr(freq_correction_ppm=0, serial="00000001")
    # a fresh device already holds 0: we must not ask librtlsdr to set it again
    assert sdr.set_ppm_calls == [] and sdr.freq_correction == 0
    assert sdr.serial == "00000001"
    signal_acquisition.sdrStore.sdr = None


def test_continuous_streamer_delivers_tone():
    fake = FakeSdr(tone_hz=150e3)
    s = ContinuousStreamer(
        StreamConfig(center_freq=100e6, callback_bytes=2 * 24000),
        sdr_factory=lambda _s: fake,
    )
    s.start()
    deadline = time.time() + 2
    while s.buffer.total_written < 48000 and time.time() < deadline:
        time.sleep(0.01)
    iq, _ = s.buffer.read_latest(4096)
    s.stop()
    assert fake.closed
    freqs, p = averaged_spectrum(iq, 1024, 2.4e6)
    assert freqs[np.argmax(p)] == pytest.approx(150e3, abs=2 * 2.4e6 / 1024)
    assert s.stats.samples_received >= 48000


def test_continuous_streamer_retune_tracks_generation():
    fake = FakeSdr()
    s = ContinuousStreamer(
        StreamConfig(callback_bytes=4800), sdr_factory=lambda _s: fake
    )
    s.start()
    time.sleep(0.05)
    s.retune(120e6)
    assert fake.center_freq == 120e6 and s.stats.tune_generation == 1
    s.stop()


def test_sweeper_stitches_contiguous_range():
    cfg = SweepConfig(
        start_hz=100e6,
        stop_hz=110e6,
        fft_size=256,
        settle_samples=1024,
        dwell_samples=4096,
    )
    with FrequencySweeper(cfg, sdr_factory=lambda _s: FakeSdr()) as sw:
        centers = sw.hop_centers()
        freqs, power = sw.sweep()
    assert len(centers) == int(np.ceil(10e6 / (2.4e6 * 0.8)))
    assert freqs[0] >= 100e6 - 2.4e6 / 256 and freqs[-1] >= 110e6 - 2.4e6
    assert np.all(np.diff(freqs) > 0)  # monotonic, no overlaps
    assert len(freqs) == len(power)
