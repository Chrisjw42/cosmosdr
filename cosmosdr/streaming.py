"""
Continuous (gap-free) streaming and frequency sweeping for the RTL-SDR.

`SignalStreamer` in signal_acquisition is a *snapshot* streamer: it reads a block, sleeps, and drops everything in
between. That is ideal for ADSB burst hunting, but a continuous waterfall, audio demodulation or long integrations need
every sample. `ContinuousStreamer` uses librtlsdr's async read API to land every sample into a ring buffer.

`FrequencySweeper` hops the tuner across a range wider than one 2.4 MHz slice, discarding the samples captured while
the tuner settles after each retune, and stitches the central part of each slice's spectrum together.

    librtlsdr async thread ──uint8 IQ──► _on_bytes ──complex64──► RingBuffer ──► read_latest / read_since
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import structlog

from cosmosdr.signal_acquisition import open_sdr, set_freq_correction

logger = structlog.get_logger()

# uint8 -> float32 lookup: the RTL's ADC is offset-binary, centred on 127.5.
_U8_TO_F32 = ((np.arange(256, dtype=np.float32) - 127.5) / 127.5).astype(np.float32)


def u8_to_complex64(raw: np.ndarray) -> np.ndarray:
    """Convert interleaved uint8 I,Q,I,Q... to complex64 via a lookup table (several times faster than arithmetic)."""
    return _U8_TO_F32[raw].view(np.complex64)


class RingBuffer:
    """A fixed-size, lock-protected ring of complex64 samples, addressed by an ever-increasing sample index."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._buf = np.zeros(self.capacity, dtype=np.complex64)
        self._lock = threading.Lock()
        self.total_written = 0  # index of the next sample to be written

    def write(self, samples: np.ndarray) -> None:
        n = len(samples)
        if n == 0:
            return
        with self._lock:
            if n >= self.capacity:
                samples = samples[-self.capacity :]
                self.total_written += n - self.capacity
                n = self.capacity
            start = self.total_written % self.capacity
            first = min(n, self.capacity - start)
            self._buf[start : start + first] = samples[:first]
            if first < n:
                self._buf[: n - first] = samples[first:]
            self.total_written += n

    def _copy_range(self, start_idx: int, end_idx: int) -> np.ndarray:
        n = end_idx - start_idx
        out = np.empty(n, dtype=np.complex64)
        s = start_idx % self.capacity
        first = min(n, self.capacity - s)
        out[:first] = self._buf[s : s + first]
        if first < n:
            out[first:] = self._buf[: n - first]
        return out

    def read_latest(self, n: int) -> tuple[np.ndarray, int]:
        """The most recent n samples (fewer if not yet available), and the index of the first returned sample."""
        with self._lock:
            end = self.total_written
            n = min(n, end, self.capacity)
            return self._copy_range(end - n, end), end - n

    def read_since(
        self, index: int, max_n: int | None = None
    ) -> tuple[np.ndarray, int, int]:
        """Samples from `index` onwards, for consumers that must not miss any (audio, integration).

        Returns (samples, next_index, overrun) where overrun counts samples the consumer was too slow to read —
        they were overwritten before it got to them.
        """
        with self._lock:
            end = self.total_written
            oldest = max(0, end - self.capacity)
            overrun = max(0, oldest - index)
            start = max(index, oldest)
            if max_n is not None:
                end = min(end, start + max_n)
            return self._copy_range(start, end), end, overrun


@dataclass
class StreamStats:
    samples_received: int = 0
    dropped_samples: int = 0  # estimated from wall clock vs. samples delivered
    callbacks: int = 0
    started_at: float = 0.0
    last_tune_index: int = (
        0  # sample index at which the most recent retune took effect (approx.)
    )
    tune_generation: int = 0
    clipped_fraction: float = (
        0.0  # fraction of raw bytes at 0 or 255 in the latest block (ADC clipping)
    )


@dataclass
class StreamConfig:
    center_freq: float = 100e6
    sample_rate: float = 2.4e6
    gain: float | str = "auto"
    freq_correction_ppm: int = 0
    serial: str | None = None
    buffer_seconds: float = 4.0
    # Bytes per async callback. 2 bytes per sample; 256 KiB -> ~55 ms at 2.4 MS/s.
    callback_bytes: int = 256 * 1024
    extra: dict = field(default_factory=dict)


class ContinuousStreamer:
    """Gap-free streaming into a ring buffer using librtlsdr's async reads.

    Usage:
        s = ContinuousStreamer(StreamConfig(center_freq=100e6))
        s.start()
        iq, idx = s.buffer.read_latest(4096)
        s.retune(101e6)
        s.stop()

    `sdr_factory` exists for tests: pass a callable returning a mock with the RtlSdr interface.
    """

    # After this many seconds of streaming, start comparing wall-clock against samples delivered.
    DROP_ESTIMATE_WARMUP_S = 1.0

    def __init__(
        self,
        config: StreamConfig,
        sdr_factory: Callable[[str | None], object] = open_sdr,
    ):
        self.config = config
        self._sdr_factory = sdr_factory
        self.sdr = None
        self.buffer = RingBuffer(int(config.buffer_seconds * config.sample_rate))
        self.stats = StreamStats()
        self._thread: threading.Thread | None = None
        self._ctl_lock = threading.Lock()
        self._error: BaseException | None = None
        self._expected_baseline: tuple[float, int] | None = None
        self.running = False

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        with self._ctl_lock:
            if self.running:
                return
            cfg = self.config
            sdr = self._sdr_factory(cfg.serial)
            try:
                sdr.sample_rate = cfg.sample_rate
                sdr.center_freq = cfg.center_freq
                set_freq_correction(sdr, cfg.freq_correction_ppm)
                sdr.gain = cfg.gain
            except Exception:
                sdr.close()
                raise
            self.sdr = sdr
            self.stats = StreamStats(started_at=time.monotonic())
            self._expected_baseline = None
            self._error = None
            self.running = True
            self._thread = threading.Thread(
                target=self._run, name="cosmosdr-continuous", daemon=True
            )
            self._thread.start()
            logger.info(
                "ContinuousStreamer started",
                center_freq=cfg.center_freq,
                sample_rate=cfg.sample_rate,
            )

    def _run(self) -> None:
        try:
            # Blocks until cancel_read_async() is called.
            self.sdr.read_bytes_async(self._on_bytes, self.config.callback_bytes)
        except BaseException as e:  # noqa: BLE001 - surfaced via .error
            self._error = e
            logger.exception("ContinuousStreamer async read failed")
        finally:
            self.running = False

    def stop(self, timeout: float = 5.0) -> None:
        with self._ctl_lock:
            sdr, thread = self.sdr, self._thread
            if sdr is None:
                return
            try:
                sdr.cancel_read_async()
            except Exception:
                logger.exception("cancel_read_async failed")
            if thread is not None:
                thread.join(timeout)
                if thread.is_alive():
                    logger.warning(
                        "ContinuousStreamer thread did not exit within timeout"
                    )
            try:
                sdr.close()
            except Exception:
                logger.exception("Error closing SDR")
            self.sdr = None
            self._thread = None
            self.running = False
            logger.info("ContinuousStreamer stopped", **self.stats_dict())

    @property
    def error(self) -> BaseException | None:
        return self._error

    # ---------------------------------------------------------------- control
    def retune(self, center_freq: float) -> None:
        """Change the centre frequency without stopping the stream. Samples before stats.last_tune_index are from
        the old frequency (and a few ms after it are the tuner settling)."""
        if self.sdr is None:
            raise RuntimeError("not streaming")
        self.sdr.center_freq = center_freq
        self.config.center_freq = center_freq
        self.stats.last_tune_index = self.buffer.total_written
        self.stats.tune_generation += 1

    def set_gain(self, gain: float | str) -> None:
        if self.sdr is None:
            raise RuntimeError("not streaming")
        self.sdr.gain = gain
        self.config.gain = gain

    def set_freq_correction(self, ppm: int) -> None:
        if self.sdr is None:
            raise RuntimeError("not streaming")
        set_freq_correction(self.sdr, ppm)
        self.config.freq_correction_ppm = int(round(ppm))

    # ---------------------------------------------------------------- data path
    def _on_bytes(self, values, _context) -> None:
        raw = np.frombuffer(values, dtype=np.uint8)
        # Clipping: the 8-bit ADC is pinned at its rails.
        clipped = np.count_nonzero((raw == 0) | (raw == 255))
        self.stats.clipped_fraction = clipped / max(1, len(raw))
        iq = u8_to_complex64(raw)
        self.buffer.write(iq)
        st = self.stats
        st.samples_received += len(iq)
        st.callbacks += 1
        now = time.monotonic()
        if self._expected_baseline is None:
            if now - st.started_at >= self.DROP_ESTIMATE_WARMUP_S:
                self._expected_baseline = (now, st.samples_received)
        else:
            t0, n0 = self._expected_baseline
            expected = (now - t0) * self.config.sample_rate
            got = st.samples_received - n0
            # One callback of slack: delivery is chunked.
            deficit = expected - got - len(iq)
            if deficit > st.dropped_samples:
                st.dropped_samples = int(deficit)

    def stats_dict(self) -> dict:
        st = self.stats
        elapsed = max(1e-9, time.monotonic() - st.started_at) if st.started_at else 0.0
        return {
            "samples_received": st.samples_received,
            "dropped_samples": st.dropped_samples,
            "callbacks": st.callbacks,
            "effective_rate": st.samples_received / elapsed if elapsed else 0.0,
            "clipped_fraction": st.clipped_fraction,
            "tune_generation": st.tune_generation,
        }


# ------------------------------------------------------------------------------------------------------- sweeping


@dataclass
class SweepConfig:
    start_hz: float = 24e6
    stop_hz: float = 1500e6
    sample_rate: float = 2.4e6
    usable_fraction: float = (
        0.8  # central part of each slice kept (avoid filter roll-off and the DC spike)
    )
    fft_size: int = 1024
    settle_samples: int = 2 * 16384  # discarded after each retune
    dwell_samples: int = 16 * 1024  # averaged per hop
    gain: float | str = 30.0
    freq_correction_ppm: int = 0
    serial: str | None = None


@dataclass
class HopResult:
    center_hz: float
    freqs_hz: np.ndarray
    power_db: np.ndarray
    retune_s: float
    read_s: float


class FrequencySweeper:
    """Hop across [start, stop] one usable slice at a time and stitch the spectra.

    Uses synchronous reads (retuning mid-async-stream would make settle bookkeeping fiddly). Each hop:
    retune -> discard settle_samples -> read dwell_samples -> FFT-average -> keep the central usable_fraction.
    """

    def __init__(
        self,
        config: SweepConfig,
        sdr_factory: Callable[[str | None], object] = open_sdr,
    ):
        self.config = config
        self._sdr_factory = sdr_factory
        self.sdr = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    def open(self) -> None:
        cfg = self.config
        sdr = self._sdr_factory(cfg.serial)
        try:
            sdr.sample_rate = cfg.sample_rate
            set_freq_correction(sdr, cfg.freq_correction_ppm)
            sdr.gain = cfg.gain
        except Exception:
            sdr.close()
            raise
        self.sdr = sdr

    def close(self) -> None:
        if self.sdr is not None:
            self.sdr.close()
            self.sdr = None

    @property
    def step_hz(self) -> float:
        return self.config.sample_rate * self.config.usable_fraction

    def hop_centers(self) -> np.ndarray:
        cfg = self.config
        step = self.step_hz
        first = cfg.start_hz + step / 2
        n = int(np.ceil((cfg.stop_hz - cfg.start_hz) / step))
        return first + step * np.arange(n)

    def hop(self, center_hz: float) -> HopResult:
        cfg = self.config
        t0 = time.perf_counter()
        self.sdr.center_freq = center_hz
        t1 = time.perf_counter()
        # Settle: read and throw away. read_bytes returns 2 bytes per sample.
        self.sdr.read_bytes(2 * cfg.settle_samples)
        raw = np.frombuffer(self.sdr.read_bytes(2 * cfg.dwell_samples), dtype=np.uint8)
        t2 = time.perf_counter()
        iq = u8_to_complex64(raw)
        freqs, power_db = averaged_spectrum(
            iq, cfg.fft_size, cfg.sample_rate, center_hz
        )
        keep = _central_mask(cfg.fft_size, cfg.usable_fraction)
        return HopResult(
            center_hz, freqs[keep], power_db[keep], retune_s=t1 - t0, read_s=t2 - t1
        )

    def sweep(
        self, on_hop: Callable[[HopResult], None] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """One full sweep. Returns (freqs_hz, power_db) stitched across all hops."""
        fs, ps = [], []
        for c in self.hop_centers():
            r = self.hop(float(c))
            if on_hop:
                on_hop(r)
            fs.append(r.freqs_hz)
            ps.append(r.power_db)
        return np.concatenate(fs), np.concatenate(ps)


def averaged_spectrum(
    iq: np.ndarray, fft_size: int, sample_rate: float, center_hz: float = 0.0
):
    """Hann-windowed, power-averaged spectrum (Welch without overlap). Returns (freqs_hz, power_db)."""
    n_frames = len(iq) // fft_size
    if n_frames == 0:
        raise ValueError("not enough samples for one FFT frame")
    frames = iq[: n_frames * fft_size].reshape(n_frames, fft_size)
    win = np.hanning(fft_size).astype(np.float32)
    spec = np.fft.fftshift(np.fft.fft(frames * win, axis=1), axes=1)
    power = (np.abs(spec) ** 2).mean(axis=0) / (win**2).sum()
    freqs = center_hz + np.fft.fftshift(np.fft.fftfreq(fft_size, 1 / sample_rate))
    return freqs, 10 * np.log10(power + 1e-20)


def _central_mask(n: int, fraction: float) -> np.ndarray:
    keep = int(round(n * fraction))
    lo = (n - keep) // 2
    mask = np.zeros(n, dtype=bool)
    mask[lo : lo + keep] = True
    return mask
