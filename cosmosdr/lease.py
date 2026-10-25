"""Sharing one RTL-SDR between programs on one machine: the lease.

An RTL-SDR can be open in one process at a time. On cosbox, plane-sailing's collector normally has it; radioviz's
live view and the night shift's nightly sweep borrow it. A borrower takes a *lease*: it writes the lease file, stops
the collector, uses the device, then starts the collector again and removes the file.

    acquire:  refuse if a live holder or a release request exists ─► write the lease file ─► stop the collector
    (use the device)
    release:  start the collector if it was running ─► verify ─► remove the lease file

The lease file is the source of truth. It is written *before* the collector is stopped and removed only *after* the
collector is confirmed running again, so whatever crashes in between, the file is still there, and
`release_if_orphaned()` (run by a reaper timer, or by the next borrower) can finish the job.

Two rules make the hand-over safe:

* **Foreign leases.** A lease whose pid is alive belongs to someone else: refuse, never overwrite it.
* **Inherited duty.** A lease whose pid is dead is stale and may be replaced, but if it says the collector was
  running when the SDR was taken, the new lease says so too. Otherwise the duty to restart the collector would be
  lost in the hand-over: the dead holder stopped the collector, so it *looks* stopped-by-choice to the newcomer.

The file format is radioviz's (`radioviz/sdr/lease.py`), so radioviz, plane-sailing's midnight cycle script (which
greps `"pid": <n>`) and this module can all read each other's leases. Fields beyond radioviz's go inside `extra`
(`expires_at`, `command`, `inherited_from`): radioviz's reader rejects unknown top-level keys as a corrupt file.

This module uses only the standard library and never opens the device: the borrower does that.

    from cosmosdr.lease import ComposeCollector, Lease, LeaseConfig

    cfg = LeaseConfig()                                   # ~/.local/state/radioviz/, or $COSMOSDR_LEASE_DIR
    collector = ComposeCollector.for_plane_sailing()
    with Lease(cfg, "nightshift", "night_sweep", collector, max_s=20 * 60) as lease:
        ...                                               # open the device; poll lease.release_requested()

or, for a whole command: `python -m cosmosdr.lease run --holder nightshift --max-minutes 20 -- rtl_power ...`
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

# radioviz and plane-sailing look here today. Moving it means changing all three repos at once.
DEFAULT_STATE_DIR = Path.home() / ".local" / "state" / "radioviz"
OVERDUE_GRACE_S = 600  # a live holder this far past its expires_at may be terminated by the reaper
EX_TEMPFAIL = 75  # the SDR was busy, or the lease was cut short by a release request
EX_TIMEOUT = 124  # the command ran past --max-minutes


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------------------------- the lease file
@dataclass
class LeaseInfo:
    holder: str
    pid: int
    started_at: str
    reason: str
    # Was the collector running when the SDR was taken? If not (e.g. an experiment finished and stopped it),
    # releasing must not start it.
    collector_was_running: bool = True
    lease_id: int | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> LeaseInfo:
        """Lenient reader: unknown top-level keys are kept in `extra` rather than making the file "corrupt"."""
        known = {f.name for f in fields(cls)}
        info = cls(**{k: v for k, v in data.items() if k in known})
        for k, v in data.items():
            if k not in known:
                info.extra.setdefault(k, v)
        return info

    @property
    def expires_at(self) -> datetime | None:
        try:
            return datetime.fromisoformat(self.extra["expires_at"])
        except (KeyError, TypeError, ValueError):
            return None

    def age_s(self, now: datetime | None = None) -> float:
        try:
            started = datetime.fromisoformat(self.started_at)
        except ValueError:
            return float("inf")
        return ((now or utcnow()) - started).total_seconds()


CORRUPT = dict(holder="unknown", pid=-1, started_at="", reason="corrupt")


class LeaseFile:
    def __init__(self, path: Path):
        self.path = Path(path)

    def read(self) -> LeaseInfo | None:
        """None if there is no lease. An unreadable file reads as a stale lease whose collector *was* running: when
        in doubt, the safe thing is to give the SDR back."""
        try:
            data = json.loads(self.path.read_text())
            return LeaseInfo.from_dict(data)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            logger.error("corrupt lease file %s; treating as stale: %s", self.path, e)
            return LeaseInfo(**CORRUPT)

    def write(self, info: LeaseInfo) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        # plane-sailing's cycle script greps `"pid": <n>`; json.dumps' default separators produce exactly that.
        tmp.write_text(json.dumps(asdict(info)))
        os.replace(tmp, self.path)

    def remove(self) -> None:
        self.path.unlink(missing_ok=True)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers kill(0); treat it as dead.
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(") ", 1)[1][0] != "Z"
    except (FileNotFoundError, IndexError):
        return True


@dataclass
class LeaseConfig:
    state_dir: Path = field(default_factory=lambda: Path(os.environ.get("COSMOSDR_LEASE_DIR", DEFAULT_STATE_DIR)))
    # Append-only history of acquires and releases (the gap record for anyone analysing the collector's data).
    # None turns it off. radioviz also logs leases to Postgres; this file needs no database.
    log_name: str | None = "leases.jsonl"

    @property
    def lease_file(self) -> Path:
        return self.state_dir / "sdr.lease"

    @property
    def release_request_file(self) -> Path:
        return self.state_dir / "sdr.release-request"

    @property
    def lock_file(self) -> Path:
        return self.state_dir / "sdr.lease.lock"

    @property
    def log_file(self) -> Path | None:
        return self.state_dir / self.log_name if self.log_name else None


@contextmanager
def _locked(cfg: LeaseConfig) -> Iterator[None]:
    """Serialise check-then-write between cosmosdr users (radioviz doesn't take this lock yet)."""
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.lock_file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _log(cfg: LeaseConfig, event: str, info: LeaseInfo, **kw) -> None:
    if cfg.log_file is None:
        return
    rec = {"event": event, "at": utcnow().isoformat(), "holder": info.holder, "pid": info.pid,
           "reason": info.reason, "started_at": info.started_at, **kw}
    try:
        with open(cfg.log_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError as e:  # the history must never block the SDR
        logger.warning("could not append to the lease log: %s", e)


# --------------------------------------------------------------------------------------------- the collector
class Collector(Protocol):
    """Whatever normally holds the SDR and must be stopped while it is lent."""

    def is_running(self) -> bool: ...
    def stop(self) -> None: ...
    def start(self) -> None: ...


class NoCollector:
    """For machines where nothing else uses the SDR: nothing to stop or restore."""

    def is_running(self) -> bool:
        return False

    def stop(self) -> None:
        pass

    def start(self) -> None:
        pass


class ComposeCollector:
    """A docker compose service (plane-sailing's collector)."""

    def __init__(self, compose_cmd: list[str], service: str, container: str, run=subprocess.run):
        self.compose_cmd, self.service, self.container, self._run = compose_cmd, service, container, run

    @classmethod
    def for_plane_sailing(cls, infra: Path | None = None) -> ComposeCollector:
        infra = infra or Path.home() / "repo" / "plane-sailing" / "infra"
        cmd = ["docker", "compose", "-f", str(infra / "docker-compose.yaml"), "--project-directory", str(infra)]
        return cls(cmd, "collector", "plane-sailing-collector")

    def status(self) -> str:
        r = self._run(["docker", "inspect", "-f", "{{.State.Status}}", self.container],
                      capture_output=True, text=True, timeout=15, check=False)
        return r.stdout.strip() if r.returncode == 0 else "missing"

    def is_running(self) -> bool:
        return self.status() == "running"

    def _compose(self, *args: str) -> None:
        r = self._run([*self.compose_cmd, *args], capture_output=True, text=True, timeout=120, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"compose {' '.join(args)} failed: {r.stderr.strip()}")

    def stop(self) -> None:
        self._compose("stop", self.service)

    def start(self) -> None:
        # `up -d`, not `start`: picks up any .env change made while the SDR was lent.
        self._compose("up", "-d", self.service)


def restore_collector(collector: Collector, attempts: int = 4, verify_s: float = 20.0,
                      sleep: Callable[[float], None] | None = None) -> bool:
    """Bring the collector back, retrying with backoff. True once it reports running."""
    sleep = sleep or time.sleep  # looked up per call, so a test can patch time.sleep
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            collector.start()
            deadline = time.monotonic() + verify_s
            while True:
                if collector.is_running():
                    logger.info("collector restored (attempt %d)", attempt)
                    return True
                if time.monotonic() >= deadline:
                    break
                sleep(1)
            logger.warning("collector not running after start (attempt %d)", attempt)
        except Exception as e:  # noqa: BLE001
            logger.error("collector start failed (attempt %d): %s", attempt, e)
        if attempt < attempts:
            sleep(delay)
            delay *= 2
    return False


# --------------------------------------------------------------------------------------------- borrowing
class SdrBusy(Exception):
    """Someone else holds the SDR, or plane-sailing has asked for it back."""


class Lease:
    """Borrow the SDR. Use as a context manager, or call acquire() / release() yourself."""

    def __init__(self, cfg: LeaseConfig, holder: str, reason: str, collector: Collector,
                 max_s: float | None = None, extra: dict | None = None, restore: Callable[..., bool] = restore_collector):
        self.cfg, self.holder, self.reason, self.collector = cfg, holder, reason, collector
        self.max_s, self.extra, self.restore = max_s, dict(extra or {}), restore
        self.file = LeaseFile(cfg.lease_file)
        self.info: LeaseInfo | None = None

    def acquire(self) -> LeaseInfo:
        with _locked(self.cfg):
            if self.cfg.release_request_file.exists():
                raise SdrBusy("a release request is pending (plane-sailing's midnight flip)")
            existing = self.file.read()
            if existing is not None and pid_alive(existing.pid):
                raise SdrBusy(f"the SDR is leased by {existing.holder} (pid {existing.pid}, {existing.reason})")
            running = self.collector.is_running()
            extra = dict(self.extra)
            if self.max_s is not None:
                extra["expires_at"] = (utcnow() + timedelta(seconds=self.max_s)).isoformat()
            inherited = existing is not None and existing.collector_was_running
            if existing is not None:
                logger.warning("replacing a stale lease (%s, pid %d)", existing.holder, existing.pid)
                extra["inherited_from"] = {"holder": existing.holder, "pid": existing.pid,
                                           "started_at": existing.started_at}
            # Lease file first: from this instant, crash recovery knows the collector is ours to restore.
            self.info = LeaseInfo(self.holder, os.getpid(), utcnow().isoformat(), self.reason,
                                  collector_was_running=running or inherited, extra=extra)
            self.file.write(self.info)
        _log(self.cfg, "acquire", self.info, collector_was_running=self.info.collector_was_running)
        if running:
            try:
                self.collector.stop()
            except Exception:
                # Don't leave a lease naming a live process that never got the SDR: give it back now.
                logger.exception("could not stop the collector; releasing")
                self.release("error")
                raise
        logger.info("lease acquired by %s (%s)", self.holder, self.reason)
        return self.info

    def release_requested(self) -> bool:
        """plane-sailing's midnight cycle drops this file when it needs the SDR back. Honour it within seconds."""
        return self.cfg.release_request_file.exists()

    def release(self, ended_by: str = "done") -> bool:
        """Give the SDR back. False if the collector didn't come back: the lease file is kept so the reaper retries."""
        if self.info is None:
            return True
        ok = True
        if self.info.collector_was_running and not self.collector.is_running():
            ok = self.restore(self.collector)
        _log(self.cfg, "release", self.info, ended_by=ended_by, collector_ok=ok)
        if ok:
            self.file.remove()
            self.info = None
        else:
            logger.error("COLLECTOR DID NOT COME BACK; lease file kept so the reaper retries")
        return ok

    def __enter__(self) -> Lease:
        self.acquire()
        return self

    def __exit__(self, exc_type, *_) -> bool:
        self.release("error" if exc_type else "done")
        return False


def release_if_orphaned(cfg: LeaseConfig, collector: Collector, ended_by: str = "reaper",
                        terminate_overdue: bool = False, now: datetime | None = None,
                        restore: Callable[..., bool] = restore_collector) -> str:
    """Idempotently give the SDR back if the lease has no live holder. Safe to run at any time (a reaper timer).

    With terminate_overdue, a live holder more than OVERDUE_GRACE_S past its `expires_at` is sent SIGTERM (then
    SIGKILL): a hung borrower would otherwise keep the collector off indefinitely. A lease with no `expires_at`
    (radioviz's own) is never terminated here; radioviz enforces its own cap.
    """
    lf = LeaseFile(cfg.lease_file)
    info = lf.read()
    if info is None:
        return "no-lease"
    if pid_alive(info.pid) and info.pid != os.getpid():
        exp = info.expires_at
        overdue = exp is not None and (now or utcnow()) > exp + timedelta(seconds=OVERDUE_GRACE_S)
        if not (terminate_overdue and overdue):
            return "held-by-live-process"
        logger.error("lease overdue; terminating holder pid %d", info.pid)
        _terminate(info.pid)
    if info.collector_was_running and not collector.is_running() and not restore(collector):
        logger.error("COLLECTOR DID NOT COME BACK; lease file kept so the next run retries")
        _log(cfg, "release", info, ended_by=ended_by, collector_ok=False)
        return "collector-restore-failed"
    _log(cfg, "release", info, ended_by=ended_by, collector_ok=True)
    lf.remove()
    return "released"


def _terminate(pid: int, grace_s: float = 30.0) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# --------------------------------------------------------------------------------------------- one command
def _die_with_parent() -> None:  # runs in the child between fork and exec
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except OSError:
        pass


def _stop(child: subprocess.Popen, grace_s: float) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def run_leased(cfg: LeaseConfig, cmd: list[str], holder: str, reason: str, max_s: float, collector: Collector,
               poll_s: float = 1.0, grace_s: float = 10.0, restore: Callable[..., bool] = restore_collector) -> int:
    """Hold the SDR for exactly as long as `cmd` runs (it opens the device itself).

    Returns the command's status; EX_TEMPFAIL if the SDR was busy or a release request cut it short; EX_TIMEOUT if it
    ran past max_s; 1 if the collector could not be restored (the lease file is kept for the reaper). The command
    gets PR_SET_PDEATHSIG, so it can't outlive this process holding the device; if this process is killed outright,
    the lease names a dead pid and release_if_orphaned() recovers.
    """
    lease = Lease(cfg, holder, reason, collector, max_s=max_s, extra={"command": cmd[0]}, restore=restore)
    try:
        lease.acquire()
    except SdrBusy as e:
        logger.warning("not leasing: %s", e)
        return EX_TEMPFAIL
    status, ended_by = 0, "done"
    try:
        child = subprocess.Popen(cmd, preexec_fn=_die_with_parent)  # noqa: PLW1509 - prctl only
        deadline = time.monotonic() + max_s
        try:
            while (rc := child.poll()) is None:
                if lease.release_requested():
                    logger.warning("release requested; stopping the command")
                    status, ended_by = EX_TEMPFAIL, "release_request"
                    break
                if time.monotonic() >= deadline:
                    logger.warning("lease time is up; stopping the command")
                    status, ended_by = EX_TIMEOUT, "cap"
                    break
                time.sleep(poll_s)
            else:
                status = rc
        finally:
            _stop(child, grace_s)
    except BaseException:
        ended_by = "error"
        raise
    finally:
        if not lease.release(ended_by):
            status = 1
    return status


# --------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cosmosdr-lease", description="Share the RTL-SDR: the lease.")
    p.add_argument("--state-dir", type=Path, help=f"default $COSMOSDR_LEASE_DIR or {DEFAULT_STATE_DIR}")
    p.add_argument("--no-collector", action="store_true", help="nothing else uses the SDR on this machine")
    sub = p.add_subparsers(dest="cmd_name", required=True)
    sub.add_parser("status", help="print the current lease as JSON (or null)")
    r = sub.add_parser("release", help="give the SDR back if the lease's holder is dead (a reaper)")
    r.add_argument("--reap", action="store_true", help="also terminate a live holder long past its expires_at")
    run = sub.add_parser("run", help="hold the SDR while one command runs")
    run.add_argument("--holder", required=True)
    run.add_argument("--reason", default="night_sweep")
    run.add_argument("--max-minutes", type=float, required=True)
    run.add_argument("cmd", nargs=argparse.REMAINDER, help="-- then the command")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s cosmosdr-lease %(levelname)s %(message)s")

    cfg = LeaseConfig(state_dir=a.state_dir) if a.state_dir else LeaseConfig()
    collector: Collector = NoCollector() if a.no_collector else ComposeCollector.for_plane_sailing()
    if a.cmd_name == "status":
        info = LeaseFile(cfg.lease_file).read()
        print(json.dumps(None if info is None else {**asdict(info), "alive": pid_alive(info.pid)}))
        return 0
    if a.cmd_name == "release":
        outcome = release_if_orphaned(cfg, collector, terminate_overdue=a.reap)
        print(outcome)
        return 1 if outcome == "collector-restore-failed" else 0
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        p.error("give the command after --")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(EX_TEMPFAIL))  # systemd stop → the finally blocks release
    return run_leased(cfg, cmd, a.holder, a.reason, a.max_minutes * 60, collector)


if __name__ == "__main__":
    sys.exit(main())
