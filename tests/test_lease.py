"""The SDR lease (cosmosdr.lease): no device and no docker needed; a fake collector and temp state dirs."""

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest

from cosmosdr.lease import (
    EX_TEMPFAIL,
    EX_TIMEOUT,
    ComposeCollector,
    Lease,
    LeaseConfig,
    LeaseFile,
    LeaseInfo,
    SdrBusy,
    main,
    pid_alive,
    release_if_orphaned,
    run_leased,
    utcnow,
)

DEAD_PID = 2**22 + 12345  # above Linux's maximum pid_max, so never a live process
# The fields radioviz's LeaseInfo accepts (radioviz/backend/radioviz/sdr/lease.py). It reads the file with
# LeaseInfo(**data), so any other top-level key makes radioviz treat the lease as corrupt and reap it.
RADIOVIZ_FIELDS = {"holder", "pid", "started_at", "reason", "collector_was_running", "lease_id", "extra"}


class FakeCollector:
    def __init__(self, running=True, fail_starts=0):
        self.running, self.fail_starts, self.calls = running, fail_starts, []

    def is_running(self):
        return self.running

    def stop(self):
        self.calls.append("stop")
        self.running = False

    def start(self):
        self.calls.append("start")
        if self.fail_starts > 0:
            self.fail_starts -= 1
            raise RuntimeError("compose up failed")
        self.running = True


def no_wait_restore(collector):
    from cosmosdr.lease import restore_collector

    return restore_collector(collector, attempts=2, verify_s=0, sleep=lambda s: None)


@pytest.fixture
def live_pid():
    """A disposable live process to stand in for another lease holder. Never use a real one such as os.getppid():
    a reaper bug would signal it (a mutant of release_if_orphaned once sent SIGTERM to the shell running pytest)."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    yield p.pid
    p.kill()
    p.wait()


@pytest.fixture
def cfg(tmp_path):
    return LeaseConfig(state_dir=tmp_path / "state")


def write_lease(cfg, pid, **kw):
    info = LeaseInfo(**{"holder": "radioviz", "pid": pid, "started_at": utcnow().isoformat(),
                        "reason": "interactive", **kw})
    LeaseFile(cfg.lease_file).write(info)
    return info


def log_events(cfg):
    return [json.loads(line) for line in cfg.log_file.read_text().splitlines()]


# --------------------------------------------------------------------------------------------- file format
def test_file_is_readable_by_radioviz_and_plane_sailing(cfg):
    lease = Lease(cfg, "nightshift", "night_sweep", FakeCollector(), max_s=600, extra={"command": "rtl_power"})
    lease.acquire()
    raw = cfg.lease_file.read_text()
    data = json.loads(raw)
    assert set(data) <= RADIOVIZ_FIELDS  # radioviz's strict reader accepts it
    assert "expires_at" in data["extra"] and data["extra"]["command"] == "rtl_power"
    # plane-sailing's cosmos_n_cycle.sh: grep -o '"pid": *[0-9]*' | grep -o '[0-9]*$'
    assert re.search(r'"pid": *([0-9]*)', raw).group(1) == str(os.getpid())
    lease.release()


def test_unknown_top_level_keys_are_kept_not_corrupt(cfg):
    cfg.state_dir.mkdir(parents=True)
    cfg.lease_file.write_text(json.dumps({"holder": "nightshift", "pid": 5, "started_at": "", "reason": "x",
                                          "expires_at": "2026-10-22T02:00:00+00:00"}))
    info = LeaseFile(cfg.lease_file).read()
    assert info.holder == "nightshift" and info.expires_at is not None


@pytest.mark.parametrize("content", ["{not json", "[]", '{"holder": "x"}'])
def test_unreadable_file_reads_as_stale_with_the_collector_to_restore(cfg, content):
    cfg.state_dir.mkdir(parents=True)
    cfg.lease_file.write_text(content)
    info = LeaseFile(cfg.lease_file).read()
    assert info.reason == "corrupt" and not pid_alive(info.pid) and info.collector_was_running


# --------------------------------------------------------------------------------------------- acquire / release
def test_lease_stops_and_restores_a_running_collector(cfg):
    col = FakeCollector(running=True)
    with Lease(cfg, "nightshift", "night_sweep", col, restore=no_wait_restore):
        assert not col.running and cfg.lease_file.exists()
    assert col.calls == ["stop", "start"] and col.running
    assert not cfg.lease_file.exists()
    assert [e["event"] for e in log_events(cfg)] == ["acquire", "release"]
    assert log_events(cfg)[1]["ended_by"] == "done" and log_events(cfg)[1]["collector_ok"]


def test_a_stopped_collector_stays_stopped(cfg):
    col = FakeCollector(running=False)  # e.g. the COSMOS_N experiment finished and stopped it
    with Lease(cfg, "nightshift", "night_sweep", col):
        pass
    assert col.calls == [] and not cfg.lease_file.exists()


def test_refuses_a_live_foreign_lease_and_leaves_it_alone(cfg, live_pid):
    write_lease(cfg, live_pid)  # radioviz's live view, held by a live process that isn't us
    before = cfg.lease_file.read_text()
    col = FakeCollector()
    with pytest.raises(SdrBusy, match="radioviz"):
        Lease(cfg, "nightshift", "night_sweep", col).acquire()
    assert col.calls == [] and cfg.lease_file.read_text() == before


def test_refuses_a_second_lease_in_the_same_process(cfg):
    col = FakeCollector()
    with Lease(cfg, "a", "test", col, restore=no_wait_restore):
        with pytest.raises(SdrBusy):
            Lease(cfg, "b", "test", col).acquire()


def test_refuses_while_a_release_request_is_pending(cfg):
    cfg.state_dir.mkdir(parents=True)
    cfg.release_request_file.write_text('{"requested_by": "cosmos_n_cycle"}')
    col = FakeCollector()
    with pytest.raises(SdrBusy, match="release request"):
        Lease(cfg, "nightshift", "night_sweep", col).acquire()
    assert col.calls == [] and not cfg.lease_file.exists()


def test_replacing_a_stale_lease_inherits_the_duty_to_restore(cfg):
    """A holder took the SDR from a running collector and died. The collector is stopped *because of it*, so the
    next holder must restart it on release, even though it found it stopped. (radioviz 2026-10-22 loses this.)"""
    write_lease(cfg, DEAD_PID, holder="nightshift", reason="night_sweep", collector_was_running=True)
    col = FakeCollector(running=False)
    lease = Lease(cfg, "radioviz", "interactive", col, restore=no_wait_restore)
    info = lease.acquire()
    assert info.collector_was_running and info.extra["inherited_from"]["pid"] == DEAD_PID
    assert lease.release()
    assert col.calls == ["start"] and col.running and not cfg.lease_file.exists()


def test_replacing_a_stale_lease_that_found_the_collector_stopped_does_not_start_it(cfg):
    write_lease(cfg, DEAD_PID, collector_was_running=False)
    col = FakeCollector(running=False)
    with Lease(cfg, "nightshift", "night_sweep", col):
        pass
    assert col.calls == []


def test_collector_that_will_not_come_back_keeps_the_lease_file(cfg):
    col = FakeCollector(running=True, fail_starts=5)
    lease = Lease(cfg, "nightshift", "night_sweep", col, restore=no_wait_restore)
    lease.acquire()
    assert not lease.release()
    assert cfg.lease_file.exists()  # the reaper (or the next holder) will retry
    col.fail_starts = 0
    assert release_if_orphaned(cfg, col, restore=no_wait_restore) == "released"
    assert col.running and not cfg.lease_file.exists()


def test_collector_that_will_not_stop_releases_the_lease_at_once(cfg):
    class Stubborn(FakeCollector):
        def stop(self):
            self.calls.append("stop")
            raise RuntimeError("compose stop failed")

    col = Stubborn(running=True)
    with pytest.raises(RuntimeError, match="compose stop failed"):
        Lease(cfg, "nightshift", "night_sweep", col).acquire()
    assert col.calls == ["stop"] and col.running  # still running: nothing to restart
    assert not cfg.lease_file.exists()  # and no lease left naming a live process
    assert [e["event"] for e in log_events(cfg)] == ["acquire", "release"]
    assert log_events(cfg)[1]["ended_by"] == "error"


def test_concurrent_acquires_one_winner(cfg):
    """Four processes race for the SDR: the lock makes exactly one of them win. Reading the lease file is slowed
    down to 0.2 s so that, without the lock, all four would read "no lease" before any of them wrote one."""
    go = cfg.state_dir.parent / "go"
    code = (
        "import sys, time, pathlib; from cosmosdr import lease as L\n"
        "from cosmosdr.lease import Lease, LeaseConfig, NoCollector, SdrBusy\n"
        "read = L.LeaseFile.read; L.LeaseFile.read = lambda self: (read(self), time.sleep(0.2))[0]\n"
        f"cfg = LeaseConfig(state_dir=pathlib.Path({str(cfg.state_dir)!r}))\n"
        f"while not pathlib.Path({str(go)!r}).exists(): time.sleep(0.005)\n"  # start together, after the imports
        "try:\n    Lease(cfg, 'p', 'test', NoCollector()).acquire(); print('won', flush=True); sys.stdin.read()\n"
        "except SdrBusy:\n    print('busy', flush=True)\n"
    )
    procs = [subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              text=True) for _ in range(4)]
    go.write_text("")
    outs = sorted(p.stdout.readline().strip() for p in procs)  # the winner holds its lease until stdin closes
    for p in procs:
        p.stdin.close()
        p.wait(timeout=30)
    assert outs == ["busy", "busy", "busy", "won"]


# --------------------------------------------------------------------------------------------- the reaper
def test_reaper_outcomes(cfg, live_pid):
    col = FakeCollector(running=False)
    assert release_if_orphaned(cfg, col) == "no-lease"
    write_lease(cfg, live_pid)
    assert release_if_orphaned(cfg, col) == "held-by-live-process"
    write_lease(cfg, DEAD_PID, collector_was_running=True)
    assert release_if_orphaned(cfg, col, restore=no_wait_restore) == "released"
    assert col.running and not cfg.lease_file.exists()
    assert log_events(cfg)[-1]["ended_by"] == "reaper"


def test_reaper_terminates_a_holder_long_past_expires_at(cfg):
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        past = (utcnow() - timedelta(minutes=30)).isoformat()
        write_lease(cfg, holder.pid, holder="nightshift", reason="night_sweep", extra={"expires_at": past})
        col = FakeCollector(running=False)
        assert release_if_orphaned(cfg, col, restore=no_wait_restore) == "held-by-live-process"  # no --reap
        assert release_if_orphaned(cfg, col, terminate_overdue=True, restore=no_wait_restore) == "released"
        assert holder.wait(timeout=10) != 0 and col.running
    finally:
        holder.kill()


def test_reaper_never_terminates_a_lease_without_expires_at(cfg, live_pid):
    write_lease(cfg, live_pid, started_at=(utcnow() - timedelta(days=2)).isoformat())  # radioviz enforces its own cap
    assert release_if_orphaned(cfg, FakeCollector(), terminate_overdue=True) == "held-by-live-process"


# --------------------------------------------------------------------------------------------- one command
def _py(code):
    return [sys.executable, "-c", code]


def test_run_holds_the_lease_while_the_command_runs(cfg, tmp_path):
    seen = tmp_path / "seen.json"
    col = FakeCollector(running=True)
    cmd = _py(f"import shutil; shutil.copy({str(cfg.lease_file)!r}, {str(seen)!r}); raise SystemExit(3)")
    rc = run_leased(cfg, cmd, "nightshift", "night_sweep", 30, col, poll_s=0.05, restore=no_wait_restore)
    assert rc == 3  # the command's own status
    held = json.loads(seen.read_text())
    assert held["holder"] == "nightshift" and held["pid"] == os.getpid() and "expires_at" in held["extra"]
    assert col.calls == ["stop", "start"] and not cfg.lease_file.exists()


def test_run_refuses_when_busy(cfg, live_pid):
    write_lease(cfg, live_pid)
    col = FakeCollector()
    assert run_leased(cfg, _py("pass"), "nightshift", "night_sweep", 30, col) == EX_TEMPFAIL
    assert col.calls == []


def test_run_stops_at_max_time(cfg):
    col = FakeCollector(running=True)
    t0 = time.monotonic()
    rc = run_leased(cfg, _py("import time; time.sleep(30)"), "nightshift", "night_sweep", 0.5, col,
                    poll_s=0.05, grace_s=2, restore=no_wait_restore)
    assert rc == EX_TIMEOUT and time.monotonic() - t0 < 10
    assert col.running and not cfg.lease_file.exists()
    assert log_events(cfg)[-1]["ended_by"] == "cap"


def test_release_request_cuts_the_command_short(cfg):
    col = FakeCollector(running=True)

    def request():
        time.sleep(0.5)
        cfg.release_request_file.write_text('{"requested_by": "cosmos_n_cycle"}')

    threading.Thread(target=request, daemon=True).start()
    t0 = time.monotonic()
    rc = run_leased(cfg, _py("import time; time.sleep(30)"), "nightshift", "night_sweep", 60, col,
                    poll_s=0.05, grace_s=2, restore=no_wait_restore)
    assert rc == EX_TEMPFAIL and time.monotonic() - t0 < 10
    assert col.running and log_events(cfg)[-1]["ended_by"] == "release_request"


def test_wrapper_killed_outright_takes_the_command_with_it_and_the_reaper_recovers(cfg, tmp_path):
    script = tmp_path / "wrapper.py"
    script.write_text(f"""
import sys
from pathlib import Path
from cosmosdr.lease import LeaseConfig, NoCollector, run_leased
cfg = LeaseConfig(state_dir=Path({str(cfg.state_dir)!r}))
run_leased(cfg, [sys.executable, "-c", "import time; print('child up', flush=True); time.sleep(60)"],
           "nightshift", "night_sweep", 60, NoCollector(), poll_s=0.05)
""")
    p = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        if line.strip() == "child up":
            break
    children = [int(x) for x in subprocess.run(["pgrep", "-P", str(p.pid)], capture_output=True,
                                               text=True).stdout.split()]
    assert children
    p.kill()
    p.wait()
    time.sleep(0.5)
    assert not any(pid_alive(c) for c in children)  # PR_SET_PDEATHSIG took the command down too
    info = LeaseFile(cfg.lease_file).read()
    assert info.holder == "nightshift" and not pid_alive(info.pid)
    info.collector_was_running = True  # as if the collector had been running when the wrapper took the SDR
    LeaseFile(cfg.lease_file).write(info)
    col = FakeCollector(running=False)
    assert release_if_orphaned(cfg, col, restore=no_wait_restore) == "released"
    assert col.running and not cfg.lease_file.exists()


# --------------------------------------------------------------------------------------------- CLI and compose
def test_cli_status_and_release(cfg, capsys, live_pid):
    assert main(["--state-dir", str(cfg.state_dir), "--no-collector", "status"]) == 0
    assert capsys.readouterr().out.strip() == "null"
    write_lease(cfg, live_pid)
    main(["--state-dir", str(cfg.state_dir), "--no-collector", "status"])
    out = json.loads(capsys.readouterr().out)
    assert out["holder"] == "radioviz" and out["alive"] is True
    assert main(["--state-dir", str(cfg.state_dir), "--no-collector", "release"]) == 0
    assert capsys.readouterr().out.strip() == "held-by-live-process"


def test_cli_run_needs_a_command(cfg, capsys):
    with pytest.raises(SystemExit) as e:
        main(["--state-dir", str(cfg.state_dir), "run", "--holder", "nightshift", "--max-minutes", "1"])
    assert e.value.code == 2 and "give the command after --" in capsys.readouterr().err


def test_compose_collector_uses_up_d_and_inspect():
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="running\n", stderr="")

    c = ComposeCollector.for_plane_sailing(Path("/x/infra"))
    c._run = fake_run
    assert c.is_running()
    c.stop()
    c.start()
    assert calls[0] == ["docker", "inspect", "-f", "{{.State.Status}}", "plane-sailing-collector"]
    assert calls[1][-2:] == ["stop", "collector"] and calls[2][-3:] == ["up", "-d", "collector"]
    assert calls[1][:5] == ["docker", "compose", "-f", "/x/infra/docker-compose.yaml", "--project-directory"]
