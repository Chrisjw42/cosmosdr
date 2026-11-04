# cosmosdr
A utility for signal acquisition and processing, from an RTL SDR device.

Utilised by https://github.com/Chrisjw42/plane-sailing for processing aircrafy signals.

## Sharing the SDR: `cosmosdr.lease`

One RTL-SDR, several programs: plane-sailing's collector normally has it; radioviz's live view and the night
shift's nightly sweep borrow it. A borrower takes a *lease*: write the lease file, stop the collector, use the
device, start the collector again, remove the file. The file (`~/.local/state/radioviz/sdr.lease`, or
`$COSMOSDR_LEASE_DIR`) uses radioviz's format, so radioviz and plane-sailing's midnight cycle script read it as before.

```python
from cosmosdr.lease import ComposeCollector, Lease, LeaseConfig

with Lease(LeaseConfig(), "nightshift", "night_sweep", ComposeCollector.for_plane_sailing(), max_s=1200) as lease:
    ...  # open the device; stop early if lease.release_requested()
```

```bash
cosmosdr-lease status                       # the current lease as JSON, or null
cosmosdr-lease release --reap               # a reaper: give the SDR back if the holder died (or hung long past expires_at)
cosmosdr-lease run --holder nightshift --max-minutes 20 -- rtl_power -f 24M:1766M:100k -g 30 -i 10m -1 sweep.csv
```

It refuses a lease that a live process holds, and refuses while plane-sailing has asked for the SDR back. Replacing a
dead holder's lease keeps that lease's duty to restart the collector. Acquires and releases are appended to
`leases.jsonl` next to the lease file, so gaps in the collector's data can be explained without a database.
