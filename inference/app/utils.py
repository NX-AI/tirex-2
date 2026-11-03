# Copyright (c) NXAI GmbH.
# Licensed under the Apache License, Version 2.0; see LICENSE for details.

from pathlib import Path


def cpu_quota(cgroup: Path = Path("/sys/fs/cgroup")) -> int | None:
    """Number of CPUs under a cgroup v2 CPU limit (e.g. `docker run --cpus`), or None without a limit."""
    try:
        quota, period = (cgroup / "cpu.max").read_text().split()
    except (OSError, ValueError):
        return None
    if quota == "max":
        return None
    return max(1, int(quota) // int(period))
