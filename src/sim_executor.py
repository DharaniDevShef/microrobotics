"""
Sim Executor - runs many roblet_simulator.py evaluations in parallel OS
processes.

Generalizes parallel_executor.py's subprocess.Popen pattern (launch N
`roblet_simulator.py --headless` processes, one per model, then wait for
all of them) from its fixed list of 4 hardcoded models to an arbitrary
batch of jobs, capped at `max_workers` concurrent processes at a time so
a generation bigger than the CPU's core count doesn't oversubscribe it.
"""

import os
import subprocess
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROBLET_SIMULATOR = os.path.join(_SRC_DIR, "roblet_simulator.py")


def run_batch(jobs, max_workers=None, max_sim_time=7.0):
    """jobs: list of (xml_path, stats_path) pairs.

    Runs one `roblet_simulator.py --headless` OS process per job - its
    module-level globals (parent_body_magnet_map, torque_history, ...)
    are cleared at the top of run_headless(), so the same script is safe
    to launch repeatedly - in batches of at most `max_workers` concurrent
    processes.
    """
    if not jobs:
        return
    max_workers = max_workers or os.cpu_count() or 4

    for batch_start in range(0, len(jobs), max_workers):
        batch = jobs[batch_start:batch_start + max_workers]
        processes = []
        for xml_path, stats_path in batch:
            cmd = [
                sys.executable, _ROBLET_SIMULATOR,
                "--m", xml_path, "--o", stats_path,
                "--headless", "--sweep_b", "--max_sim_time", str(max_sim_time),
                "--capture_img"
            ]
            processes.append(subprocess.Popen(cmd, cwd=_SRC_DIR))
        for p in processes:
            p.wait()
