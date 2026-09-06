"""
Sim Executor - runs many roblet_simulator.py evaluations in parallel OS
processes.

Generalizes parallel_executor.py's subprocess.Popen pattern (launch N
`roblet_simulator.py --headless` processes, one per model, then wait for
all of them) from its fixed list of 4 hardcoded models to an arbitrary
batch of jobs, capped at `max_workers` concurrent processes at a time so
a generation bigger than the CPU's core count doesn't oversubscribe it.
"""

import logging
import os
import subprocess
import sys
import json

import roblet_simulator as rs

logger = logging.getLogger(__name__)

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROBLET_SIMULATOR = os.path.join(_SRC_DIR, "roblet_simulator.py")

# --sweep_b (always passed below) makes each subprocess run
# len(B_SWEEP_VALUES) candidate passes PLUS one final "winner" re-run -
# up to max_sim_time simulated seconds each - not just one run's worth.
# The subprocess timeout has to cover that whole multiplier or individuals
# get killed mid-sweep before ever finishing (which is exactly what a
# flat "+20s" buffer was doing: sized for 1 run, actually needed for up
# to 5). Read from roblet_simulator.py itself so this stays in sync if
# B_SWEEP_VALUES ever changes.
_SWEEP_RUNS_PER_JOB = len(rs.B_SWEEP_VALUES) + 1
# Per-run overhead beyond raw sim-time: process startup, model
# load/compile, the pre-gait settle phase, screenshot/JSON I/O.
_PER_RUN_OVERHEAD_S = 8.0

# Each simulation is a single-threaded physics loop, but NumPy/MuJoCo's
# BLAS backend still defaults to spawning one thread per CPU core. With
# max_workers processes launched at once, that's max_workers x core_count
# threads fighting over core_count cores, which was inflating wall-clock
# time well past the sim-time budget. Pinning each subprocess to a single
# BLAS thread lets them run genuinely in parallel instead of thrashing.
_SUBPROCESS_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def run_batch(jobs, max_workers=None, max_sim_time=7.0):
    """jobs: list of (xml_path, stats_path) pairs.

    Runs one `roblet_simulator.py --headless` OS process per job - its
    module-level globals (parent_body_magnet_map, torque_history, ...)
    are cleared at the top of run_headless(), so the same script is safe
    to launch repeatedly - in batches of at most `max_workers` concurrent
    processes.
    """
    if not jobs:
        logger.info("No simulation jobs to run.")
        return
    max_workers = max_workers or os.cpu_count() or 4
    process_timeout = _SWEEP_RUNS_PER_JOB * (max_sim_time + _PER_RUN_OVERHEAD_S)
    logger.info(
        "Running %d simulation jobs in batches of %d workers (per-job timeout=%.1fs, "
        "covering %d sweep runs of up to %.1fs each)",
        len(jobs), max_workers, process_timeout, _SWEEP_RUNS_PER_JOB, max_sim_time,
    )

    for batch_start in range(0, len(jobs), max_workers):
        batch = jobs[batch_start:batch_start + max_workers]
        logger.info("Starting batch %d: %d processes", batch_start // max_workers + 1, len(batch))
        processes = []
        for xml_path, stats_path in batch:
            log_path = os.path.splitext(stats_path)[0] + ".log"
            cmd = [
                sys.executable, _ROBLET_SIMULATOR,
                "--m", xml_path, "--o", stats_path,
                "--headless", "--sweep_b", "--max_sim_time", str(max_sim_time),
                #"--capture_img", 
                #"--log-file", log_path,
            ]
            p = subprocess.Popen(cmd, cwd=_SRC_DIR, env=_SUBPROCESS_ENV)
            processes.append((p, xml_path, stats_path))

        # Wait for each process with a safe timeout and handle failures
        for p, xml_path, stats_path in processes:
            try:
                p.wait(timeout=process_timeout)
            except subprocess.TimeoutExpired:
                logger.error("Simulation timed out for %s, killing process", xml_path)
                try:
                    p.kill()
                except Exception:
                    pass
                p.wait()
                _write_failed_stats(stats_path)
                continue

            if p.returncode is None or p.returncode != 0:
                logger.error("Simulation failed (returncode=%s) for %s", str(p.returncode), xml_path)
                _write_failed_stats(stats_path)
            else:
                logger.info("Simulation finished successfully (PID %d) -> %s", p.pid, stats_path)


def _write_failed_stats(stats_path: str):
    """Write a minimal failed stats JSON to the given path so downstream
    consumers don't crash if a subprocess fails or times out."""
    failed = {
        "success": 0,
        "physics_ok": 0,
        "is_stable": 0,
        "average_velocity_mmps": 0.0,
        "total_distance_mm": 0.0,
    }
    try:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=4)
    except Exception:
        logger.exception("Could not write failed stats to %s", stats_path)
