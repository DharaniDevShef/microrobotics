"""
Sim Executor - runs many roblet_simulator.py evaluations in parallel OS
processes, capped at `max_workers` concurrent processes at a time so a
generation bigger than the CPU's core count doesn't oversubscribe it.
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

# --sweep_b makes each subprocess run len(B_SWEEP_VALUES) candidate passes
# plus one final winner re-run, so the timeout must cover that whole multiplier.
_SWEEP_RUNS_PER_JOB = len(rs.B_SWEEP_VALUES) + 1
_PER_RUN_OVERHEAD_S = 8.0  # process startup, model load, settle phase, I/O

# Pin each subprocess to a single BLAS thread so max_workers processes don't
# oversubscribe the CPU (NumPy/MuJoCo's BLAS backend defaults to one thread/core).
_SUBPROCESS_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def run_batch(jobs, max_workers=None, max_sim_time=7.0):
    """jobs: list of (xml_path, stats_path) pairs. Runs one
    `roblet_simulator.py --headless` OS process per job, in batches of at
    most `max_workers` concurrent processes."""
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
