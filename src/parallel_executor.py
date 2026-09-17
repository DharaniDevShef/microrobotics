"""
Parallel Executor - launches four fixed roblet_simulator.py --headless runs
(assembly0..3.xml) as concurrent OS processes and waits for them all to
finish. Superseded by sim_executor.py for the real evolutionary loop; kept
as a quick standalone throughput check.
"""

import os
import subprocess
import time

# Direct path to your venv's Python binary
VENV_PYTHON = r"d:\microrobotics\.venv\Scripts\python.exe"

# Each simulation is a single-threaded physics loop, but NumPy/MuJoCo's
# BLAS backend still defaults to spawning one thread per CPU core. With
# N processes launched at once, that's N x core_count threads fighting
# over core_count cores - the actual reason a 7s (simulated) run was
# taking ~16s of wall-clock time here. Pinning each subprocess to a
# single BLAS thread lets them run genuinely in parallel instead of
# thrashing each other.
_SUBPROCESS_ENV = {
    **os.environ,
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

commands = [
    [
        VENV_PYTHON,
        "roblet_simulator.py",
        "--m", "../models/assembly0.xml",
        "--o", "../output/simulation_stats0.json",
        "--headless",
        "--capture_img",
        "--sweep_b",
        # "--capture_gif"
    ],
    [
        VENV_PYTHON,
        "roblet_simulator.py",
        "--m", "../models/assembly1.xml",
        "--o", "../output/simulation_stats1.json",
        "--headless",
        "--capture_img",
        "--sweep_b",
        # "--capture_gif"
    ],
    [
        VENV_PYTHON,
        "roblet_simulator.py",
        "--m", "../models/assembly2.xml",
        "--o", "../output/simulation_stats2.json",
        "--headless",
        "--capture_img",
        "--sweep_b",
        # "--capture_gif"
    ],
    [
        VENV_PYTHON,
        "roblet_simulator.py",
        "--m", "../models/assembly3.xml",
        "--o", "../output/simulation_stats3.json",
        "--headless",
        "--capture_img",
        "--sweep_b",
        # "--capture_gif"
    ],
]

processes = []

start_time = time.time()
print(f"Starting {len(commands)} simulations in parallel...")

# Start all simulations without blocking
for cmd in commands:
    p = subprocess.Popen(cmd, env=_SUBPROCESS_ENV)
    processes.append(p)

# Wait for every simulation to finish naturally
for p in processes:
    return_code = p.wait()

    if return_code == 0:
        print(f"Simulation finished successfully (PID {p.pid})")
    else:
        print(f"Simulation failed (PID {p.pid}, return code {return_code})")

elapsed = time.time() - start_time
print(f"All simulations finished in {elapsed:.2f} s.")

