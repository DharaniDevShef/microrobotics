import subprocess
import time

# Direct path to your venv's Python binary
VENV_PYTHON = r"d:\microrobotics\.venv\Scripts\python.exe"

commands = [
    [
        VENV_PYTHON,
        "roblet_simulator.py",
        "--m", "../models/assembly0.xml",
        "--o", "../output/simulation_stats0.json",
        "--headless",
        "--capture-media"
    ],
    [
        VENV_PYTHON,
        "roblet_simulator.py",
        "--m", "../models/assembly1.xml",
        "--o", "../output/simulation_stats1.json",
        "--headless",
        "--capture-media"
    ],
    [
        VENV_PYTHON,
        "roblet_simulator.py",
        "--m", "../models/assembly2.xml",
        "--o", "../output/simulation_stats2.json",
        "--headless",
        "--capture-media"
    ],
    [
        VENV_PYTHON,
        "roblet_simulator.py",
        "--m", "../models/assembly3.xml",
        "--o", "../output/simulation_stats3.json",
        "--headless",
        "--capture-media"
    ],
]

processes = []

start_time = time.time()
print(f"Starting {len(commands)} simulations in parallel...")

# Start all simulations without blocking
for cmd in commands:
    p = subprocess.Popen(cmd)
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

