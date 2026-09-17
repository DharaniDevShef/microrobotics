# Microrobotics - Evolutionary Roblet Morphology

Code for evolving the morphology of self-folding, self-assembling
magnetically-actuated microrobots ("**Roblets**") using a Sobol-seeded,
RL-guided NSGA-III multi-objective evolutionary loop, with candidate
morphologies simulated in [MuJoCo](https://mujoco.readthedocs.io/).

Each individual is a directed graph of foldable modules (a "genotype").
Every generation the graphs are bred (mutation/crossover, either driven by
a trained graph-transformer PPO policy or a random baseline), compiled into
an MJCF assembly, physically simulated, scored on multiple objectives
(e.g. folded-gait velocity), and selected via NSGA-III. See
[`src/main.py`](src/main.py) for the full loop description.

## Repository layout

| Path                                                                                                | What it's for                                                                                                              |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| [`src/main.py`](src/main.py)                                                                       | Entry point — runs the RL-guided NSGA-III evolutionary loop end-to-end (checkpointing, resuming, plotting).               |
| [`src/roblet_simulator.py`](src/roblet_simulator.py)                                               | MuJoCo simulation of a single roblet assembly (folding + magnetic actuation); also runnable standalone with a live viewer. |
| [`src/moo_api.py`](src/moo_api.py)                                                                 | Multi-objective optimization engine: population loop, breeding, evaluation, NSGA-III survival.                             |
| [`src/objectives_api.py`](src/objectives_api.py)                                                   | Converts a simulation's`stats.json` into the f1..f4 objective vector used for selection/RL reward.                       |
| [`src/rl_api.py`](src/rl_api.py)                                                                   | Graph-transformer actor-critic (PPO) policy that selects/parameterizes mutation & crossover actions.                       |
| [`src/random_baseline.py`](src/random_baseline.py)                                                 | Uniform-random genetic-operator arm, used when RL-assisted breeding is switched off (classic-GA comparison).               |
| [`src/roblet_grammar.py`](src/roblet_grammar.py)                                                   | Shared genotype schema and grammar-legal graph edit primitives (mutation/crossover actions).                               |
| [`src/symmetry.py`](src/symmetry.py)                                                               | Mirrors a half-genotype graph into a full bilaterally symmetric shape at build time.                                       |
| [`src/entropy_api.py`](src/entropy_api.py)                                                         | Multi-scale Shannon shape-entropy metric over module positions.                                                            |
| [`src/sim_executor.py`](src/sim_executor.py)                                                       | Runs many`roblet_simulator.py` evaluations in parallel OS processes.                                                     |
| [`src/parallel_executor.py`](src/parallel_executor.py)                                             | Standalone 4-way parallel simulation throughput check.                                                                     |
| [`src/checkpoint.py`](src/checkpoint.py)                                                           | Saves/loads full run state (RL weights, population, RNG, generation) for crash-safe resuming.                              |
| [`src/plotting_api.py`](src/plotting_api.py)                                                       | Generates all per-run figures (Pareto front, fitness trends, convergence, RL diagnostics, ...).                            |
| [`src/compare.py`](src/compare.py)                                                                 | Renders the RL-guided vs. random-baseline comparison figure from two completed runs.                                       |
| [`src/simple_roblet.py`](src/simple_roblet.py)                                                     | Minimal single-hinge MuJoCo viewer sanity check.                                                                           |
| [`helper_scripts/mjcf_generator.py`](helper_scripts/mjcf_generator.py)                             | Compiles a genotype graph into an MJCF assembly XML.                                                                       |
| [`helper_scripts/graph_visualizer.py`](helper_scripts/graph_visualizer.py)                         | Draws a genotype graph JSON as a top-down tree.                                                                            |
| [`helper_scripts/pattern_gen.py`](helper_scripts/pattern_gen.py)                                   | PyQt6 GUI for hand-authoring a module layout and exporting it as a genotype graph.                                         |
| [`helper_scripts/evolution_results_visualizer.py`](helper_scripts/evolution_results_visualizer.py) | PyQt6 GUI for browsing a completed/in-progress evolution run (population, screenshots, lineage).                           |
| [`helper_scripts/plot_morphology_grid.py`](helper_scripts/plot_morphology_grid.py)                 | Builds a "morphology over generations" comparison figure.                                                                  |
| [`helper_scripts/plot_real_mutations.py`](helper_scripts/plot_real_mutations.py)                   | Builds a figure from real breeding-decision examples in a run.                                                             |
| [`helper_scripts/plot_symmetry_before_after.py`](helper_scripts/plot_symmetry_before_after.py)     | Before/after symmetry-mirroring figure for one individual.                                                                 |
| [`helper_scripts/report_best_last_generation.py`](helper_scripts/report_best_last_generation.py)   | Reports the best individual from a run's last generation with its design variables and objectives.                         |
| [`helper_scripts/magnet_mesh_gen.py`](helper_scripts/magnet_mesh_gen.py)                           | Generates the cylindrical magnet STL mesh used in the module CAD.                                                          |
| [`helper_scripts/sample_moo.py`](helper_scripts/sample_moo.py)                                     | Standalone pymoo NSGA-III demo (DTLZ1 benchmark), unrelated to the roblet pipeline.                                        |
| [`fusion_scripts/`](fusion_scripts)                                                                | Autodesk Fusion 360 script add-ins (run*inside* Fusion, not with `uv`/`python`) — see below.                        |
| [`assets/cad_models`](assets/cad_models)                                                           | Fusion 360 CAD source files for the module bodies and sub-assemblies.                                                      |
| [`meshes/`](meshes)                                                                                | Exported STL meshes referenced by the MJCF assemblies.                                                                     |
| [`models/`](models)                                                                                | Example/generated MJCF assembly XMLs.                                                                                      |
| [`graphs/`](graphs)                                                                                | Example genotype graph JSONs (hand-authored morphologies for testing).                                                     |
| [`doc/`](doc)                                                                                      | Design notes (module/magnet specs).                                                                                        |
| [`output/`](output)                                                                                | Default destination for evolution-run artifacts (checkpoints, per-generation data, plots, screenshots). Git-ignored.       |

## Requirements

- Windows 10/11 (these instructions target Windows; the code itself is cross-platform Python)
- Python **3.14+**
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A GPU is not required, but training runs are significantly faster with one (PyTorch/MuJoCo)

## Setup (Windows)

1. **Install Python 3.14+** and confirm it's on your `PATH`:

   ```powershell
   python --version
   ```
2. **Install `uv`**:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Follow the installer's instructions to add `uv` to your `PATH` (it will
   print the exact line to add, or you may need to restart your terminal),
   then confirm it works:

   ```powershell
   uv --version
   ```
3. **Clone the repo and install dependencies**:

   ```powershell
   git clone <this-repo-url>
   cd microrobotics
   uv sync
   ```

   `uv sync` creates a `.venv/` and installs everything listed in
   [`pyproject.toml`](pyproject.toml) (MuJoCo, pymoo, PyTorch + torch-geometric,
   stable-baselines3, PyQt6, trimesh, etc.).
4. **Activate the virtual environment** (only needed if you want to run
   `python ...` directly instead of `uv run ...`):

   ```powershell
   ./.venv/Scripts/activate
   ```

## Usage

All `src/` and `helper_scripts/` scripts assume they're run **from their own
directory** (they use relative paths like `../output`, `../models`,
`../graphs`). Either `cd` into the directory first, or use `uv run` with the
full path — the examples below `cd` for clarity.

### Run the evolutionary loop

```powershell
cd src
uv run python main.py
```

This runs the full Sobol-seeded, RL-guided NSGA-III loop (see the module
docstring in [`src/main.py`](src/main.py) for the per-generation pipeline).
Key settings are constants at the top of `main.py`:

| Constant                           | Meaning                                                                                                                                 |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `POP_SIZE`                       | Population size per generation                                                                                                          |
| `N_GENERATIONS`                  | Number of generations to run                                                                                                            |
| `SIM_SECONDS`                    | Simulated seconds per individual evaluation                                                                                             |
| `SEED`                           | RNG seed for a fresh run (ignored when resuming)                                                                                        |
| `RL_ASSISTED_GENETIC_OPERATIONS` | `True` = PPO-trained breeding policy, `False` = random-baseline (classic GA) arm — each writes to its own `output/` subdirectory |
| `PHEROMONE_RESPONSE_TYPE`        | `"attractive"` or `"repulsive"` light-response objective variant                                                                    |

**Resuming:** the script always checks for `output/<run>/checkpoint.pt` at
startup and automatically continues from the last completed generation — no
flag needed. Re-running `python main.py` after a crash, Ctrl-C, or error
just picks up where it left off.

### Run a single simulation

```powershell
cd src
uv run python roblet_simulator.py --m ../models/simple_roblet.xml
```

Opens a live MuJoCo viewer for the given model. Useful flags:

| Flag                                            | Meaning                                                                   |
| ----------------------------------------------- | ------------------------------------------------------------------------- |
| `--m <path>`                                  | MJCF model to load                                                        |
| `--o <path>`                                  | Output path for the simulation stats JSON                                 |
| `--headless`                                  | Run without the live viewer (required for the flags below)                |
| `--capture_img`                               | Save a final-pose screenshot PNG                                          |
| `--capture_gif`                               | Save a movement GIF                                                       |
| `--sweep_b`                                   | Sweep magnetic field intensity and keep the best-performing run           |
| `--light_tests`                               | Run the two-stage pheromone/light-response test instead of free-roam gait |
| `--capture_video`                             | With`--light_tests`, save an MP4 of each stage                          |
| `--log-file <path>` / `--log-level <LEVEL>` | Logging options                                                           |

Run `uv run python roblet_simulator.py --help` for the complete list.

### Compare RL-guided vs. random-baseline runs

After both an `RL_ASSISTED_GENETIC_OPERATIONS = True` and `= False` run have
completed:

```powershell
cd src
uv run python compare.py
```

### Helper / visualization scripts

Run from the `helper_scripts/` directory:

```powershell
cd helper_scripts
uv run python evolution_results_visualizer.py   # browse a run's population/screenshots/lineage (GUI)
uv run python pattern_gen.py                    # hand-author a module layout -> genotype graph (GUI)
uv run python graph_visualizer.py               # quick static plot of a genotype graph JSON
uv run python report_best_last_generation.py    # print the best individual of the last generation
```

The `plot_*.py` scripts regenerate specific figures from a completed run's
output directory — see each file's docstring for what it expects.

### Fusion 360 scripts

The scripts in [`fusion_scripts/`](fusion_scripts) (`ExportToMeshes.py`,
`MJCFGenerator.py`, `ModulePosFinder.py`) are **Fusion 360 add-in scripts**,
not standalone Python — they use the `adsk` API and only run inside Fusion
360 (Utilities → Add-Ins → Scripts and Add-Ins → Run). Use them from an open
Fusion 360 design to export module meshes to STL, generate an MJCF layout
from the current CAD assembly, or snap CAD occurrences to match a target
MJCF/graph layout.

## Output

A run writes to `output/<run-name>/` (e.g. `output/evolution_run/`):

- `checkpoint.pt` — RL weights/optimizer state, population, RNG state, current generation
- `generation_<n>/` — per-generation MJCF XMLs, simulation stats, screenshots, breeding events
- `population_history.json`, `generation_stats.json` — cumulative per-generation data
- `pareto_archive.json` — running Pareto-optimal archive
- `main.log` — run log
- Pareto front / fitness-trend / convergence / RL-diagnostic PNGs (regenerated every `PLOT_EVERY_N_GENERATIONS` generations, and always on the last one)

## Further reading

- [`doc/design_info.txt`](doc/design_info.txt) — physical module/magnet specifications
