"""Report the best XML-backed candidate from the last generation of runs.

Usage from the repository root::

    python helper_scripts/report_best_last_generation.py
    python helper_scripts/report_best_last_generation.py rank=2
    python helper_scripts/report_best_last_generation.py --rank 2
    python helper_scripts/report_best_last_generation.py output/evolution_run output/evolution_run_repulsive

The scalar fitness uses the same equal-weight min-max normalization as
``objectives_api.scalarize``, reconstructed from the saved population history.
"""

import argparse
import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import objectives_api  # noqa: E402


def _load_history(run_dir):
    with (run_dir / "population_history.json").open("r", encoding="utf-8") as stream:
        history = json.load(stream)
    return sorted(history, key=lambda entry: entry["generation"])


def _normalization_ranges(history):
    ranges = {}
    for generation in history:
        for candidate in generation.get("population", []):
            objectives = objectives_api.migrate_legacy_objectives(candidate["objectives"])
            for name in objectives_api.OBJECTIVE_NAMES:
                value = float(objectives.get(name, 0.0))
                if name not in ranges:
                    ranges[name] = [value, value]
                else:
                    ranges[name][0] = min(ranges[name][0], value)
                    ranges[name][1] = max(ranges[name][1], value)
    return ranges


def _fitness(objectives, ranges, is_repulsive):
    contributions = []
    for name in objectives_api.OBJECTIVE_NAMES:
        low, high = ranges.get(name, [0.0, 0.0])
        if high - low < 1e-9:
            continue
        normalized = (float(objectives.get(name, 0.0)) - low) / (high - low)
        maximize = objectives_api.MAXIMIZE[name]
        if is_repulsive and name in {
            "f3_pheromone_yaw_response",
            "f4_pheromone_speed_response",
        }:
            maximize = not maximize
        contributions.append(normalized if maximize else 1.0 - normalized)

    if contributions:
        return sum(contributions) / len(contributions)

    return sum(
        (float(objectives.get(name, 0.0)) if objectives_api.MAXIMIZE[name] else -float(objectives.get(name, 0.0)))
        for name in objectives_api.OBJECTIVE_NAMES
    )


def _best_candidate(run_dir, rank=1):
    history = _load_history(run_dir)
    if not history:
        raise ValueError("population_history.json contains no generations")

    last_generation = history[-1]
    generation_dir = run_dir / f"generation_{last_generation['generation']}"
    ranges = _normalization_ranges(history)
    is_repulsive = "repulsive" in run_dir.name.lower()
    candidates = []
    for candidate in last_generation.get("population", []):
        candidate_id = candidate.get("ind_id")
        if candidate_id is None:
            continue
        xml_path = generation_dir / f"ind{candidate_id}_assembly.xml"
        if not xml_path.exists():
            continue
        objectives = objectives_api.migrate_legacy_objectives(candidate["objectives"])
        candidates.append((
            _fitness(objectives, ranges, is_repulsive),
            candidate_id,
            objectives,
            xml_path.resolve(),
        ))

    if not candidates:
        raise ValueError(f"no XML-backed candidates found in {generation_dir}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    if rank > len(candidates):
        raise ValueError(
            f"rank {rank} is unavailable in {generation_dir}; "
            f"only {len(candidates)} XML-backed candidates found"
        )
    return last_generation["generation"], candidates[rank - 1]


def _report(run_dir, rank):
    generation, (fitness, candidate_id, objectives, xml_path) = _best_candidate(run_dir, rank)
    print(f"{run_dir}")
    print(f"  last generation: {generation}")
    print(f"  rank: {rank}")
    print(f"  candidate: ind{candidate_id}")
    print(f"  fitness: {fitness:.6f}")
    for name in objectives_api.OBJECTIVE_NAMES:
        print(f"  {name}: {float(objectives.get(name, 0.0)):.6f}")
    print(f"  xml: {xml_path}")


def main():
    argv = sys.argv[1:]
    rank_equals = [argument for argument in argv if argument.startswith("rank=")]
    if rank_equals:
        if len(rank_equals) > 1:
            raise SystemExit("rank= may only be specified once")
        argv.remove(rank_equals[0])
        argv.extend(["--rank", rank_equals[0].split("=", 1)[1]])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="1-based fitness rank in the last generation (1=best; default: 1).",
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        default=[
            ROOT_DIR / "output" / "evolution_run",
            ROOT_DIR / "output" / "evolution_run_repulsive",
        ],
        help="Evolution run directories (default: attractive and repulsive output runs).",
    )
    args = parser.parse_args(argv)
    if args.rank < 1:
        parser.error("--rank must be at least 1")

    failed = False
    for run_dir in args.run_dirs:
        try:
            _report(run_dir, args.rank)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            failed = True
            print(f"{run_dir}: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())