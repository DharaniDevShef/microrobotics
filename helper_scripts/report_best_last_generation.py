"""Report the best XML-backed candidate from the last generation of runs,
including its design-variable values (Table~tab:design-variables) and its
four raw objective values - the exact data needed for the dissertation's
"Evolved Morphologies" design-variable/objective table.

Usage from the repository root::

    python helper_scripts/report_best_last_generation.py
    python helper_scripts/report_best_last_generation.py rank=2
    python helper_scripts/report_best_last_generation.py --rank 2
    python helper_scripts/report_best_last_generation.py output/evolution_run output/evolution_run_repulsive
    python helper_scripts/report_best_last_generation.py output/evolution_run_repulsive/generation_100/ind17_assembly.xml

Passing a path to a specific ``ind{id}_assembly.xml`` file reports that single
candidate (looked up by generation/ind_id in the run's population_history.json)
instead of the best candidate of the last generation.

The scalar fitness uses the same equal-weight min-max normalization as
``objectives_api.scalarize``, reconstructed from the saved population history.
Design variables are read from the candidate's own ind{id}_graph.json, which
already stores the full, post-mirror morphology.
"""

import argparse
import json
import re
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


def _fold_type_bucket(module_type):
    """Map a graph.json module_type string onto one of the three design
    variable categories (Section~sec:evo-learning / Table~tab:design-
    variables): rigid, mountain-fold, valley-fold. Matching is
    case-insensitive since saved graphs mix "non-foldable"/"Mountain
    fold"/"valley fold" casing across runs."""
    normalized = (module_type or "").strip().lower()
    if "mountain" in normalized:
        return "mountain"
    if "valley" in normalized:
        return "valley"
    return "rigid"


def _uniform_angle(values):
    """Design (Section~sec:evo-learning) applies one shared hinge_angle/
    light_hinge_angle value across all foldable/light-sensitive nodes in
    a genotype, so this normally collapses to a single number. Returns
    (value, is_uniform): is_uniform is False if the saved graph actually
    contains more than one distinct nonzero value, so a caller can flag
    that rather than silently reporting just one of them."""
    distinct = sorted({round(float(v), 4) for v in values if abs(float(v)) > 1e-9})
    if not distinct:
        return 0.0, True
    if len(distinct) == 1:
        return distinct[0], True
    return distinct, False


def _design_variables(graph_path):
    """Extract the six design-variable values (Table~tab:design-variables)
    from one individual's saved genotype graph (ind{id}_graph.json). The
    graph is already the full, post-mirror morphology (Section~
    sec:symmetry), so len(nodes) here is the post-mirror module count."""
    with graph_path.open("r", encoding="utf-8") as stream:
        graph = json.load(stream)
    nodes = graph.get("nodes", [])

    fold_counts = {"rigid": 0, "mountain": 0, "valley": 0}
    hinge_angles = []
    light_hinge_angles = []
    light_sensitive_count = 0
    for node in nodes:
        fold_counts[_fold_type_bucket(node.get("module_type"))] += 1
        if node.get("module_type", "").strip().lower() != "non-foldable":
            hinge_angles.append(node.get("hinge_angle", 0.0))
        if node.get("light_sensitive"):
            light_sensitive_count += 1
            light_hinge_angles.append(node.get("light_hinge_angle", 0.0))

    hinge_angle, hinge_uniform = _uniform_angle(hinge_angles)
    light_hinge_angle, light_hinge_uniform = _uniform_angle(light_hinge_angles)

    return {
        "module_count": len(nodes),
        "fold_counts": fold_counts,
        "hinge_angle": hinge_angle,
        "hinge_angle_uniform": hinge_uniform,
        "light_sensitive_count": light_sensitive_count,
        "light_hinge_angle": light_hinge_angle,
        "light_hinge_angle_uniform": light_hinge_uniform,
    }


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
        graph_path = generation_dir / f"ind{candidate_id}_graph.json"
        if not xml_path.exists() or not graph_path.exists():
            continue
        objectives = objectives_api.migrate_legacy_objectives(candidate["objectives"])
        candidates.append((
            _fitness(objectives, ranges, is_repulsive),
            candidate_id,
            objectives,
            xml_path.resolve(),
            graph_path.resolve(),
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


_XML_NAME_RE = re.compile(r"^ind(\d+)_assembly\.xml$")
_GENERATION_DIR_RE = re.compile(r"^generation_(\d+)$")


def _candidate_from_xml(xml_path):
    """Resolve a directly-given ind{id}_assembly.xml path (run_dir/generation_N/
    ind{id}_assembly.xml) into the same (generation, (fitness, candidate_id,
    objectives, xml_path, graph_path)) shape that _best_candidate returns, plus
    that candidate's rank within its own generation."""
    xml_path = xml_path.resolve()
    xml_match = _XML_NAME_RE.match(xml_path.name)
    if not xml_match:
        raise ValueError(f"{xml_path} is not an ind{{id}}_assembly.xml file")
    candidate_id = int(xml_match.group(1))

    generation_dir = xml_path.parent
    generation_match = _GENERATION_DIR_RE.match(generation_dir.name)
    if not generation_match:
        raise ValueError(f"{xml_path} is not inside a generation_<N> directory")
    generation_number = int(generation_match.group(1))

    run_dir = generation_dir.parent
    graph_path = generation_dir / f"ind{candidate_id}_graph.json"
    if not graph_path.exists():
        raise ValueError(f"{graph_path} not found")

    history = _load_history(run_dir)
    generation_entry = next(
        (entry for entry in history if entry["generation"] == generation_number), None
    )
    if generation_entry is None:
        raise ValueError(f"generation {generation_number} not found in {run_dir / 'population_history.json'}")

    ranges = _normalization_ranges(history)
    is_repulsive = "repulsive" in run_dir.name.lower()

    scored = []
    target_index = None
    for candidate in generation_entry.get("population", []):
        if candidate.get("ind_id") is None:
            continue
        objectives = objectives_api.migrate_legacy_objectives(candidate["objectives"])
        scored.append((_fitness(objectives, ranges, is_repulsive), candidate["ind_id"], objectives))
        if candidate["ind_id"] == candidate_id:
            target_index = len(scored) - 1

    if target_index is None:
        raise ValueError(f"ind_id {candidate_id} not found in generation {generation_number} of {run_dir}")

    scored.sort(key=lambda item: item[0], reverse=True)
    rank = next(index for index, item in enumerate(scored, start=1) if item[1] == candidate_id)
    fitness, _, objectives = next(item for item in scored if item[1] == candidate_id)

    return run_dir, generation_number, rank, (fitness, candidate_id, objectives, xml_path, graph_path)


_OBJECTIVE_LABELS = {
    "f1_folded_gait_velocity": ("f1 folded-gait velocity", "m/s"),
    "f2_entropy": ("f2 entropy gain (H3D - H2D)", ""),
    "f3_pheromone_yaw_response": ("f3 pheromone yaw response", "deg"),
    "f4_pheromone_speed_response": ("f4 pheromone speed response", ""),
}


def _print_report(run_dir, generation, rank, generation_label, candidate):
    fitness, candidate_id, objectives, xml_path, graph_path = candidate
    design_variables = _design_variables(graph_path)

    print(f"{run_dir}")
    print(f"  {generation_label}: {generation}")
    print(f"  rank: {rank}")
    print(f"  candidate: ind{candidate_id}")
    print(f"  fitness: {fitness:.6f}")

    print("  design variables:")
    print(f"    module count (post-mirror): {design_variables['module_count']}")
    fold_counts = design_variables["fold_counts"]
    print(
        "    fold-type distribution (rigid/mountain/valley): "
        f"{fold_counts['rigid']}/{fold_counts['mountain']}/{fold_counts['valley']}"
    )
    hinge_note = "" if design_variables["hinge_angle_uniform"] else " (NOT uniform across foldable nodes - see raw values)"
    print(f"    hinge angle: {design_variables['hinge_angle']}{hinge_note}")
    print(f"    light-sensitive nodes: {design_variables['light_sensitive_count']}")
    light_hinge_note = "" if design_variables["light_hinge_angle_uniform"] else " (NOT uniform across light-sensitive nodes - see raw values)"
    print(f"    light-detection hinge angle: {design_variables['light_hinge_angle']}{light_hinge_note}")

    print("  objective values:")
    for name in objectives_api.OBJECTIVE_NAMES:
        label, unit = _OBJECTIVE_LABELS[name]
        value = float(objectives.get(name, 0.0))
        suffix = f" {unit}" if unit else ""
        print(f"    {label}: {value:.6f}{suffix}")

    print(f"  xml: {xml_path}")
    print(f"  graph: {graph_path}")


def _report(run_dir, rank):
    generation, candidate = _best_candidate(run_dir, rank)
    _print_report(run_dir, generation, rank, "last generation", candidate)


def _report_xml(xml_path):
    run_dir, generation, rank, candidate = _candidate_from_xml(xml_path)
    _print_report(run_dir, generation, rank, "generation", candidate)


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
        "targets",
        nargs="*",
        type=Path,
        default=[
            ROOT_DIR / "output" / "evolution_run",
            ROOT_DIR / "output" / "evolution_run_repulsive",
        ],
        help=(
            "Evolution run directories (default: attractive and repulsive output "
            "runs), and/or specific ind{id}_assembly.xml file paths to report a "
            "single candidate."
        ),
    )
    args = parser.parse_args(argv)
    if args.rank < 1:
        parser.error("--rank must be at least 1")

    failed = False
    for target in args.targets:
        try:
            if target.is_file():
                _report_xml(target)
            else:
                _report(target, args.rank)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            failed = True
            print(f"{target}: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())