"""
Before/after symmetry figure - draws one evolved individual's genotype
graph twice, side by side: the half-graph as it exists pre-mirror
(Section~sec:symmetry) on the left, and the saved full graph exactly as
mjcf_generator built it (symmetry.build_symmetric_graph's output) on the
right.

moo_api._prepare_assembly only ever writes the FULL (post-mirror)
ind{id}_graph.json to disk - the half-graph itself is never persisted
(see its docstring). So "before" here is reconstructed by undoing
build_symmetric_graph: for each mirror anchor (root, and the root's
port-1 spine child if grown - roblet_grammar.is_mirror_anchor), drop
whatever subtree hangs off that anchor's port 3, since that subtree is
exactly the mirrored clone build_symmetric_graph added and nothing else
in the saved graph depends on it (_mirror_anchor_port2 only ever adds
brand-new node ids for the clone, never touches the original half).

Usage from the repository root::

    python helper_scripts/plot_symmetry_before_after.py
    python helper_scripts/plot_symmetry_before_after.py output/evolution_run/generation_1/ind10_graph.json
    python helper_scripts/plot_symmetry_before_after.py --out my_figure.png
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
HELPER_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (SRC_DIR, HELPER_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import roblet_grammar as rg  # noqa: E402
from graph_visualizer import load_graph_json, draw_graph_to_figure  # noqa: E402

DPI = 600


def strip_symmetry(full_G):
    """Returns a NEW graph: full_G with every mirror anchor's port-3
    subtree removed, i.e. the pre-mirror half-graph symmetry.
    build_symmetric_graph originally mirrored FROM. No-op copy if neither
    anchor has anything on port 3 (graph was already a half-graph, or
    both evolvable sides are empty)."""
    half_G = full_G.copy()
    root = rg.root_node(half_G)
    anchors = [root]
    spine_child = _port(half_G.nodes[root]["connectors"], 1)
    if spine_child is not None:
        anchors.append(spine_child)

    for anchor in anchors:
        connectors = half_G.nodes[anchor]["connectors"]
        mirrored_root = _port(connectors, 3)
        if mirrored_root is None:
            continue
        doomed = rg.subtree_nodes(half_G, mirrored_root)
        half_G.remove_nodes_from(doomed)
        connectors[3 if 3 in connectors else "3"] = None

    return half_G


def _port(connectors, port):
    """Reads `connectors[port]`, tolerating both int keys (the in-memory
    graphs roblet_grammar/symmetry build) and str keys (what a saved
    graph JSON round-trips to, since JSON object keys are always
    strings - nx.node_link_graph/load_graph_json don't convert them
    back)."""
    if port in connectors:
        return connectors[port]
    return connectors.get(str(port))


def _default_graph_path():
    """First ind{id}_graph.json found under output/evolution_run (searching
    generation folders in ascending order), so the script runs with no
    arguments out of the box."""
    run_dir = os.path.join(ROOT_DIR, "output", "evolution_run")
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"{run_dir} not found - pass a graph JSON path explicitly")

    gen_dir_re = re.compile(r"^generation_(\d+)$")
    gen_dirs = sorted(
        (d for d in os.listdir(run_dir) if gen_dir_re.match(d)),
        key=lambda d: int(gen_dir_re.match(d).group(1)),
    )
    for gen_dir in gen_dirs:
        full_gen_path = os.path.join(run_dir, gen_dir)
        for name in sorted(os.listdir(full_gen_path)):
            if name.startswith("ind") and name.endswith("_graph.json") and not name.startswith("_"):
                return os.path.join(full_gen_path, name)

    raise FileNotFoundError(f"No ind{{id}}_graph.json files found under {run_dir}")


def plot_before_after(graph_json_path, before_out, after_out):
    full_G = load_graph_json(graph_json_path)
    half_G = strip_symmetry(full_G)

    fig, _ = draw_graph_to_figure(half_G, title="Before symmetry (half genotype)")
    fig.savefig(before_out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    fig, _ = draw_graph_to_figure(full_G, title="After symmetry (full mirrored morphology)")
    fig.savefig(after_out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    return before_out, after_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "graph_json", nargs="?", default=None,
        help="Path to an ind{id}_graph.json file (default: first one found under output/evolution_run)",
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Directory for the two output PNGs (default: the graph JSON's own folder)",
    )
    args = parser.parse_args()

    graph_json_path = args.graph_json or _default_graph_path()
    out_dir = args.out_dir or os.path.dirname(graph_json_path)
    base = os.path.splitext(os.path.basename(graph_json_path))[0]
    before_out = os.path.join(out_dir, f"{base}_before_symmetry.png")
    after_out = os.path.join(out_dir, f"{base}_after_symmetry.png")

    plot_before_after(graph_json_path, before_out, after_out)
    print(f"Saved: {before_out}")
    print(f"Saved: {after_out}")


if __name__ == "__main__":
    main()
