"""Builds paper/Figures/mutations.png from real breeding-decision examples in
an actual evolutionary run, rather than a hand-drawn schematic.

For each target grammar action (Delete Node, Prune Subtree, Add Node) this
scans a run's generation_*/breeding_events.json logs for the first recorded
decision of that type meeting a minimum offspring-size requirement (see
TARGET_ACTIONS), loads the real parent and child ind{i}_graph.json graphs it
produced, and renders them side by side with the added/removed node(s)
circled. The three parent/child pairs are stacked into one 3x2 grid figure
at report quality (600 DPI).

Usage (from the repository root, or anywhere -- paths are resolved relative
to this file):
    python helper_scripts/plot_real_mutations.py
    python helper_scripts/plot_real_mutations.py --run_dir output/evolution_run_norl
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
HELPER_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (SRC_DIR, HELPER_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from graph_visualizer import load_graph_json, ensure_node_positions, draw_graph_to_figure  # noqa: E402

DPI = 600
DEFAULT_RUN_DIR = os.path.join(ROOT_DIR, "output", "evolution_run")
DEFAULT_OUT_PATH = os.path.join(ROOT_DIR, "paper", "Figures", "mutations.png")
HIGHLIGHT_COLOR = "red"
HIGHLIGHT_RADIUS = 0.9

# (breeding_events.json action name, row label, which panel to circle the
# changed node(s) on -- "parent" for a node/subtree that disappears in the
# child, "child" for a node that only exists in the child -- and the
# minimum number of nodes the offspring must still have, so e.g. Prune
# Subtree doesn't land on a degenerate example that prunes down to a single
# root node).
TARGET_ACTIONS = [
    ("DELETE_NODE", "Delete Node", "parent", 1),
    ("PRUNE_SUBTREE", "Prune Subtree", "parent", 3),
    ("ADD_NODE", "Add Node", "child", 1),
]


def _generation_dirs(run_dir):
    """Every generation_* subdirectory of run_dir, in ascending order."""
    found = []
    for name in os.listdir(run_dir):
        if not name.startswith("generation_"):
            continue
        try:
            idx = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        found.append((idx, os.path.join(run_dir, name)))
    return [path for _, path in sorted(found)]


def _node_count(graph_json_path):
    with open(graph_json_path, "r", encoding="utf-8") as f:
        return len(json.load(f).get("nodes", []))


def _find_example(run_dir, action_name, min_child_nodes=1):
    """First breeding_events.json entry matching action_name (scanning
    generations in order) whose parent/child ind{i}_graph.json both exist
    and whose child has at least min_child_nodes nodes.
    Returns (parent_graph_path, child_graph_path)."""
    for gen_dir in _generation_dirs(run_dir):
        events_path = os.path.join(gen_dir, "breeding_events.json")
        if not os.path.exists(events_path):
            continue
        with open(events_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for event in payload.get("events", []):
            if event.get("action") != action_name:
                continue
            parent_path = os.path.join(gen_dir, f"ind{event['parent_ids'][0]}_graph.json")
            child_path = os.path.join(gen_dir, f"ind{event['child_ids'][0]}_graph.json")
            if not (os.path.exists(parent_path) and os.path.exists(child_path)):
                continue
            if _node_count(child_path) < min_child_nodes:
                continue
            return parent_path, child_path
    raise RuntimeError(
        f"No real {action_name} example with both parent/child graphs and >= {min_child_nodes} "
        f"offspring nodes found under {run_dir}"
    )


def _diff_nodes(parent_G, child_G):
    """(nodes removed by the mutation, nodes added by it), by raw node id --
    e.g. {'module_9'} -- comparing the two full (post-mirror) graphs."""
    removed = set(parent_G.nodes()) - set(child_G.nodes())
    added = set(child_G.nodes()) - set(parent_G.nodes())
    return removed, added


def _circle_nodes(ax, G, node_ids):
    """Draws a highlight circle at each node_id's stored position in G. Uses
    G's own (pre-relabel) 'pos'/'_pos' attributes, which draw_graph_to_figure
    leaves numerically unchanged when it relabels nodes for display -- so the
    circle lands exactly on the node draw_graph_to_figure produces for the
    same graph object."""
    pos = {n: attrs.get("pos", attrs.get("_pos", (0.0, 0.0))) for n, attrs in G.nodes(data=True)}
    for node_id in node_ids:
        if node_id not in pos:
            continue
        ax.add_patch(plt.Circle(
            pos[node_id], radius=HIGHLIGHT_RADIUS, fill=False,
            edgecolor=HIGHLIGHT_COLOR, linewidth=3.5, zorder=6,
        ))


def build_figure(run_dir):
    fig, axes = plt.subplots(len(TARGET_ACTIONS), 2, figsize=(9.0, 4.4 * len(TARGET_ACTIONS)))

    for row, (action_name, label, highlight_side, min_child_nodes) in enumerate(TARGET_ACTIONS):
        parent_path, child_path = _find_example(run_dir, action_name, min_child_nodes)
        parent_G = ensure_node_positions(load_graph_json(parent_path))
        child_G = ensure_node_positions(load_graph_json(child_path))
        removed, added = _diff_nodes(parent_G, child_G)

        # aspect_equal=False: a strict 1:1 aspect squeezes a shallow/narrow
        # post-mutation graph (e.g. a 2-node remainder after Delete Node)
        # into an unreadable sliver within its grid cell -- matching
        # evolution_results_visualizer.py's per-individual thumbnails, which
        # use the same tree-schematic (not to-scale) rendering.
        ax_parent, ax_child = axes[row]
        draw_graph_to_figure(parent_G, ax=ax_parent, aspect_equal=False)
        draw_graph_to_figure(child_G, ax=ax_child, aspect_equal=False)

        if highlight_side == "parent":
            _circle_nodes(ax_parent, parent_G, removed)
        else:
            _circle_nodes(ax_child, child_G, added)

        ax_parent.set_ylabel(label, fontsize=13, fontweight="bold", labelpad=14)
        if row == 0:
            ax_parent.set_title("Parent", fontsize=13)
            ax_child.set_title("Offspring", fontsize=13)

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", default=DEFAULT_RUN_DIR,
                         help="Evolutionary run directory with generation_*/breeding_events.json (default: output/evolution_run)")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="Output PNG path (default: paper/Figures/mutations.png)")
    args = parser.parse_args()

    fig = build_figure(args.run_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=DPI, bbox_inches="tight")
    print(f"Saved {args.out} ({DPI} DPI)")


if __name__ == "__main__":
    main()
