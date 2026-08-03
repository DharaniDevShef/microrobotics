import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_graph_json(graph_json_path):
    with open(graph_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    # nx.node_link_graph carries every node/edge attribute through verbatim
    # (module_type, _pos, connectors, depth, parent, type_id, ...), so this
    # stays in sync automatically as the schema gains fields for GNN/graph
    # transformer/pymoo use, instead of hand-listing keys here.
    G = nx.node_link_graph(data, edges="edges")
    if not G.is_directed():
        raise ValueError(
            f"{graph_json_path} is an undirected graph; re-save it with the "
            "updated Pattern Generator to get module_1-rooted hierarchy edges."
        )
    return G


def compute_layout_positions(G, x_gap=2.5, y_gap=2.5):
    """Lay out nodes top-down by tree depth, using edges (and parent/depth
    attrs if present) to find the hierarchy. Used as a fallback for graphs
    that have no saved _pos/pos (e.g. graphs straight out of roblet_grammar,
    before any physical folding geometry has been assigned).
    """
    roots = [n for n, d in G.in_degree() if d == 0] or [next(iter(G.nodes()))]

    children = {n: [] for n in G.nodes()}
    for u, v in G.edges():
        children[u].append(v)

    positions = {}
    visited = set()
    next_leaf_x = [0.0]

    def place(node, depth):
        visited.add(node)
        kids = [c for c in children[node] if c not in visited]
        if not kids:
            x = next_leaf_x[0]
            next_leaf_x[0] += x_gap
        else:
            for kid in kids:
                place(kid, depth + 1)
            x = sum(positions[kid][0] for kid in kids) / len(kids)
        positions[node] = (x, -depth * y_gap)

    for root in roots:
        if root not in visited:
            place(root, 0)

    # Any nodes unreachable from a root (disconnected fragments) still need a spot.
    for node in G.nodes():
        if node not in positions:
            positions[node] = (next_leaf_x[0], 0.0)
            next_leaf_x[0] += x_gap

    return normalize_positions(positions)


def normalize_positions(positions, target_extent=10.0):
    """Rescale positions to fit within a target_extent x target_extent box,
    preserving relative proportions (no independent x/y stretch) so the
    layout stays undistorted under ax.set_aspect("equal").
    """
    if not positions:
        return positions

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    x_range = max(xs) - min(xs) or 1.0
    y_range = max(ys) - min(ys) or 1.0
    scale = target_extent / max(x_range, y_range)
    x0, y0 = min(xs), min(ys)
    return {node: ((x - x0) * scale, (y - y0) * scale) for node, (x, y) in positions.items()}


def ensure_node_positions(G):
    """Return a graph where every node has a 'pos' (or '_pos') attribute,
    computing a fallback layout automatically when the source JSON has none.
    """
    if all("pos" in attrs or "_pos" in attrs for _, attrs in G.nodes(data=True)):
        return G

    G = G.copy()
    layout = compute_layout_positions(G)
    for node, attrs in G.nodes(data=True):
        if "pos" not in attrs and "_pos" not in attrs:
            attrs["pos"] = layout[node]
    return G


def draw_graph_to_figure(graph_or_data, figure=None, title="Roblet Morphology Graph (Directed)", aspect_equal=True):
    if isinstance(graph_or_data, str):
        G = load_graph_json(graph_or_data)
    elif isinstance(graph_or_data, dict):
        G = nx.node_link_graph(graph_or_data, edges="edges")
    else:
        G = graph_or_data.copy()

    G = ensure_node_positions(G)

    relabel_map = {node: str(node).split("_")[-1] for node in G.nodes()}
    G = nx.relabel_nodes(G, relabel_map)

    if figure is None:
        figure = plt.figure(figsize=(9, 9))

    ax = figure.subplots()
    pos = {
        node_id: attrs.get("pos", attrs.get("_pos", (0.0, 0.0)))
        for node_id, attrs in G.nodes(data=True)
    }

    module_colors = {
        "non-foldable": "gray",
        "Mountain fold": "#0066CC",
        "valley fold": "#8A2BE2",
    }

    node_colors = []
    for node_id, attrs in G.nodes(data=True):
        if node_id == "1":
            color = "orange"
        else:
            module_type = attrs.get("module_type", "non-foldable")
            color = module_colors.get(module_type, "gray")
        node_colors.append(color)

    NODE_SIZE = 1000

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=NODE_SIZE,
        edgecolors="black",
        ax=ax,
    )

    # Edge drawing with explicit arrow visibility parameters
    nx.draw_networkx_edges(
        G,
        pos,
        width=2,
        node_size=NODE_SIZE,      # Sync node size so arrows stop outside nodes
        arrows=True,                # Enable arrows for directed edges
        arrowstyle="-|>",         # Sharp arrowhead style
        arrowsize=25,             # Enlarged size so it stands out
        min_target_margin=12,     # Offset margin from node border
        edge_color="dimgray",
        ax=ax,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
        font_weight="bold",
        font_color="white" if any(c in ["#0066CC", "#8A2BE2"] for c in node_colors) else "black",
        ax=ax,
    )

    edge_labels = {}
    for u, v, attr in G.edges(data=True):
        edge_labels[(u, v)] = f"{attr.get('connector1', '?')}-{attr.get('connector2', '?')}"

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        label_pos=0.5,
        ax=ax,
    )

    legend_patches = [
        mpatches.Patch(color="orange", label="Base Node (Module 1)"),
        mpatches.Patch(color="gray", label="Non-Foldable"),
        mpatches.Patch(color="#0066CC", label="Mountain Fold"),
        mpatches.Patch(color="#8A2BE2", label="Valley Fold"),
    ]

    # ax.legend(
    #     handles=legend_patches,
    #     loc="upper left",
    #     title="Module Configuration",
    #     framealpha=0.9,
    #     fontsize=10,
    # )
    if aspect_equal:
        ax.set_aspect("equal")
    # ax.set_title(title, fontsize=12, fontweight="bold")
    figure.tight_layout()
    return figure, ax


def show_graph(graph_json_path):
    figure, _ = draw_graph_to_figure(graph_json_path)
    plt.show()


if __name__ == "__main__":
    show_graph("../graphs/assembly_graph.json")