import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_graph_json(graph_json_path):
    with open(graph_json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    G = nx.DiGraph()
    for node in data["nodes"]:
        node_id = node["id"].split("_")[-1]  # Extract numeric part (e.g., "1")
        G.add_node(
            node_id,
            module_type=node["module_type"],
            hinge_angle=node["hinge_angle"],
            pos=node.get("_pos", (0.0, 0.0))
        )

    for edge in data["edges"]:
        source_id = str(edge["source"]).split("_")[-1]
        target_id = str(edge["target"]).split("_")[-1]
        G.add_edge(
            source_id,
            target_id,
            connector1=edge["connector1"],
            connector2=edge["connector2"]
        )

    return G


def draw_graph_to_figure(graph_or_data, figure=None, title="Roblet Morphology Graph (Directed)"):
    if isinstance(graph_or_data, str):
        G = load_graph_json(graph_or_data)
    elif isinstance(graph_or_data, dict):
        G = nx.node_link_graph(graph_or_data)
    else:
        G = graph_or_data.copy()

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

    ax.legend(
        handles=legend_patches,
        loc="upper left",
        title="Module Configuration",
        framealpha=0.9,
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold")
    figure.tight_layout()
    return figure, ax


def show_graph(graph_json_path):
    figure, _ = draw_graph_to_figure(graph_json_path)
    plt.show()


if __name__ == "__main__":
    show_graph("../graphs/star.json")