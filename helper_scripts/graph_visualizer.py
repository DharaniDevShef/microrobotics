import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ----------------------------
# Load JSON
# ----------------------------
with open("beetle_graph.json", "r") as f:
    data = json.load(f)

# ----------------------------
# Create Directed Graph
# ----------------------------
G = nx.DiGraph()

for node in data["nodes"]:
    G.add_node(
        node["id"],
        module_type=node["module_type"],
        hinge_angle=node["hinge_angle"],
        pos=node["_pos"]
    )

for edge in data["edges"]:
    G.add_edge(
        edge["source"],
        edge["target"],
        connector1=edge["connector1"],
        connector2=edge["connector2"]
    )

# ----------------------------
# Position from JSON
# ----------------------------
pos = {
    node["id"]: node["_pos"]
    for node in data["nodes"]
}

# ----------------------------
# Color based on module type
# ----------------------------
# Updated to match exact UI color specifications
module_colors = {
    "non-foldable": "gray",          # Non-fold -> Gray
    "Mountain fold": "#0066CC",      # Mountain fold -> Blue
    "valley fold": "#8A2BE2"         # Valley fold -> Violet
}

node_colors = []

for node_id, attrs in G.nodes(data=True):
    # Base node module_1 keeps its distinctive soft gold styling
    if node_id == "module_1":
        color = "orange"
    else:
        module_type = attrs.get("module_type", "non-foldable")
        color = module_colors.get(module_type, "gray")
        
    node_colors.append(color)

# ----------------------------
# Draw graph
# ----------------------------
plt.figure(figsize=(9, 9))

NODE_SIZE = 3000

nx.draw_networkx_nodes(
    G,
    pos,
    node_color=node_colors,
    node_size=NODE_SIZE,
    edgecolors="black"
)

nx.draw_networkx_edges(
    G,
    pos,
    width=2,
    node_size=NODE_SIZE,     # Keeps arrowheads cleanly placed on the boundary
    arrowstyle="->",
    arrowsize=20,
    edge_color="dimgray"
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=9,
    font_weight="bold",
    font_color="white" if any(c in ["#0066CC", "#8A2BE2"] for c in node_colors) else "black"
)

# ----------------------------
# Connector labels
# ----------------------------
edge_labels = {}

for u, v, attr in G.edges(data=True):
    edge_labels[(u, v)] = (
        f"{attr['connector1']}-{attr['connector2']}"
    )

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=8,
    label_pos=0.5
)

# ----------------------------
# Legend generation
# ----------------------------
# Create custom patches for the color legend
legend_patches = [
    mpatches.Patch(color="orange", label="Base Node (module_1)"),
    mpatches.Patch(color="gray", label="Non-Foldable"),
    mpatches.Patch(color="#0066CC", label="Mountain Fold"),
    mpatches.Patch(color="#8A2BE2", label="Valley Fold")
]

plt.legend(
    handles=legend_patches, 
    loc="upper left", 
    title="Module Configuration", 
    framealpha=0.9, 
    fontsize=10
)

plt.axis("equal")
plt.title("Roblet Morphology Graph (Directed)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()