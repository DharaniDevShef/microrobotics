"""
Roblet Grammar - Shared genotype schema and grammar-legal graph edit
primitives for the RL-guided NSGA-III evolutionary framework.

The genotype is a NetworkX DiGraph (see graphs/*.json for the on-disk
schema, and helper_scripts/mjcf_generator.py for how it becomes an MJCF
assembly). This module is the single place that knows how to read/write
that schema, so moo_api.py, rl_api.py, and mjcf_generator.py all agree on it.

Design variables (per the design doc):
    1. Module Type (Vi)      - categorical {non-foldable, Mountain fold, valley fold}
    2. Graph Adjacency (Aij) - port-to-port connectivity, ports in {1, 2, 3}
    3. Module Count (N)      - N in [2, 40]
    4. Hinge Angle (theta_i) - continuous, [0, 45] degrees
"""

import copy
import random
from enum import Enum, auto

import networkx as nx

MODULE_TYPES = ["non-foldable", "Mountain fold", "valley fold"]
MODULE_TYPE_IDS = {name: idx for idx, name in enumerate(MODULE_TYPES)}
PORTS = (1, 2, 3)
MIN_MODULES = 2
# Every genotype evolved here is only HALF the final shape - the root's
# port 3 is reserved for an auto-mirrored clone of whatever grows on port
# 2 (see growable_ports() below and symmetry.py's build_symmetric_graph),
# built only at MJCF-generation time. So this bounds the HALF, not the
# final module count: a half of MAX_MODULES yields a final shape of up to
# 2*MAX_MODULES - 1 modules (root shared, port-2 subtree + its mirror on
# port 3) - kept at 20 here so that stays within the design doc's original
# N in [2, 40] for the physical robot.
MAX_MODULES = 20
MIN_HINGE_ANGLE = 0.0
MAX_HINGE_ANGLE = 45.0


class Action(Enum):
    """Grammar actions. rl_api.py's policy selects among ALL of these
    (mutation AND crossover) from one unified, masked action head."""
    ADD_NODE = auto()
    DELETE_NODE = auto()
    PRUNE_SUBTREE = auto()
    MUTATE_FOLD_TYPE = auto()
    MUTATE_HINGE_ANGLE = auto()
    RECONNECT_PORT = auto()
    GRAFT_SUBTREE = auto()
    SWAP_SUBTREES = auto()


MUTATION_ACTIONS = [
    Action.ADD_NODE,
    Action.DELETE_NODE,
    Action.PRUNE_SUBTREE,
    Action.MUTATE_FOLD_TYPE,
    Action.MUTATE_HINGE_ANGLE,
    Action.RECONNECT_PORT,
]

# Crossover needs a second parent graph (donor/partner), unlike the
# single-graph MUTATION_ACTIONS above - see rl_api.py for how the policy
# handles that (a second Graph Transformer encoder pass + cross-attention).
CROSSOVER_ACTIONS = [Action.GRAFT_SUBTREE, Action.SWAP_SUBTREES]
ALL_ACTIONS = MUTATION_ACTIONS + CROSSOVER_ACTIONS


# ---------------------------------------------------------------------
# Basic accessors
# ---------------------------------------------------------------------

def next_free_id(G):
    used = [int(n.split("_")[1]) for n in G.nodes]
    return f"module_{(max(used) + 1) if used else 1}"


def free_ports(G, node_id):
    """Ports on `node_id` with no neighbor attached."""
    connectors = G.nodes[node_id]["connectors"]
    return [int(p) for p in PORTS if connectors.get(p) is None and connectors.get(str(p)) is None]


def occupied_ports(G, node_id):
    return [p for p in PORTS if p not in free_ports(G, node_id)]


def growable_ports(G, node_id):
    """Ports on `node_id` available for NEW growth (ADD_NODE, or the
    new_port side of RECONNECT_PORT) - same as free_ports(), except the
    graph's root additionally never offers port 3, which is reserved
    exclusively for the auto-mirrored symmetric half (see symmetry.py's
    build_symmetric_graph - it's the only thing that ever populates
    port 3). Structural queries about what's ACTUALLY attached
    (occupied_ports, or reading .connectors directly) are untouched by
    this - port 3 genuinely IS unoccupied in the genotype until build
    time, this just stops evolution from growing onto it itself."""
    ports = free_ports(G, node_id)
    if is_root(G, node_id) and 3 in ports:
        ports.remove(3)
    return ports


def is_leaf(G, node_id):
    """Degree-1 in the undirected sense (exactly one grammar connection)."""
    return (G.in_degree(node_id) + G.out_degree(node_id)) == 1


def is_root(G, node_id):
    return G.nodes[node_id].get("parent") is None


def root_node(G):
    for n in G.nodes:
        if is_root(G, n):
            return n
    raise ValueError("Graph has no root (node with parent=None)")


def subtree_nodes(G, node_id):
    """`node_id` plus every descendant, following parent/child (edge) direction."""
    seen = {node_id}
    stack = [node_id]
    while stack:
        cur = stack.pop()
        for child in G.successors(cur):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


# ---------------------------------------------------------------------
# Grammar masking rules (per design doc)
# ---------------------------------------------------------------------

def compute_node_action_mask(G, node_id):
    """dict[Action -> bool], True meaning the action IS ALLOWED on this node."""
    n_modules = G.number_of_nodes()
    has_growable_port = len(growable_ports(G, node_id)) > 0
    at_module_limit = n_modules >= MAX_MODULES
    module_type = G.nodes[node_id]["module_type"]

    mask = {
        # Port Availability Mask: needs a growable port to attach to (see
        # growable_ports() - excludes the root's reserved port 3).
        Action.ADD_NODE: has_growable_port and not at_module_limit,
        # Leaf Node Protection Mask + Root Protection Mask.
        Action.DELETE_NODE: is_leaf(G, node_id) and not is_root(G, node_id),
        # Root Protection Mask.
        Action.PRUNE_SUBTREE: not is_root(G, node_id),
        # Root Protection Mask (extended): module_1 (the graph root) always
        # carries the IMU + rangefinder sensor sites, which
        # helper_scripts/mjcf_generator.py mounts on a `bodyRigid_1` element -
        # i.e. it hard-requires the root module to stay "non-foldable".
        Action.MUTATE_FOLD_TYPE: not is_root(G, node_id),
        # Fold Consistency Mask: only foldable modules have a hinge.
        Action.MUTATE_HINGE_ANGLE: module_type != "non-foldable",
        # RECONNECT_PORT needs a growable port to move the connection to,
        # and at least one occupied port to move it from. Port
        # Availability Mask applies to the (growable) target port.
        Action.RECONNECT_PORT: has_growable_port and len(occupied_ports(G, node_id)) > 0,
    }
    return mask
    # NOTE: the "2D Planar Overlap Mask" from the design doc (new module
    # placement must not overlap an existing one) is NOT evaluated here.
    # pattern_gen.py's layout math is specific to the fixed 6-around-1 hex
    # grid the manual editor uses, not to arbitrary RL-grown trees, so a
    # general polygon-overlap check is out of scope for this pass. ADD_NODE
    # / GRAFT_SUBTREE / RECONNECT_PORT are therefore only masked by port
    # availability and the module-count limit for now.


def any_node_allows(G, action):
    return any(compute_node_action_mask(G, n)[action] for n in G.nodes)


def graft_host_eligible(G, node_id):
    """True if `node_id` can host a GRAFT_SUBTREE (same Port Availability /
    Module Count Limit masks as ADD_NODE - attaching a donor subtree is
    grammar-equivalent to attaching a single new node)."""
    return compute_node_action_mask(G, node_id)[Action.ADD_NODE]


def swap_eligible(G, node_id):
    """True if `node_id` can take part in a SWAP_SUBTREES (Root Protection
    Mask: never swap out a graph's root module)."""
    return not is_root(G, node_id)


# ---------------------------------------------------------------------
# Mutation operators - each returns a NEW graph (deep copy), leaving the
# input graph untouched, and raises ValueError if the grammar mask forbids
# the action (callers should check the mask first; this is a safety net).
# ---------------------------------------------------------------------

def _new_node_attrs(module_type, hinge_angle, parent, depth):
    return dict(
        module_type=module_type,
        connectors={1: None, 2: None, 3: None},
        hinge_angle=float(hinge_angle),
        depth=depth,
        parent=parent,
        type_id=MODULE_TYPE_IDS[module_type],
    )


def add_node(G, target_node, port, module_type, hinge_angle=0.0, rng=None):
    if not compute_node_action_mask(G, target_node)[Action.ADD_NODE]:
        raise ValueError(f"ADD_NODE not allowed on {target_node}")
    if port not in growable_ports(G, target_node):
        raise ValueError(f"Port {port} on {target_node} is occupied or reserved")

    # deepcopy, not G.copy(): networkx's shallow copy shares each node's
    # `connectors` dict OBJECT with the original graph, so mutating G2's
    # connectors below would silently corrupt every parent still holding
    # a reference to `G` (e.g. the population list in moo_api.py).
    G2 = copy.deepcopy(G)
    new_id = next_free_id(G2)
    depth = G2.nodes[target_node]["depth"] + 1
    G2.add_node(new_id, id=new_id, **_new_node_attrs(module_type, hinge_angle, target_node, depth))
    # New node's connectors[1] mates back to the parent's chosen port; this
    # mirrors the "connector1 is always the incoming/parent connector"
    # convention used throughout graphs/*.json.
    G2.nodes[new_id]["connectors"][1] = target_node
    G2.nodes[target_node]["connectors"][port] = new_id
    G2.add_edge(target_node, new_id, connector1=port, connector2=1)
    return G2


def delete_node(G, target_node):
    if not compute_node_action_mask(G, target_node)[Action.DELETE_NODE]:
        raise ValueError(f"DELETE_NODE not allowed on {target_node}")
    G2 = copy.deepcopy(G)
    parent = G2.nodes[target_node]["parent"]
    if parent is not None:
        for port, nbr in G2.nodes[parent]["connectors"].items():
            if nbr == target_node:
                G2.nodes[parent]["connectors"][port] = None
    G2.remove_node(target_node)
    return G2


def prune_subtree(G, target_node):
    if not compute_node_action_mask(G, target_node)[Action.PRUNE_SUBTREE]:
        raise ValueError(f"PRUNE_SUBTREE not allowed on {target_node}")
    G2 = copy.deepcopy(G)
    parent = G2.nodes[target_node]["parent"]
    if parent is not None:
        for port, nbr in G2.nodes[parent]["connectors"].items():
            if nbr == target_node:
                G2.nodes[parent]["connectors"][port] = None
    G2.remove_nodes_from(subtree_nodes(G2, target_node))
    return G2


def mutate_fold_type(G, target_node, new_fold_type):
    if new_fold_type not in MODULE_TYPES:
        raise ValueError(f"Unknown fold type {new_fold_type}")
    if not compute_node_action_mask(G, target_node)[Action.MUTATE_FOLD_TYPE]:
        raise ValueError(f"MUTATE_FOLD_TYPE not allowed on {target_node}")
    G2 = copy.deepcopy(G)
    G2.nodes[target_node]["module_type"] = new_fold_type
    G2.nodes[target_node]["type_id"] = MODULE_TYPE_IDS[new_fold_type]
    if new_fold_type == "non-foldable":
        G2.nodes[target_node]["hinge_angle"] = 0.0
    return G2


def mutate_hinge_angle(G, target_node, new_angle):
    if not compute_node_action_mask(G, target_node)[Action.MUTATE_HINGE_ANGLE]:
        raise ValueError(f"MUTATE_HINGE_ANGLE not allowed on {target_node}")
    G2 = copy.deepcopy(G)
    G2.nodes[target_node]["hinge_angle"] = float(min(max(new_angle, MIN_HINGE_ANGLE), MAX_HINGE_ANGLE))
    return G2


def reconnect_port(G, target_node, old_port, new_port):
    if not compute_node_action_mask(G, target_node)[Action.RECONNECT_PORT]:
        raise ValueError(f"RECONNECT_PORT not allowed on {target_node}")
    if old_port not in occupied_ports(G, target_node):
        raise ValueError(f"Port {old_port} on {target_node} is not occupied")
    if new_port not in growable_ports(G, target_node):
        raise ValueError(f"Port {new_port} on {target_node} is occupied or reserved")

    G2 = copy.deepcopy(G)
    nbr = G2.nodes[target_node]["connectors"][old_port]
    G2.nodes[target_node]["connectors"][old_port] = None
    G2.nodes[target_node]["connectors"][new_port] = nbr

    if G2.has_edge(target_node, nbr):
        u, v = target_node, nbr
        d = G2.edges[u, v]
        d["connector1"] = new_port
    else:
        u, v = nbr, target_node
        d = G2.edges[u, v]
        d["connector2"] = new_port
    return G2


# ---------------------------------------------------------------------
# Crossover operators (grammar-legal graph ops; rl_api.py's policy picks
# the donor/partner nodes that get passed in here)
# ---------------------------------------------------------------------

def graft_subtree(host_G, host_node, host_port, donor_G, donor_root, rng=None):
    """Copy donor_G's subtree rooted at donor_root onto host_G at host_node/host_port."""
    rng = rng or random
    if not compute_node_action_mask(host_G, host_node)[Action.ADD_NODE]:
        raise ValueError(f"GRAFT_SUBTREE not allowed on {host_node} (port/limit mask)")

    G2 = copy.deepcopy(host_G)
    donor_nodes = subtree_nodes(donor_G, donor_root)
    remaining_capacity = MAX_MODULES - G2.number_of_nodes()
    if remaining_capacity <= 0:
        raise ValueError("Host graph already at MAX_MODULES")

    id_map = {}
    ordered = sorted(donor_nodes, key=lambda n: donor_G.nodes[n]["depth"])[:remaining_capacity]
    ordered_set = set(ordered)
    for old_id in ordered:
        new_id = next_free_id(G2)
        attrs = donor_G.nodes[old_id]
        G2.add_node(
            new_id, id=new_id,
            module_type=attrs["module_type"],
            connectors={1: None, 2: None, 3: None},
            hinge_angle=attrs["hinge_angle"],
            depth=None, parent=None,
            type_id=attrs["type_id"],
        )
        id_map[old_id] = new_id

    for old_u, old_v, d in donor_G.edges(data=True):
        if old_u in ordered_set and old_v in ordered_set:
            new_u, new_v = id_map[old_u], id_map[old_v]
            G2.add_edge(new_u, new_v, connector1=d["connector1"], connector2=d["connector2"])
            G2.nodes[new_u]["connectors"][d["connector1"]] = new_v
            G2.nodes[new_v]["connectors"][d["connector2"]] = new_u

    new_root = id_map[donor_root]
    G2.nodes[new_root]["connectors"][1] = host_node
    G2.nodes[host_node]["connectors"][host_port] = new_root
    G2.add_edge(host_node, new_root, connector1=host_port, connector2=1)
    G2.nodes[new_root]["parent"] = host_node

    _recompute_depths(G2, root_node(G2))
    return G2


def swap_subtrees(G_a, node_a, G_b, node_b):
    """Exchange the subtrees rooted at node_a (in G_a) and node_b (in G_b).

    Neither node_a nor node_b may be a root (Root Protection Mask covers
    PRUNE_SUBTREE, which this operator is built from).
    Returns (new_G_a, new_G_b).
    """
    if is_root(G_a, node_a) or is_root(G_b, node_b):
        raise ValueError("SWAP_SUBTREES cannot target a root module")

    parent_a, port_a = _parent_and_port(G_a, node_a)
    parent_b, port_b = _parent_and_port(G_b, node_b)

    sub_a = G_a.subgraph(subtree_nodes(G_a, node_a)).copy()
    sub_b = G_b.subgraph(subtree_nodes(G_b, node_b)).copy()

    new_a = prune_subtree(G_a, node_a)
    new_b = prune_subtree(G_b, node_b)

    new_a = graft_subtree(new_a, parent_a, port_a, sub_b, node_b)
    new_b = graft_subtree(new_b, parent_b, port_b, sub_a, node_a)
    return new_a, new_b


def _parent_and_port(G, node_id):
    parent = G.nodes[node_id]["parent"]
    for port, nbr in G.nodes[parent]["connectors"].items():
        if nbr == node_id:
            return parent, int(port)
    raise ValueError(f"Could not find parent port for {node_id}")


def _recompute_depths(G, root):
    G.nodes[root]["depth"] = 0
    G.nodes[root]["parent"] = None
    stack = [root]
    seen = {root}
    while stack:
        cur = stack.pop()
        for child in G.successors(cur):
            if child in seen:
                continue
            seen.add(child)
            G.nodes[child]["depth"] = G.nodes[cur]["depth"] + 1
            G.nodes[child]["parent"] = cur
            stack.append(child)


# ---------------------------------------------------------------------
# Seed graph construction (used by moo_api.py's Sobol-seeded initial pop)
# ---------------------------------------------------------------------

def random_seed_graph(rng, n_modules, module_type_choices=None, hinge_angle_fn=None):
    """Builds a random grammar-legal tree with `n_modules` nodes.

    `module_type_choices`: optional list of MODULE_TYPES values, one per
    node in build order (root first); sampled uniformly if not given.
    `hinge_angle_fn`: optional callable() -> float in [0, 90]; defaults to
    a uniform draw.
    """
    n_modules = max(MIN_MODULES, min(MAX_MODULES, int(n_modules)))
    hinge_angle_fn = hinge_angle_fn or (lambda: rng.uniform(MIN_HINGE_ANGLE, MAX_HINGE_ANGLE))

    def pick_type(i):
        if module_type_choices and i < len(module_type_choices):
            return module_type_choices[i]
        return rng.choice(MODULE_TYPES)

    # module_1 (root) must stay "non-foldable": mjcf_generator.py mounts the
    # IMU + rangefinder sensor sites on a `bodyRigid_1` element, which only
    # exists for a non-foldable module (see compute_node_action_mask's
    # MUTATE_FOLD_TYPE rule for the same constraint during mutation).
    root_type = "non-foldable"
    root_angle = 0.0
    G = nx.DiGraph()
    G.add_node("module_1", id="module_1", **_new_node_attrs(root_type, root_angle, None, 0))

    for i in range(1, n_modules):
        candidates = [n for n in G.nodes if growable_ports(G, n)]
        if not candidates:
            break
        parent = rng.choice(candidates)
        port = rng.choice(growable_ports(G, parent))
        m_type = pick_type(i)
        angle = 0.0 if m_type == "non-foldable" else hinge_angle_fn()
        G = add_node(G, parent, port, m_type, hinge_angle=angle, rng=rng)
    return G
