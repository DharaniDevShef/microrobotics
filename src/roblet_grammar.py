"""
Roblet Grammar - shared genotype schema and grammar-legal graph edit
primitives for the RL-guided NSGA-III evolutionary framework. The genotype
is a NetworkX DiGraph; this module is the single place that reads/writes it.

Design variables:
    1. Module Type (Vi)      - categorical {non-foldable, Mountain fold, valley fold}
    2. Graph Adjacency (Aij) - port-to-port connectivity, ports in {1, 2, 3}
    3. Module Count (N)      - N in [2, 40]
    4. Hinge Angle (theta_i) - continuous, [0, 45] degrees
    5. Light-Sensitive Joint Selection - which foldable modules carry a
       light-sensitive strip (bool per foldable module)
    6. Hinge Angle on Light Detection (theta_light) - continuous, [0, 45]
       degrees, angle a light-sensitive joint moves to when triggered
"""

import copy
import random
from enum import Enum, auto

import networkx as nx

MODULE_TYPES = ["non-foldable", "Mountain fold", "valley fold"]
MODULE_TYPE_IDS = {name: idx for idx, name in enumerate(MODULE_TYPES)}
PORTS = (1, 2, 3)
MIN_MODULES = 2
# Genotypes here are only HALF the final shape - each mirror anchor's port 3
# is reserved for an auto-mirrored clone built at MJCF-generation time, so
# this bounds the half (final shape is up to ~2*MAX_MODULES).
MAX_MODULES = 20
MIN_HINGE_ANGLE = 0.0
MAX_HINGE_ANGLE = 45.0

# Design Variable 4 (theta_i): "uniform" locks every foldable module to the
# same hinge angle; "per_module" restores full per-module independence.
HINGE_ANGLE_MODE = "uniform"  # "uniform" | "per_module"

# Design Variable 5 (light-sensitive joint selection): per-node bool, only
# ever True on a foldable module. Mirrored 1:1 by symmetry.py so a mirrored
# joint pair stays either both sensitive or both not.
#
# Design Variable 6 (hinge_angle_on_light_detection, theta_light): angle a
# light-sensitive joint moves to once triggered. LIGHT_HINGE_ANGLE_MODE
# mirrors HINGE_ANGLE_MODE's uniform/per_module split, scoped to
# light_sensitive nodes.
LIGHT_HINGE_ANGLE_MODE = "uniform"  # "uniform" | "per_module"


class Action(Enum):
    """Grammar actions; rl_api.py's policy selects among all of these
    (mutation AND crossover) from one unified, masked action head."""
    ADD_NODE = auto()
    DELETE_NODE = auto()
    PRUNE_SUBTREE = auto()
    MUTATE_FOLD_TYPE = auto()
    MUTATE_HINGE_ANGLE = auto()
    RECONNECT_PORT = auto()
    TOGGLE_LIGHT_SENSOR = auto()
    MUTATE_LIGHT_HINGE_ANGLE = auto()
    GRAFT_SUBTREE = auto()
    SWAP_SUBTREES = auto()


MUTATION_ACTIONS = [
    Action.ADD_NODE,
    Action.DELETE_NODE,
    Action.PRUNE_SUBTREE,
    Action.MUTATE_FOLD_TYPE,
    Action.MUTATE_HINGE_ANGLE,
    Action.RECONNECT_PORT,
    Action.TOGGLE_LIGHT_SENSOR,
    Action.MUTATE_LIGHT_HINGE_ANGLE,
]

# Crossover needs a second parent graph (donor/partner), unlike mutation.
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


def own_parent_port(G, node_id):
    """Port on `node_id` used for its own edge to its parent, or None for the root."""
    parent = G.nodes[node_id]["parent"]
    if parent is None:
        return None
    if G.has_edge(parent, node_id):
        return G.edges[parent, node_id]["connector2"]
    return G.edges[node_id, parent]["connector1"]


def reconnectable_ports(G, node_id):
    """Occupied ports RECONNECT_PORT may legally move (its `old_port`):
    every occupied port except `node_id`'s own link to its parent, which
    must stay on port 1 for symmetry.py's mirror math to stay valid."""
    parent_port = own_parent_port(G, node_id)
    return [p for p in occupied_ports(G, node_id) if p != parent_port]


def is_mirror_anchor(G, node_id):
    """True for nodes on the bilateral mirror plane: the root, and the
    root's port-1 "spine" child if one has been grown."""
    if is_root(G, node_id):
        return True
    root = root_node(G)
    return G.nodes[root]["connectors"].get(1) == node_id


def growable_ports(G, node_id):
    """Ports on `node_id` available for NEW growth (ADD_NODE, or the new_port
    side of RECONNECT_PORT) - like free_ports(), but a mirror-anchor node
    never offers port 3, which is reserved for its auto-mirrored half."""
    ports = free_ports(G, node_id)
    if is_mirror_anchor(G, node_id) and 3 in ports:
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
        # Needs a growable port to attach to; excludes root's reserved port 3.
        Action.ADD_NODE: has_growable_port and not at_module_limit,
        Action.DELETE_NODE: is_leaf(G, node_id) and not is_root(G, node_id),
        Action.PRUNE_SUBTREE: not is_root(G, node_id),
        # Root always carries the IMU + rangefinder sensor sites, which
        # require it to stay "non-foldable".
        Action.MUTATE_FOLD_TYPE: not is_root(G, node_id),
        Action.MUTATE_HINGE_ANGLE: module_type != "non-foldable",
        Action.TOGGLE_LIGHT_SENSOR: module_type != "non-foldable",
        Action.MUTATE_LIGHT_HINGE_ANGLE: bool(G.nodes[node_id].get("light_sensitive", False)),
        # Root excluded outright: its ports 1/2 have fixed, load-bearing
        # meaning for symmetry.py's mirroring, so neither may be moved.
        Action.RECONNECT_PORT: (
            has_growable_port and not is_root(G, node_id)
            and len(reconnectable_ports(G, node_id)) > 0
        ),
    }
    return mask


def any_node_allows(G, action):
    return any(compute_node_action_mask(G, n)[action] for n in G.nodes)


def graft_host_eligible(G, node_id):
    """True if `node_id` can host a GRAFT_SUBTREE (same masks as ADD_NODE)."""
    return compute_node_action_mask(G, node_id)[Action.ADD_NODE]


def swap_eligible(G, node_id):
    """True if `node_id` can take part in a SWAP_SUBTREES (never the root)."""
    return not is_root(G, node_id)


# ---------------------------------------------------------------------
# Mutation operators - each returns a NEW graph (deep copy), leaving the
# input graph untouched, and raises ValueError if the grammar mask forbids
# the action (callers should check the mask first; this is a safety net).
# ---------------------------------------------------------------------

def _shared_hinge_angle(G, exclude=None):
    """The single hinge angle every foldable module shares under
    HINGE_ANGLE_MODE == "uniform", or None if there isn't one yet."""
    for n in G.nodes:
        if n == exclude:
            continue
        if G.nodes[n]["module_type"] != "non-foldable":
            return G.nodes[n]["hinge_angle"]
    return None


def _broadcast_hinge_angle(G, angle):
    """In place: locks every foldable module in G to `angle`."""
    for n in G.nodes:
        if G.nodes[n]["module_type"] != "non-foldable":
            G.nodes[n]["hinge_angle"] = angle


def _shared_light_hinge_angle(G, exclude=None):
    """The single light-triggered hinge angle every light_sensitive module
    shares under LIGHT_HINGE_ANGLE_MODE == "uniform", or None if none yet."""
    for n in G.nodes:
        if n == exclude:
            continue
        if G.nodes[n].get("light_sensitive", False):
            return G.nodes[n]["light_hinge_angle"]
    return None


def _broadcast_light_hinge_angle(G, angle):
    """In place: locks every currently light_sensitive module in G to `angle`."""
    for n in G.nodes:
        if G.nodes[n].get("light_sensitive", False):
            G.nodes[n]["light_hinge_angle"] = angle


def _new_node_attrs(module_type, hinge_angle, parent, depth, light_sensitive=False, light_hinge_angle=0.0):
    return dict(
        module_type=module_type,
        connectors={1: None, 2: None, 3: None},
        hinge_angle=round(float(hinge_angle), 2),
        light_sensitive=bool(light_sensitive),
        light_hinge_angle=round(float(light_hinge_angle), 2),
        depth=depth,
        parent=parent,
        type_id=MODULE_TYPE_IDS[module_type],
    )


def add_node(G, target_node, port, module_type, hinge_angle=0.0, rng=None):
    if not compute_node_action_mask(G, target_node)[Action.ADD_NODE]:
        raise ValueError(f"ADD_NODE not allowed on {target_node}")
    if port not in growable_ports(G, target_node):
        raise ValueError(f"Port {port} on {target_node} is occupied or reserved")

    # deepcopy, not G.copy(): shallow copy would share the `connectors`
    # dicts with the original graph and corrupt it on mutation.
    G2 = copy.deepcopy(G)
    if HINGE_ANGLE_MODE == "uniform" and module_type != "non-foldable":
        shared = _shared_hinge_angle(G2)
        if shared is not None:
            hinge_angle = shared
    # A new node never starts light_sensitive, but pre-sync its angle to
    # any existing shared trigger angle so a later toggle-on needs no resync.
    light_hinge_angle = 0.0
    if LIGHT_HINGE_ANGLE_MODE == "uniform" and module_type != "non-foldable":
        shared_light = _shared_light_hinge_angle(G2)
        if shared_light is not None:
            light_hinge_angle = shared_light
    new_id = next_free_id(G2)
    depth = G2.nodes[target_node]["depth"] + 1
    G2.add_node(new_id, id=new_id,
                **_new_node_attrs(module_type, hinge_angle, target_node, depth,
                                   light_hinge_angle=light_hinge_angle))
    # connectors[1] is always the incoming/parent connector, by convention.
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
        # A rigid module has no hinge to mount the light-sensitive strip on.
        G2.nodes[target_node]["light_sensitive"] = False
        G2.nodes[target_node]["light_hinge_angle"] = 0.0
    elif HINGE_ANGLE_MODE == "uniform":
        shared = _shared_hinge_angle(G2, exclude=target_node)
        if shared is not None:
            G2.nodes[target_node]["hinge_angle"] = shared
    return G2


def mutate_hinge_angle(G, target_node, new_angle):
    if not compute_node_action_mask(G, target_node)[Action.MUTATE_HINGE_ANGLE]:
        raise ValueError(f"MUTATE_HINGE_ANGLE not allowed on {target_node}")
    G2 = copy.deepcopy(G)
    clipped = round(float(min(max(new_angle, MIN_HINGE_ANGLE), MAX_HINGE_ANGLE)), 2)
    if HINGE_ANGLE_MODE == "uniform":
        _broadcast_hinge_angle(G2, clipped)
    else:
        G2.nodes[target_node]["hinge_angle"] = clipped
    return G2


def toggle_light_sensor(G, target_node):
    """Flips whether `target_node` carries the light-sensitive strip
    (Design Variable 5). Turning one on syncs it to the assembly's
    existing shared light_hinge_angle, if one is already established."""
    if not compute_node_action_mask(G, target_node)[Action.TOGGLE_LIGHT_SENSOR]:
        raise ValueError(f"TOGGLE_LIGHT_SENSOR not allowed on {target_node}")
    G2 = copy.deepcopy(G)
    turning_on = not G2.nodes[target_node].get("light_sensitive", False)
    G2.nodes[target_node]["light_sensitive"] = turning_on
    if turning_on and LIGHT_HINGE_ANGLE_MODE == "uniform":
        shared = _shared_light_hinge_angle(G2, exclude=target_node)
        if shared is not None:
            G2.nodes[target_node]["light_hinge_angle"] = shared
    return G2


def mutate_light_hinge_angle(G, target_node, new_angle):
    """Mutates Design Variable 6 (hinge_angle_on_light_detection). Free to
    land above or below the module's baseline hinge_angle - that's what
    governs attracted-vs-repulsive response, not enforced here."""
    if not compute_node_action_mask(G, target_node)[Action.MUTATE_LIGHT_HINGE_ANGLE]:
        raise ValueError(f"MUTATE_LIGHT_HINGE_ANGLE not allowed on {target_node}")
    G2 = copy.deepcopy(G)
    clipped = round(float(min(max(new_angle, MIN_HINGE_ANGLE), MAX_HINGE_ANGLE)), 2)
    if LIGHT_HINGE_ANGLE_MODE == "uniform":
        _broadcast_light_hinge_angle(G2, clipped)
    else:
        G2.nodes[target_node]["light_hinge_angle"] = clipped
    return G2


def ensure_min_light_sensitive(G, rng):
    """In place: guarantees at least one foldable module in G carries the
    light-sensitive strip, whenever G has any foldable module at all.
    No-op if one already exists or G has none foldable."""
    if any(G.nodes[n].get("light_sensitive", False) for n in G.nodes):
        return G
    foldable = [n for n in G.nodes if G.nodes[n]["module_type"] != "non-foldable"]
    if not foldable:
        return G
    chosen = rng.choice(foldable)
    G.nodes[chosen]["light_sensitive"] = True
    G.nodes[chosen]["light_hinge_angle"] = round(float(rng.uniform(MIN_HINGE_ANGLE, MAX_HINGE_ANGLE)), 2)
    return G


def reconnect_port(G, target_node, old_port, new_port):
    if not compute_node_action_mask(G, target_node)[Action.RECONNECT_PORT]:
        raise ValueError(f"RECONNECT_PORT not allowed on {target_node}")
    if old_port not in reconnectable_ports(G, target_node):
        raise ValueError(
            f"Port {old_port} on {target_node} is not occupied, or is its own "
            "link to its parent (reconnectable_ports() never offers that one - "
            "see its docstring)"
        )
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

class GraftPortConflict(ValueError):
    """Raised when donor_root's own port 1 is already used by one of its
    real (copied) children, so graft_subtree can't also point it at the
    new host. Callers should reject the graft and retry with a different
    action, not crash."""


def graft_subtree(host_G, host_node, host_port, donor_G, donor_root, rng=None):
    """Copy donor_G's subtree rooted at donor_root onto host_G at
    host_node/host_port. Raises GraftPortConflict if donor_root is
    donor_G's own root and its port 1 is already used by a copied child
    (port 1 must stay the parent-link, per convention)."""
    rng = rng or random
    if not compute_node_action_mask(host_G, host_node)[Action.ADD_NODE]:
        raise ValueError(f"GRAFT_SUBTREE not allowed on {host_node} (port/limit mask)")

    G2 = copy.deepcopy(host_G)
    host_shared_angle = _shared_hinge_angle(G2) if HINGE_ANGLE_MODE == "uniform" else None
    host_shared_light_angle = _shared_light_hinge_angle(G2) if LIGHT_HINGE_ANGLE_MODE == "uniform" else None
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
            light_sensitive=attrs.get("light_sensitive", False),
            light_hinge_angle=attrs.get("light_hinge_angle", 0.0),
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
    if G2.nodes[new_root]["connectors"].get(1) is not None:
        raise GraftPortConflict(
            f"Donor root {donor_root}'s port 1 is already used by one of its "
            "own copied children (it must itself be donor_G's root, with no "
            "external parent edge to exclude it) - can't also attach it to "
            "the host via port 1 without breaking the port-1-is-always-the-"
            "parent-link convention. See GraftPortConflict's docstring."
        )
    G2.nodes[new_root]["connectors"][1] = host_node
    G2.nodes[host_node]["connectors"][host_port] = new_root
    G2.add_edge(host_node, new_root, connector1=host_port, connector2=1)
    G2.nodes[new_root]["parent"] = host_node

    _recompute_depths(G2, root_node(G2))
    if HINGE_ANGLE_MODE == "uniform":
        # Prefer the host's pre-existing shared angle over the donor's, so
        # grafting doesn't silently change the rest of the host.
        target = host_shared_angle if host_shared_angle is not None else _shared_hinge_angle(G2)
        if target is not None:
            _broadcast_hinge_angle(G2, target)
    if LIGHT_HINGE_ANGLE_MODE == "uniform":
        # Same reconciliation as hinge_angle, scoped to light_sensitive nodes.
        light_target = host_shared_light_angle if host_shared_light_angle is not None else _shared_light_hinge_angle(G2)
        if light_target is not None:
            _broadcast_light_hinge_angle(G2, light_target)
    return G2


def swap_subtrees(G_a, node_a, G_b, node_b):
    """Exchange the subtrees rooted at node_a (in G_a) and node_b (in G_b).
    Neither may be a root. Returns (new_G_a, new_G_b)."""
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

def random_seed_graph(rng, n_modules, module_type_choices=None, hinge_angle_fn=None,
                       light_sensitive_fn=None, light_hinge_angle_fn=None):
    """Builds a random grammar-legal tree with `n_modules` nodes.

    `module_type_choices`: optional list of MODULE_TYPES, one per node in
    build order; sampled uniformly if not given. `hinge_angle_fn`,
    `light_sensitive_fn`, `light_hinge_angle_fn`: optional callables to
    override the default uniform-draw / fair-coin-flip sampling for
    hinge angle, light sensitivity, and light-trigger angle."""
    n_modules = max(MIN_MODULES, min(MAX_MODULES, int(n_modules)))
    hinge_angle_fn = hinge_angle_fn or (lambda: rng.uniform(MIN_HINGE_ANGLE, MAX_HINGE_ANGLE))
    light_sensitive_fn = light_sensitive_fn or (lambda: rng.random() < 0.5)
    light_hinge_angle_fn = light_hinge_angle_fn or (lambda: rng.uniform(MIN_HINGE_ANGLE, MAX_HINGE_ANGLE))

    def pick_type(i):
        if module_type_choices and i < len(module_type_choices):
            return module_type_choices[i]
        return rng.choice(MODULE_TYPES)

    # module_1 (root) must stay "non-foldable": it carries the IMU +
    # rangefinder sensor sites.
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

    # Design Variables 5/6: roll light-sensitivity per foldable module, then
    # (if uniform mode) pick one shared trigger angle for the selected ones.
    foldable = [n for n in G.nodes if G.nodes[n]["module_type"] != "non-foldable"]
    selected = [n for n in foldable if light_sensitive_fn()]
    if foldable and not selected:
        # A seed with a foldable module and zero light-sensitive joints is
        # not a valid genotype (see ensure_min_light_sensitive).
        selected = [rng.choice(foldable)]
    if selected:
        if LIGHT_HINGE_ANGLE_MODE == "uniform":
            shared_light_angle = round(float(light_hinge_angle_fn()), 2)
            for n in selected:
                G.nodes[n]["light_sensitive"] = True
                G.nodes[n]["light_hinge_angle"] = shared_light_angle
        else:
            for n in selected:
                G.nodes[n]["light_sensitive"] = True
                G.nodes[n]["light_hinge_angle"] = round(float(light_hinge_angle_fn()), 2)
    return G
