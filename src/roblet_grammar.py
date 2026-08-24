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
    5. Light-Sensitive Joint Selection - which foldable modules carry a
       light-sensitive on their own hinge (bool per foldable
       module - see TOGGLE_LIGHT_SENSOR/toggle_light_sensor below)
    6. Hinge Angle on Light Detection (theta_light) - continuous, [0, 45]
       degrees, the angle a light-sensitive joint moves to once ITS OWN
       sensor is triggered (see MUTATE_LIGHT_HINGE_ANGLE/
       mutate_light_hinge_angle below)
"""

import copy
import random
from enum import Enum, auto

import networkx as nx

MODULE_TYPES = ["non-foldable", "Mountain fold", "valley fold"]
MODULE_TYPE_IDS = {name: idx for idx, name in enumerate(MODULE_TYPES)}
PORTS = (1, 2, 3)
MIN_MODULES = 2
# Every genotype evolved here is only HALF the final shape - each mirror
# anchor's (is_mirror_anchor() below - the root, and its port-1 "spine"
# child if grown) port 3 is reserved for an auto-mirrored clone of
# whatever grows on its port 2 (see growable_ports() and symmetry.py's
# build_symmetric_graph), built only at MJCF-generation time. So this
# bounds the HALF, not the final module count: a half of MAX_MODULES
# yields a final shape of up to roughly 2*MAX_MODULES modules (the two
# anchors shared, everything else mirrored in pairs) - kept at 20 here so
# that stays within the design doc's original N in [2, 40] for the
# physical robot.
MAX_MODULES = 20
MIN_HINGE_ANGLE = 0.0
MAX_HINGE_ANGLE = 45.0

# Hinge Angle mode for Design Variable 4 (theta_i, see module docstring
# above). "uniform" (default): every foldable module in an assembly is
# locked to the SAME hinge angle - mutating one, or adding a new foldable
# module, moves/sets them all together, so the RL policy / Sobol seeding
# effectively controls one shared design variable instead of one per
# foldable module. Flip to "per_module" to restore full per-module
# independence for future use - the per-node hinge_angle field and all the
# machinery around it never goes away, this flag just decides whether
# add_node/mutate_hinge_angle/mutate_fold_type/graft_subtree fan a single
# value out to every foldable node or touch just the one node involved.
HINGE_ANGLE_MODE = "uniform"  # "uniform" | "per_module"

# Design Variable 5 (light-sensitive joint selection, per the pheromone-
# response design): each foldable module's joint physically doubles as its
# own light-sensitive PVC strip or not - `light_sensitive` (bool). This is
# a per-node SELECTION variable (which foldable joints get the light-
# sensitive hinge), never broadcast the way hinge_angle is - the whole
# point is that it can vary joint-to-joint. Only ever True on a foldable
# module (module_type != "non-foldable") - a rigid module has no hinge to
# mount the strip on. symmetry.py's build_symmetric_graph mirrors this
# field 1:1 alongside module_type/hinge_angle, so a mirrored joint pair is
# always either both sensitive or both not - the "symmetry API" that keeps
# any selection here even and bilaterally symmetric for free.
#
# Design Variable 6 (hinge_angle_on_light_detection, theta_light) - the
# angle a light-sensitive joint moves to once ITS OWN sensor is triggered
# (continuous, same [0, 45] range as theta_i). LIGHT_HINGE_ANGLE_MODE below
# mirrors HINGE_ANGLE_MODE's uniform/per_module split, but scoped to just
# the light_sensitive nodes: under "uniform", every light-sensitive joint
# in an assembly shares the one evolved trigger angle. Whether that shared
# angle sits above or below the joint's own (also shared, under
# HINGE_ANGLE_MODE) baseline hinge_angle is what determines attracted vs.
# repulsive turning/speed response - see objectives_api.configure_pheromone_response.
LIGHT_HINGE_ANGLE_MODE = "uniform"  # "uniform" | "per_module"


class Action(Enum):
    """Grammar actions. rl_api.py's policy selects among ALL of these
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


def own_parent_port(G, node_id):
    """The port on `node_id` itself used for its own edge to its parent,
    or None for the root. Reads the real edge rather than assuming port 1
    - every node SHOULD use port 1 for this by convention (add_node
    always does), but this is also what reconnectable_ports() uses to
    make sure that stays true, so it can't just assume its own
    conclusion."""
    parent = G.nodes[node_id]["parent"]
    if parent is None:
        return None
    if G.has_edge(parent, node_id):
        return G.edges[parent, node_id]["connector2"]
    return G.edges[node_id, parent]["connector1"]


def reconnectable_ports(G, node_id):
    """Occupied ports RECONNECT_PORT may legally move (its `old_port`):
    every occupied port except `node_id`'s own link to its parent.

    That link has to stay wherever it is. Port 1 is, by convention
    relied on throughout this module (add_node) and by symmetry.py's
    mirror math (only port 1's geometry is symmetric under reflection -
    ports 2/3 are mirror images of EACH OTHER, never individually "on
    axis"), always the parent-facing port - move a node's own parent
    link off port 1 and any subtree hanging under it can no longer be
    validly mirrored later, plus it desyncs is_mirror_anchor() (which
    reads the ROOT's port 1 specifically to find its spine child) the
    moment it happens to the root's spine child itself."""
    parent_port = own_parent_port(G, node_id)
    return [p for p in occupied_ports(G, node_id) if p != parent_port]


def is_mirror_anchor(G, node_id):
    """True for every node that sits exactly ON the bilateral mirror plane:
    the root, and the root's port-1 child if one has been grown (the
    "spine" - see the module docstring in symmetry.py). Port 1 is only
    ever free on the root itself (every other node's port 1 is already
    spoken for, linking back to its own parent), so the spine can never
    extend past that one child - anything grown from THAT child's port 2
    or 3 has already rotated off the mirror plane and is an ordinary
    (non-anchor) node whose whole subtree gets reflected as a block by
    build_symmetric_graph, same as the root's port-2 subtree is."""
    if is_root(G, node_id):
        return True
    root = root_node(G)
    return G.nodes[root]["connectors"].get(1) == node_id


def growable_ports(G, node_id):
    """Ports on `node_id` available for NEW growth (ADD_NODE, or the
    new_port side of RECONNECT_PORT) - same as free_ports(), except a
    mirror-anchor node (is_mirror_anchor() - the root, and its port-1
    child if any) additionally never offers port 3, which is reserved
    exclusively for its auto-mirrored symmetric half (see symmetry.py's
    build_symmetric_graph - it's the only thing that ever populates
    an anchor's port 3). Structural queries about what's ACTUALLY attached
    (occupied_ports, or reading .connectors directly) are untouched by
    this - port 3 genuinely IS unoccupied in the genotype until build
    time, this just stops evolution from growing onto it itself."""
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
        # Fold Consistency Mask: the light-sensitive PVC strip is mounted
        # on the joint itself, so only a foldable module can carry one.
        Action.TOGGLE_LIGHT_SENSOR: module_type != "non-foldable",
        # Only mutate the shared light-triggered angle on a node that
        # currently carries the sensor (TOGGLE_LIGHT_SENSOR is what turns
        # this on in the first place).
        Action.MUTATE_LIGHT_HINGE_ANGLE: bool(G.nodes[node_id].get("light_sensitive", False)),
        # RECONNECT_PORT needs a growable port to move the connection to,
        # and at least one RECONNECTABLE port to move it from - occupied,
        # but not the node's own link to its parent (reconnectable_ports -
        # that port has to stay put, see its docstring). Root Protection
        # Mask (extended): the root has no parent link to protect that
        # way, but its two occupied ports (port 1 "spine", port 2
        # "evolvable") carry the same kind of fixed, load-bearing meaning
        # for is_mirror_anchor()/symmetry.py - relabeling which is which
        # would silently swap which subtree does and doesn't get
        # auto-mirrored, so the root is excluded outright rather than
        # trying to say "keep whichever of 1/2 you already have".
        Action.RECONNECT_PORT: (
            has_growable_port and not is_root(G, node_id)
            and len(reconnectable_ports(G, node_id)) > 0
        ),
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

def _shared_hinge_angle(G, exclude=None):
    """The single hinge angle every foldable module shares under
    HINGE_ANGLE_MODE == "uniform" - whichever foldable module (other than
    `exclude`, if given) comes first, or None if there isn't one yet."""
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
    shares under LIGHT_HINGE_ANGLE_MODE == "uniform" - whichever
    light_sensitive module (other than `exclude`, if given) comes first, or
    None if there isn't one yet."""
    for n in G.nodes:
        if n == exclude:
            continue
        if G.nodes[n].get("light_sensitive", False):
            return G.nodes[n]["light_hinge_angle"]
    return None


def _broadcast_light_hinge_angle(G, angle):
    """In place: locks every CURRENTLY light_sensitive module in G to
    `angle`. Non-sensitive modules are left alone (their light_hinge_angle
    is meaningless until TOGGLE_LIGHT_SENSOR turns them on)."""
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

    # deepcopy, not G.copy(): networkx's shallow copy shares each node's
    # `connectors` dict OBJECT with the original graph, so mutating G2's
    # connectors below would silently corrupt every parent still holding
    # a reference to `G` (e.g. the population list in moo_api.py).
    G2 = copy.deepcopy(G)
    if HINGE_ANGLE_MODE == "uniform" and module_type != "non-foldable":
        shared = _shared_hinge_angle(G2)
        if shared is not None:
            hinge_angle = shared
    # A new node never starts light_sensitive (TOGGLE_LIGHT_SENSOR is the
    # only thing that turns it on - see that function's docstring), but if
    # it's foldable and the assembly already has a shared light-triggered
    # angle, pre-sync it so toggling this node on later doesn't need a
    # separate resync.
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
        # A rigid module has no hinge to mount the light-sensitive PVC
        # strip on (see TOGGLE_LIGHT_SENSOR's Fold Consistency Mask) - drop
        # it here too so a MUTATE_FOLD_TYPE away from foldable can't leave
        # a "light_sensitive" module with no joint behind it.
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
    """Flips whether `target_node` carries the light-sensitive PVC strip
    (Design Variable 5 - see the module docstring). Since the sensor and
    the joint it actuates are the same physical hinge (per the hardware
    model), this is the ONLY way a module gains or loses a light sensor -
    ADD_NODE never creates one directly. Turning one on syncs it to the
    assembly's existing shared light_hinge_angle (LIGHT_HINGE_ANGLE_MODE
    == "uniform"), if one is already established, so the new sensor
    reacts consistently with every other one instead of at a stale 0.0."""
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
    """Mutates Design Variable 6 (hinge_angle_on_light_detection). Whether
    the result lands above or below the module's own (shared) baseline
    hinge_angle is what governs attracted-vs-repulsive behavior - see
    objectives_api.configure_pheromone_response - so this is intentionally
    left free to land on either side; picking a run's optimization
    direction (main.py's PHEROMONE_RESPONSE_TYPE) is what steers evolution
    to one side or the other, not a constraint enforced here."""
    if not compute_node_action_mask(G, target_node)[Action.MUTATE_LIGHT_HINGE_ANGLE]:
        raise ValueError(f"MUTATE_LIGHT_HINGE_ANGLE not allowed on {target_node}")
    G2 = copy.deepcopy(G)
    clipped = round(float(min(max(new_angle, MIN_HINGE_ANGLE), MAX_HINGE_ANGLE)), 2)
    if LIGHT_HINGE_ANGLE_MODE == "uniform":
        _broadcast_light_hinge_angle(G2, clipped)
    else:
        G2.nodes[target_node]["light_hinge_angle"] = clipped
    return G2


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
    new host - see graft_subtree's docstring. Callers should treat this
    like any other invalid-genotype rejection (mjcf_generator.
    ModuleCollisionError, symmetry.MirrorAnchorViolation): reject the
    graft and let the caller retry with a different action, not crash."""


def graft_subtree(host_G, host_node, host_port, donor_G, donor_root, rng=None):
    """Copy donor_G's subtree rooted at donor_root onto host_G at host_node/host_port.

    donor_root itself is free to be donor_G's own root (or any other
    node) - rl_api.py's crossover donor pick has no restriction against
    it. That case needs care: every node's port 1 is, by strong
    convention relied on throughout this module (see add_node) and by
    symmetry.py's mirror math (only port 1's geometry is self-symmetric
    under reflection - ports 2/3 are only mirror images of EACH OTHER,
    never individually "on axis"), always the link back to its parent.
    An ordinary donor_root has port 1 free in the copy (its real parent
    lives outside the copied subtree, so the edge-copying loop below
    never touches port 1), so pointing it at the new host preserves that
    convention for free. But donor_G's OWN root has no parent edge to
    exclude that way - its port 1 may already be a real internal child
    (e.g. its own "spine"), which the edge-copying loop faithfully
    carries over. Forcing port 1 to the new host in that case would
    require either clobbering that already-set connectors-dict entry
    (silently splitting the port between two different neighbors
    depending whether you trust connectors or the graph edges - exactly
    the corruption this used to produce) or attaching via a different
    port (which breaks the "port 1 = parent" convention this graph
    leans on elsewhere). Neither is safe, so this specific graft is
    rejected instead - see GraftPortConflict."""
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
        # Reconcile: the donor subtree may have carried its own
        # (independently-uniform) angle that differs from the host's -
        # prefer the host's pre-existing shared angle so grafting doesn't
        # silently change the rest of the host; only fall back to
        # whatever the donor contributed if the host had no foldable
        # module of its own yet.
        target = host_shared_angle if host_shared_angle is not None else _shared_hinge_angle(G2)
        if target is not None:
            _broadcast_hinge_angle(G2, target)
    if LIGHT_HINGE_ANGLE_MODE == "uniform":
        # Same reconciliation as hinge_angle just above, scoped to whatever
        # light_sensitive nodes ended up in the merged graph (host's own,
        # the donor's copied-over ones, or both).
        light_target = host_shared_light_angle if host_shared_light_angle is not None else _shared_light_hinge_angle(G2)
        if light_target is not None:
            _broadcast_light_hinge_angle(G2, light_target)
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

def random_seed_graph(rng, n_modules, module_type_choices=None, hinge_angle_fn=None,
                       light_sensitive_fn=None, light_hinge_angle_fn=None):
    """Builds a random grammar-legal tree with `n_modules` nodes.

    `module_type_choices`: optional list of MODULE_TYPES values, one per
    node in build order (root first); sampled uniformly if not given.
    `hinge_angle_fn`: optional callable() -> float in [0, 90]; defaults to
    a uniform draw.
    `light_sensitive_fn`: optional callable() -> bool, rolled once per
    foldable module to decide Design Variable 5 (which foldable joints get
    a light-sensitive PVC strip); defaults to a fair coin flip, so a Sobol-
    seeded initial population still gets a genuine spread over this
    variable instead of starting with zero sensors everywhere.
    `light_hinge_angle_fn`: optional callable() -> float in [0, 45] for
    Design Variable 6 (hinge_angle_on_light_detection), drawn once and
    shared by every module light_sensitive_fn() selected (see
    LIGHT_HINGE_ANGLE_MODE's docstring); defaults to a uniform draw.
    """
    n_modules = max(MIN_MODULES, min(MAX_MODULES, int(n_modules)))
    hinge_angle_fn = hinge_angle_fn or (lambda: rng.uniform(MIN_HINGE_ANGLE, MAX_HINGE_ANGLE))
    light_sensitive_fn = light_sensitive_fn or (lambda: rng.random() < 0.5)
    light_hinge_angle_fn = light_hinge_angle_fn or (lambda: rng.uniform(MIN_HINGE_ANGLE, MAX_HINGE_ANGLE))

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

    # Design Variables 5/6: roll light-sensitivity per foldable module,
    # then (if LIGHT_HINGE_ANGLE_MODE == "uniform") pick ONE shared
    # trigger angle for every module that came up sensitive - same
    # uniform-broadcast pattern as hinge_angle itself.
    foldable = [n for n in G.nodes if G.nodes[n]["module_type"] != "non-foldable"]
    selected = [n for n in foldable if light_sensitive_fn()]
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
