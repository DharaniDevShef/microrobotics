"""
Symmetry - mirrors a half-genotype graph into the full bilaterally
symmetric shape, at build time only.

Per the professor's requirement, every evolved morphology must be
bilaterally symmetric (cut it in half, the two halves match). Rather than
keeping the full graph as genotype and trying to keep both halves in sync
through every mutation/crossover (fragile - crossover in particular has
no natural "mirror partner" from the other parent), the genotype evolved
throughout moo_api/rl_api/roblet_grammar is only HALF the final shape:

    - module_1 (root)'s port 1 is the free "spine" direction - on the
      mirror axis, grows normally, no partner needed.
    - port 2 is the one evolvable "half" - mutation/crossover/RL only
      ever touch this side.
    - port 3 is reserved - roblet_grammar.growable_ports() never offers
      it to ADD_NODE/RECONNECT_PORT. build_symmetric_graph() below is the
      ONLY thing that ever populates it: a mirrored clone of whatever is
      on port 2.

This keeps every existing grammar/mutation/crossover/RL operator
completely unchanged - they only ever see the half-graph. Symmetry is
enforced at exactly one boundary: called from moo_api.py right before a
genotype becomes an MJCF assembly (both for the real build and for the
collision-free check - a mirrored pair can collide with ITSELF even when
the half alone doesn't, so the collision check must see the full graph too).
"""

import copy

import roblet_grammar as rg

# A mirrored fold reverses direction: a valley fold on one side becomes a
# mountain fold on the other (and vice versa). A rigid module has no fold
# direction, so it mirrors to itself.
_MIRROR_FOLD_TYPE = {
    "non-foldable": "non-foldable",
    "Mountain fold": "valley fold",
    "valley fold": "Mountain fold",
}


def mirror_fold_type(module_type):
    return _MIRROR_FOLD_TYPE[module_type]


def _mirror_port(port):
    """Port 1 (the parent-link direction) sits on the mirror axis and
    never changes; ports 2 and 3 are built as literal mirror images of
    each other (mjcf_generator.py rotates them +120deg/-120deg from port
    1), so mirroring swaps them."""
    return 5 - port if port in (2, 3) else port


def build_symmetric_graph(half_G):
    """Returns a NEW graph: half_G plus a mirrored clone of whatever is
    attached to the root's port 2, attached to the root's port 3.
    half_G itself is left untouched. A no-op (just a copy of half_G) if
    port 2 is empty.

    Deliberately does NOT go through roblet_grammar.graft_subtree /
    compute_node_action_mask / MAX_MODULES: those gate EVOLUTION-time
    growth of the half (capped so the final mirrored shape stays in
    budget - see MAX_MODULES's docstring), but this is a deterministic,
    always-legal build-time step applied to an already-valid half, not a
    mutation, so it must not be capped by the same (now much smaller,
    half-sized) ceiling.
    """
    root = rg.root_node(half_G)
    half_root_id = half_G.nodes[root]["connectors"].get(2)
    full_G = copy.deepcopy(half_G)
    if half_root_id is None:
        return full_G

    donor_nodes = rg.subtree_nodes(half_G, half_root_id)
    id_map = {}
    for old_id in sorted(donor_nodes, key=lambda n: half_G.nodes[n]["depth"]):
        new_id = rg.next_free_id(full_G)
        attrs = half_G.nodes[old_id]
        mirrored_type = mirror_fold_type(attrs["module_type"])
        full_G.add_node(
            new_id, id=new_id,
            module_type=mirrored_type,
            connectors={1: None, 2: None, 3: None},
            hinge_angle=attrs["hinge_angle"],
            depth=None, parent=None,
            type_id=rg.MODULE_TYPE_IDS[mirrored_type],
        )
        id_map[old_id] = new_id

    for old_u, old_v, d in half_G.edges(data=True):
        if old_u in donor_nodes and old_v in donor_nodes:
            new_u, new_v = id_map[old_u], id_map[old_v]
            c1, c2 = _mirror_port(d["connector1"]), _mirror_port(d["connector2"])
            full_G.add_edge(new_u, new_v, connector1=c1, connector2=c2)
            full_G.nodes[new_u]["connectors"][c1] = new_v
            full_G.nodes[new_v]["connectors"][c2] = new_u

    # Which port half_root_id itself uses to connect back to root is
    # normally 1 (add_node always wires a fresh child's port 1 to its
    # parent), but RECONNECT_PORT could in principle have moved it since -
    # so read the real port from the original edge and mirror THAT,
    # rather than assuming 1.
    original_child_port = half_G.edges[root, half_root_id]["connector2"]
    mirrored_child_port = _mirror_port(original_child_port)

    mirrored_root = id_map[half_root_id]
    full_G.nodes[mirrored_root]["connectors"][mirrored_child_port] = root
    full_G.nodes[root]["connectors"][3] = mirrored_root
    full_G.add_edge(root, mirrored_root, connector1=3, connector2=mirrored_child_port)
    full_G.nodes[mirrored_root]["parent"] = root

    rg._recompute_depths(full_G, root)
    return full_G
