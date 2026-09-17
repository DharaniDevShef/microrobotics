"""
Symmetry - mirrors a half-genotype graph into the full bilaterally
symmetric shape, at build time only (every evolved morphology must be
symmetric). Evolution (moo_api/rl_api/roblet_grammar) only ever works on
the half-graph; moo_api.py calls build_symmetric_graph() right before
turning a genotype into an MJCF assembly.

A "mirror anchor" (root, and the root's port-1 spine child if grown) sits
exactly on the mirror plane: its port 2 is the evolvable side, and its
port 3 is reserved for build_symmetric_graph()'s mirrored clone of port 2.
"""

import copy

import roblet_grammar as rg

# A rotation about the mirror plane's normal axis (every module's hinge
# axis) is unchanged by reflection, so a mirrored module keeps the same
# fold type/hinge sign as the original - not flipped.
_MIRROR_FOLD_TYPE = {
    "non-foldable": "non-foldable",
    "Mountain fold": "Mountain fold",
    "valley fold": "valley fold",
}


def mirror_fold_type(module_type):
    return _MIRROR_FOLD_TYPE[module_type]


def _mirror_port(port):
    """Port 1 stays fixed (on the mirror axis); ports 2 and 3 swap."""
    return 5 - port if port in (2, 3) else port


class MirrorAnchorViolation(ValueError):
    """Raised when a mirror anchor's port 3 is already occupied (e.g. by a
    GRAFT_SUBTREE/SWAP_SUBTREES that landed a donor node with its own port 3
    already in use). Treat like mjcf_generator.ModuleCollisionError: an
    invalid genotype to reject, not a crash."""


def _mirror_anchor_port2(full_G, half_G, anchor):
    """In place on full_G: clone whatever's on `anchor`'s port 2 (read from
    half_G) as a mirror image attached to `anchor`'s port 3. No-op if port 2
    is empty. `anchor` must be a mirror anchor (root or its port-1 child)."""
    half_root_id = half_G.nodes[anchor]["connectors"].get(2)
    if half_root_id is None:
        return
    if full_G.nodes[anchor]["connectors"].get(3) is not None:
        raise MirrorAnchorViolation(
            f"{anchor}'s port 3 is already occupied - can't auto-mirror onto "
            "it. Likely a GRAFT_SUBTREE/SWAP_SUBTREES that landed a donor "
            "node (with its own port 3 already in use) onto a host position "
            "that made it a new mirror anchor - see MirrorAnchorViolation's "
            "docstring."
        )

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
            light_sensitive=attrs.get("light_sensitive", False),
            light_hinge_angle=attrs.get("light_hinge_angle", 0.0),
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

    # Read the real port from the edge rather than assuming 1 (RECONNECT_PORT may have moved it).
    original_child_port = half_G.edges[anchor, half_root_id]["connector2"]
    mirrored_child_port = _mirror_port(original_child_port)

    mirrored_root = id_map[half_root_id]
    full_G.nodes[mirrored_root]["connectors"][mirrored_child_port] = anchor
    full_G.nodes[anchor]["connectors"][3] = mirrored_root
    full_G.add_edge(anchor, mirrored_root, connector1=3, connector2=mirrored_child_port)
    full_G.nodes[mirrored_root]["parent"] = anchor


def build_symmetric_graph(half_G):
    """Returns a NEW graph: half_G plus a mirrored clone of port 2's subtree
    on each mirror anchor (root, and its port-1 spine child if grown),
    attached to that anchor's port 3. half_G is left untouched; a no-op if
    both anchors' port 2s are empty."""
    root = rg.root_node(half_G)
    full_G = copy.deepcopy(half_G)

    _mirror_anchor_port2(full_G, half_G, root)
    spine_child = half_G.nodes[root]["connectors"].get(1)
    if spine_child is not None:
        _mirror_anchor_port2(full_G, half_G, spine_child)

    rg._recompute_depths(full_G, root)
    return full_G
