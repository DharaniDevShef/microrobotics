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
      mirror axis, grows normally, no partner needed. Port 1 is only ever
      free on the root (every other node's port 1 already links back to
      its own parent), so the spine can be at most one node deep: the
      root, and the root's port-1 child if grown. Both are "mirror
      anchors" (roblet_grammar.is_mirror_anchor()) - they're the only two
      nodes whose accumulated orientation keeps them exactly on the
      mirror plane.
    - each anchor's port 2 is the one evolvable side - mutation/
      crossover/RL only ever touch this side (and everything below it,
      once it's rotated off the mirror plane).
    - each anchor's port 3 is reserved - roblet_grammar.growable_ports()
      never offers it to ADD_NODE/RECONNECT_PORT. build_symmetric_graph()
      below is the ONLY thing that ever populates it: a mirrored clone of
      whatever is on that anchor's port 2.

This keeps every existing grammar/mutation/crossover/RL operator
completely unchanged - they only ever see the half-graph. Symmetry is
enforced at exactly one boundary: called from moo_api.py right before a
genotype becomes an MJCF assembly (both for the real build and for the
collision-free check - a mirrored pair can collide with ITSELF even when
the half alone doesn't, so the collision check must see the full graph too).
"""

import copy

import roblet_grammar as rg

# Every module's hinge axis is local X - the same axis the left/right
# mirror plane (build_symmetric_graph negates X, ports 2/3 sit at +-120deg
# from it) is normal to. A rotation about the mirror plane's own normal
# axis is UNCHANGED by reflection (only rotations about the two in-plane
# axes flip sign - see e.g. a right-handed screw viewed in a mirror held
# perpendicular to its shaft: it still turns the same way), so a mirrored
# module needs the SAME fold type and the SAME hinge_angle sign as the
# original, not a flipped one. mjcf_generator.py's per-type hinge geometry
# (joint_z, joint_range) is a fixed hardware offset keyed only by
# module_type, unrelated to left/right - swapping valley<->Mountain here
# would relocate the hinge line itself rather than mirror it, breaking the
# folded (though not the flat) pose's symmetry. A rigid module has no fold
# direction either way, so it mirrors to itself too.
_MIRROR_FOLD_TYPE = {
    "non-foldable": "non-foldable",
    "Mountain fold": "Mountain fold",
    "valley fold": "valley fold",
}


def mirror_fold_type(module_type):
    return _MIRROR_FOLD_TYPE[module_type]


def _mirror_port(port):
    """Port 1 (the parent-link direction) sits on the mirror axis and
    never changes; ports 2 and 3 are built as literal mirror images of
    each other (mjcf_generator.py rotates them +120deg/-120deg from port
    1), so mirroring swaps them."""
    return 5 - port if port in (2, 3) else port


class MirrorAnchorViolation(ValueError):
    """Raised when a mirror anchor's (roblet_grammar.is_mirror_anchor)
    port 3 already holds real content instead of being free for
    build_symmetric_graph to populate.

    growable_ports() stops ADD_NODE/RECONNECT_PORT from ever doing this,
    but GRAFT_SUBTREE/SWAP_SUBTREES (roblet_grammar.graft_subtree) copy a
    donor node's connectors verbatim - if the donor node had its OWN port
    3 legitimately occupied in the donor graph (because it wasn't an
    anchor there), and the graft lands it on the host's port 1 (making it
    the host's new spine-child anchor), that pre-existing port 3 becomes
    an illegal occupant in the new context. There's no good way to detect
    this at graft time without knowing the eventual host position, so it
    surfaces here instead - callers should treat it exactly like
    mjcf_generator.ModuleCollisionError: an invalid genotype to reject,
    not a crash (see moo_api._is_collision_free)."""


def _mirror_anchor_port2(full_G, half_G, anchor):
    """In place on full_G: clone whatever's attached to `anchor`'s port 2
    (read from half_G) as a mirror image attached to `anchor`'s port 3.
    No-op if port 2 is empty. `anchor` must be a mirror anchor
    (roblet_grammar.is_mirror_anchor) - the root or its port-1 child -
    the two nodes whose accumulated orientation keeps a plain port-2/3
    swap equivalent to a true global mirror reflection (see
    build_symmetric_graph's docstring)."""
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
            # Design Variables 5/6 (light-sensitive joint selection + its
            # trigger angle) mirror straight across, same as hinge_angle
            # just above and for the same reason (see _MIRROR_FOLD_TYPE's
            # comment: a mirrored module needs the SAME hinge behavior, not
            # a flipped one) - this is what keeps any sensor-placement
            # choice automatically even and bilaterally symmetric.
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

    # Which port half_root_id itself uses to connect back to anchor is
    # normally 1 (add_node always wires a fresh child's port 1 to its
    # parent), but RECONNECT_PORT could in principle have moved it since -
    # so read the real port from the original edge and mirror THAT,
    # rather than assuming 1.
    original_child_port = half_G.edges[anchor, half_root_id]["connector2"]
    mirrored_child_port = _mirror_port(original_child_port)

    mirrored_root = id_map[half_root_id]
    full_G.nodes[mirrored_root]["connectors"][mirrored_child_port] = anchor
    full_G.nodes[anchor]["connectors"][3] = mirrored_root
    full_G.add_edge(anchor, mirrored_root, connector1=3, connector2=mirrored_child_port)
    full_G.nodes[mirrored_root]["parent"] = anchor


def build_symmetric_graph(half_G):
    """Returns a NEW graph: half_G plus a mirrored clone of whatever is
    attached to port 2 of each mirror anchor (roblet_grammar.
    is_mirror_anchor - the root, and the root's port-1 "spine" child if
    one was grown), attached to that anchor's port 3. half_G itself is
    left untouched. A no-op (just a copy of half_G) if both anchors' port
    2s are empty.

    Both anchors need this, not just the root: the spine child sits on
    the mirror plane exactly like the root does (its accumulated
    orientation is a 180deg turn about the mirror axis, which - unlike
    any rotation further down its own port-2/3 subtrees - still commutes
    with the mirror reflection), so anything grown from ITS port 2 is
    just as capable of being visibly lopsided as the root's port 2 is,
    unless mirrored the same way.

    Deliberately does NOT go through roblet_grammar.graft_subtree /
    compute_node_action_mask / MAX_MODULES: those gate EVOLUTION-time
    growth of the half (capped so the final mirrored shape stays in
    budget - see MAX_MODULES's docstring), but this is a deterministic,
    always-legal build-time step applied to an already-valid half, not a
    mutation, so it must not be capped by the same (now much smaller,
    half-sized) ceiling.
    """
    root = rg.root_node(half_G)
    full_G = copy.deepcopy(half_G)

    _mirror_anchor_port2(full_G, half_G, root)
    spine_child = half_G.nodes[root]["connectors"].get(1)
    if spine_child is not None:
        _mirror_anchor_port2(full_G, half_G, spine_child)

    rg._recompute_depths(full_G, root)
    return full_G
