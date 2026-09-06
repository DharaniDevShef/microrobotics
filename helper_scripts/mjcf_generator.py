"""
Generic MuJoCo assembly builder for foldable-module graphs - Individual connections
"""
import json
import sys
import numpy as np
import networkx as nx
from xml.dom import minidom
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------
# 1. Coordinate Math Helpers
# ---------------------------------------------------------------------
def matrix_to_quaternion(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])


def get_rotation_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


# ---------------------------------------------------------------------
# 1.5 Physical properties -- sourced from doc/design_info.txt
#     Fusion 360 mass-property exports ("Moment of Inertia at Center of
#     Mass", g*mm^2) converted to SI (kg, kg*m^2). Body-local CoM offsets
#     (`com`) are NOT derivable from design_info.txt -- it only lists
#     assembly-global coordinates -- so the previously fitted body-local
#     offsets are kept; only mass/inertia are refreshed from the doc.
#     To update for a new design_info.txt revision, only this table and
#     the MAGNET_* constants below need to change.
# ---------------------------------------------------------------------
G_TO_KG = 1e-3
G_MM2_TO_KG_M2 = 1e-9  # 1 g*mm^2 = 1e-3 kg * (1e-3 m)^2


def _inertia_tensor(ixx, iyy, izz, ixy, ixz, iyz):
    """3x3 inertia tensor (kg*m^2) from a Fusion export (g*mm^2)."""
    return np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ]) * G_MM2_TO_KG_M2


# mesh name -> {mass_kg, com (body-local, m), inertia (3x3 kg*m^2 @ CoM)}
PART_PROPERTIES = {
    "bodyBase": dict(   # BodyFoldedSide1.stl
        mass=0.105 * G_TO_KG,
        com=np.array([-0.000000, -0.002067, -0.002280]),
        inertia=_inertia_tensor(0.983, 1.030, 0.960, 0.058, -0.021, -0.037),
    ),
    "bodyLink": dict(   # BodyFoldedSide2.stl
        mass=0.091 * G_TO_KG,
        com=np.array([-0.000000, 0.002372, -0.002389]),
        inertia=_inertia_tensor(0.971, 0.941, 0.764, 0.307, 0.137, -0.220),
    ),
    "bodyRigid": dict(  # Body.stl
        mass=0.196 * G_TO_KG,
        com=np.array([-0.000000, 0.000000, -0.002331]),
        inertia=_inertia_tensor(2.503, 2.409, 2.658, -0.099, 0.221, -0.140),
    ),
    "connectorA": dict(  # SGA.stl
        mass=0.041 * G_TO_KG,
        inertia=_inertia_tensor(0.112, 0.160, 0.125, -0.006, -0.003, 0.026),
    ),
    "connectorB": dict(  # SGB.stl
        mass=0.043 * G_TO_KG,
        inertia=_inertia_tensor(0.193, 0.143, 0.174, -0.010, -0.041, 0.008),
    ),
    "connectorC": dict(  # SGX.stl
        mass=0.067 * G_TO_KG,
        inertia=_inertia_tensor(0.283, 0.223, 0.260, -0.012, -0.049, 0.009),
    ),
}

# Body-local mounting offset (m) of a connector's own CoM along its local
# Z axis, before the magnet is added. Connector meshes A/B/C are
# similarly-sized magnetic-mate plugs at the same mounting transform, so
# the offset depends on which site/parent they're welded to, not on which
# mesh -- mass and inertia (which do vary by mesh) come from PART_PROPERTIES.
CONNECTOR_MOUNT_OFFSET_Z = {
    "site1": 0.001027,         # connector1 site (fold's bodyLink & rigid's bodyRigid)
    "site23_fold": 0.001102,   # connector2/3 sites on a foldable module's bodyBase
    "site23_rigid": 0.001471,  # connector2/3 sites on a non-foldable module's bodyRigid
}

# 2x2 mm cylindrical N42SH neodymium magnet, embedded at the center of
# each connector site, axis along the connector's local Z (design_info.txt).
MAGNET_MASS = 0.0471 * G_TO_KG           # kg
MAGNET_RADIUS = 0.001                    # m
MAGNET_HEIGHT = 0.002                    # m
MAGNET_LOCAL_POS = np.array([0.0, 0.0, 0.0009])  # site-local, m
MAGNET_INERTIA = np.diag([
    (MAGNET_MASS / 12.0) * (3 * MAGNET_RADIUS ** 2 + MAGNET_HEIGHT ** 2),
    (MAGNET_MASS / 12.0) * (3 * MAGNET_RADIUS ** 2 + MAGNET_HEIGHT ** 2),
    0.5 * MAGNET_MASS * MAGNET_RADIUS ** 2,
])

# Photometric calibration for the scene's directional light, stored in
# MuJoCo's own <light intensity="..."> field (candela for point/spot lights;
# for a directional source there's no distance falloff, so this is read
# directly as the illuminance -- in lux -- a light_sensor_* site would
# report while facing it head-on with a clear line of sight. Roughly a dim
# indoor-lighting level; see get_light_sensor_values() in roblet_simulator.py
# for the cosine/occlusion falloff applied on top of this.
LIGHT_INTENSITY_LUX = 500

# <light>'s own ambient/diffuse/specular below (rendering only -- lux
# comes from LIGHT_INTENSITY_LUX above, get_light_sensor_values() never
# reads these) are deliberately lower than MuJoCo's typical example values.
# This light is DIRECTIONAL (no distance falloff -- it covers the whole
# floor at full strength) and the viewer/renderer always adds its own
# camera-attached headlight on top by default; at the old 0.3/0.8/0.2
# values the two stacked and blew the floor out to solid white in ANY
# run (not just roblet_simulator.run_light_tests() -- confirmed with a
# render using nothing but these compiled defaults, no run_light_tests()
# code involved at all). These lower values keep the floor/robot visibly
# lit without overexposing once headlight adds in.
LIGHT_AMBIENT = "0.1 0.1 0.1"
LIGHT_DIFFUSE = "0.2 0.2 0.2"
LIGHT_SPECULAR = "0.05 0.05 0.05"

# bodyLink's own z-extent (m, local/body frame -- see BodyFoldedSide2.stl's
# bounding box) at the two points its hinge crease can physically sit: right
# at the top face for a valley fold, right at the bottom face for a mountain
# fold. joint_z below picks between them for the <joint> itself; each
# module's light_sensor_joint_N site reuses that same value so it always
# sits on the correct face for its fold type without extra logic.
JOINT_Z_VALLEY_TOP = 0.0005
JOINT_Z_MOUNTAIN_BOTTOM = -0.0055


def _combine_rigid_bodies(m1, com1, I1, m2, com2, I2):
    """Merge two (mass, CoM, inertia-about-CoM) rigid parts into one,
    via the generalized parallel-axis (Huygens-Steiner) theorem."""
    m = m1 + m2
    com = (m1 * com1 + m2 * com2) / m

    def shifted(I, mi, ci):
        d = ci - com
        return I + mi * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    return m, com, shifted(I1, m1, com1) + shifted(I2, m2, com2)


def _inertial_xml(mass, com, inertia):
    ixx, iyy, izz = inertia[0, 0], inertia[1, 1], inertia[2, 2]
    ixy, ixz, iyz = inertia[0, 1], inertia[0, 2], inertia[1, 2]
    return (f'<inertial pos="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" '
            f'mass="{mass:.8f}" '
            f'fullinertia="{ixx:.6e} {iyy:.6e} {izz:.6e} {ixy:.6e} {ixz:.6e} {iyz:.6e}"/>')


def body_inertial(part_name):
    """<inertial> for a bare structural body (no embedded magnet)."""
    part = PART_PROPERTIES[part_name]
    return _inertial_xml(part["mass"], part["com"], part["inertia"])


def connector_inertial(mesh_name, mount_site):
    """<inertial> for a connector body = its mesh (A/B/C) + embedded magnet."""
    part = PART_PROPERTIES[mesh_name]
    com1 = np.array([0.0, 0.0, CONNECTOR_MOUNT_OFFSET_Z[mount_site]])
    mass, com, inertia = _combine_rigid_bodies(
        part["mass"], com1, part["inertia"],
        MAGNET_MASS, MAGNET_LOCAL_POS, MAGNET_INERTIA,
    )
    return _inertial_xml(mass, com, inertia)


def _write_joint_target_angles(root, graph, fold_joints):
    """Write per-joint hinge angles from the graph JSON into XML metadata,
    plus (for every light-sensitive joint - Design Variable 5) its
    light-triggered target angle (Design Variable 6,
    hinge_angle_on_light_detection), under the matching
    "light_ctrl_joint{idx}" name so roblet_simulator.py's
    set_angle_to_joint can look one up from the other by just swapping the
    "ctrl_joint"/"light_ctrl_joint" prefix on an actuator's own name. Only
    written for joints that are actually light_sensitive - a joint with no
    sensor has no meaningful trigger angle to record."""
    custom_elem = ET.SubElement(root, "custom")
    for idx, num_id in enumerate(fold_joints, start=1):
        module_id = f"module_{num_id}"
        node = graph.nodes[module_id]
        hinge_angle = node.get("hinge_angle", 0.0)
        actuator_name = f"ctrl_joint{idx}"
        ET.SubElement(
            custom_elem,
            "numeric",
            name=actuator_name,
            data=f"{float(hinge_angle):.6f}",
        )
        if node.get("light_sensitive", False):
            light_hinge_angle = node.get("light_hinge_angle", 0.0)
            ET.SubElement(
                custom_elem,
                "numeric",
                name=f"light_ctrl_joint{idx}",
                data=f"{float(light_hinge_angle):.6f}",
            )


class ModuleCollisionError(ValueError):
    """Raised when two un-mated modules geometrically overlap in 3D."""


def _assert_no_unintended_collisions(xml_path, G, fold_joints):
    """Load the just-written model, evaluate contacts at the flat (qpos0)
    pose, the fully-folded pose (every hinge driven to its graph's
    `hinge_angle`), and - if the graph has any light-sensitive joint at
    all - a light-triggered pose (every light-sensitive hinge driven to
    its own `light_hinge_angle` instead, everything else left at its
    already-settled baseline fold), raising if any geom of one module
    touches/interpenetrates a geom of a *different* module in any of them.

    The light-triggered pose exists because `light_hinge_angle` (Design
    Variable 6) is free to land anywhere in [0, 45] deg independent of the
    baseline `hinge_angle` (roblet_grammar.mutate_light_hinge_angle) - a
    genotype that's perfectly collision-free in its baseline fold can still
    self-intersect the moment a real light stimulus actually triggers it,
    and that was never checked here before. Only the ALL-triggered-at-once
    case is checked (matching a full-width "front" stimulus, the most
    common and geometrically extreme case) rather than every subset of
    light-sensitive joints - a full 2^n sweep isn't tractable to run on
    every collision check.

    Mated module pairs (joined by a graph edge) already have every
    body-body pair excluded in <contact> (see step 7b above), so MuJoCo
    never generates a contact for them regardless of geometry. Any contact
    that still shows up here is therefore a genuine, un-mated overlap --
    e.g. two branches that swing into each other once folded -- not a
    false positive from the mating surfaces. Module-vs-world contacts
    (floor/walls) are ignored: resting on the floor is expected.

    The folded pose is checked (not just the flat rest pose) because
    origami-like self-collisions typically only appear once hinges are
    actually folded -- see roblet_simulator.py's actuated "folded" state /
    mujoco_api.py's State 2. Modules are independent freejoint bodies tied
    together only by <equality><weld> constraints (not a kinematic joint
    tree), and mj_forward() does NOT resolve equality constraints into
    qpos -- it only computes kinematics from whatever qpos already is. So
    directly poking a hinge joint's qpos and calling mj_forward once would
    rotate that hinge locally while every module downstream of it (linked
    through the weld chain) stays stranded at its flat-pose position,
    producing phantom overlaps that would never occur once the weld
    actually settles. Instead, the fold is driven through the real
    position actuators and mj_step()'d forward so the welds pull dependent
    modules along, exactly as happens in an actual rollout.
    """
    import mujoco  # local import: keep this generator usable without mujoco installed

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    def module_of(geom_id):
        body_id = model.geom_bodyid[geom_id]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        return name.rsplit("_", 1)[-1] if name and "_" in name else None

    def contacts_now(state_label):
        found = set()
        for i in range(data.ncon):
            con = data.contact[i]
            m1, m2 = module_of(con.geom1), module_of(con.geom2)
            if m1 is None or m2 is None or m1 == m2:
                continue  # world geom (floor/wall) or within-module contact
            pair = tuple(sorted((f"module_{m1}", f"module_{m2}"), key=lambda n: int(n.split("_")[1])))
            found.add((state_label, pair[0], pair[1]))
        return found

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    bad = contacts_now("flat")

    if fold_joints:
        for idx, num_id in enumerate(fold_joints):
            m_type = G.nodes[f"module_{num_id}"]["module_type"]
            sign = 1.0 if m_type == "valley fold" else -1.0
            angle_rad = sign * np.radians(G.nodes[f"module_{num_id}"].get("hinge_angle", 0.0))
            data.ctrl[idx] = angle_rad
        # Settle: let the weld constraints pull dependent modules along as
        # the hinges fold, same as a real rollout (welds use solref "0.01 1"
        # -- a ~10ms time constant -- so a couple of settled seconds is
        # comfortably past convergence).
        for _ in range(300):
            mujoco.mj_step(model, data)
        bad |= contacts_now("folded")

        # ---- light-triggered pose (Design Variables 5/6) ----
        # Continues from the already-settled baseline-folded state above
        # (not a fresh reset) and only changes the light-sensitive joints'
        # own ctrl targets, matching how a real "front" stage actually
        # reaches this pose - reset_to_initial_pose() back to the settled
        # baseline fold, then override just the triggered joints.
        light_sensitive_idx = [
            idx for idx, num_id in enumerate(fold_joints)
            if G.nodes[f"module_{num_id}"].get("light_sensitive", False)
        ]
        if light_sensitive_idx:
            for idx in light_sensitive_idx:
                num_id = fold_joints[idx]
                m_type = G.nodes[f"module_{num_id}"]["module_type"]
                sign = 1.0 if m_type == "valley fold" else -1.0
                light_angle_rad = sign * np.radians(G.nodes[f"module_{num_id}"].get("light_hinge_angle", 0.0))
                data.ctrl[idx] = light_angle_rad
            for _ in range(300):
                mujoco.mj_step(model, data)
            bad |= contacts_now("light_triggered")

    if bad:
        details = "\n".join(
            f"  [{state}] {m1} <-> {m2}"
            for state, m1, m2 in sorted(bad)
        )
        raise ModuleCollisionError(
            f"{xml_path}: {len(bad)} colliding module pair(s) -- "
            f"these modules are not connected by a graph edge, so they were not excluded "
            f"from contact, and the geometry actually overlaps in 3D:\n{details}"
        )


def build_assembly(graph_json_path, out_xml_path, meshdir="../meshes",
                    weld_solref="0.01 1", weld_solimp="0.99 0.999 0.0001",
                    check_collisions=True):
    with open(graph_json_path, "r") as f:
        data = json.load(f)

    # nx.node_link_graph carries every node/edge attribute through verbatim
    # (module_type, connectors, depth, parent, type_id, ...), so this stays
    # in sync automatically as the schema gains fields for GNN/graph
    # transformer/pymoo use, instead of hand-listing keys here.
    G = nx.node_link_graph(data, edges="edges")
    if not G.is_directed():
        raise ValueError(
            f"{graph_json_path} is an undirected graph; re-save it with the "
            "updated Pattern Generator so edges follow the module_1 hierarchy "
            "(needed to know which connector mesh -- A or B -- each side gets)."
        )

    # -------------------------------------------------------------
    # 2. Structural Constants & Connector-Mesh Assignments
    # -------------------------------------------------------------
    connector_assignments = {}
    for node_id in G.nodes:
        connector_assignments[node_id] = {1: "connectorC", 2: "connectorC", 3: "connectorC"}
    for u, v, d in G.edges(data=True):
        connector_assignments[u][d["connector1"]] = "connectorA"
        connector_assignments[v][d["connector2"]] = "connectorB"

    CONN_POSITIONS = {
        3: "-0.002017680 -0.001164957 -0.002500001",
        2: "0.002017941 -0.001165050 -0.002500001",
        1: "0.000000439 0.002330720 -0.002500000"
    }
    CONN_QUATS = {
        3: "0.683012546 0.183013001 -0.683012546 0.183013001",
        2: "0.683013104 0.183012443 0.683013104 -0.183012443",
        1: "0.500000000 -0.500000000 0.500000000 0.500000000"
    }

    CONN1_ROT = 180
    CONN2_ROT = 120
    CONN3_ROT = -120
    PARENT_CHILD_DIS = 0.0085  # meters

    PARENT_CONNECTOR_REL_OFFSETS = {
        1: np.array([PARENT_CHILD_DIS * np.sin(np.radians(180 - CONN1_ROT)),
                     PARENT_CHILD_DIS * np.cos(np.radians(180 - CONN1_ROT)), 0.0]),
        2: np.array([PARENT_CHILD_DIS * np.sin(np.radians(CONN2_ROT)),
                     PARENT_CHILD_DIS * np.cos(np.radians(CONN2_ROT)), 0.0]),
        3: np.array([PARENT_CHILD_DIS * np.sin(np.radians(CONN3_ROT)),
                     PARENT_CHILD_DIS * np.cos(np.radians(CONN3_ROT)), 0.0]),
    }
    PARENT_CONNECTOR_ROTATIONS = {
        1: get_rotation_z(np.radians(0)),
        2: get_rotation_z(np.radians(-CONN2_ROT)),
        3: get_rotation_z(np.radians(-CONN3_ROT)),
    }
    CHILD_CONNECTOR_ROTATIONS = {
        1: get_rotation_z(np.radians(180)),
        2: get_rotation_z(np.radians(60)),
        3: get_rotation_z(np.radians(-60)),
    }

    TRANSPARENCY = 1

    # -------------------------------------------------------------
    # 3. Shared per-edge local-transform math (unchanged)
    # -------------------------------------------------------------
    edge_lookup = {}
    for u, v, d in G.edges(data=True):
        s_u, s_v = d["connector1"], d["connector2"]
        edge_lookup[frozenset((u, v))] = {"u": u, "v": v, "s_u": s_u, "s_v": s_v}

    def edge_info(a, b):
        return edge_lookup[frozenset((a, b))]

    def edge_local_transform(current, nbr):
        info = edge_info(current, nbr)
        if info["u"] == current:
            parent_slot, child_slot = info["s_u"], info["s_v"]
        else:
            parent_slot, child_slot = info["s_v"], info["s_u"]

        R_p_conn = PARENT_CONNECTOR_ROTATIONS[parent_slot]
        pos_p_conn = PARENT_CONNECTOR_REL_OFFSETS[parent_slot].copy()
        R_c_conn = CHILD_CONNECTOR_ROTATIONS[child_slot]
        pos_c_conn = np.zeros(3)

        local_R = R_p_conn @ R_c_conn.T
        local_pos = pos_p_conn - (local_R @ pos_c_conn)
        return local_pos, local_R, parent_slot

    # ===== NEW: global pose for EVERY module, not just tree roots =====
    # -------------------------------------------------------------
    # 4. Global pose per node via BFS over the FULL graph (all edges).
    #    Every module gets a real pose here -- there is no "tree" any more.
    # -------------------------------------------------------------
    UG = G.to_undirected()
    components = list(nx.connected_components(UG))
    global_pose = {}
    COMPONENT_SPACING = 0.5  # meters, keeps disconnected assemblies apart

    for comp_idx, comp in enumerate(components):
        comp_nodes = sorted(comp, key=lambda x: int(x.split("_")[1]))
        comp_start = comp_nodes[0]
        comp_offset = np.array([comp_idx * COMPONENT_SPACING, 0.0, 0.0])
        global_pose[comp_start] = (comp_offset.copy(), np.eye(3))
        gq = [comp_start]
        gvisited = {comp_start}
        while gq:
            cur = gq.pop(0)
            cur_pos, cur_R = global_pose[cur]
            for nbr in list(G.successors(cur)) + list(G.predecessors(cur)):
                if nbr in gvisited:
                    continue
                local_pos, local_R, _ = edge_local_transform(cur, nbr)
                global_pose[nbr] = (cur_pos + cur_R @ local_pos, cur_R @ local_R)
                gvisited.add(nbr)
                gq.append(nbr)
    # ===== END NEW =====

    # -------------------------------------------------------------
    # 5. XML Templates -- every module is now always a root (own freejoint).
    #    No child modules are nested inside another module's body any more.
    # -------------------------------------------------------------
    fold_body_tpl = """
<body name="module_{module_id}" pos="{module_pos}" quat="{module_macro_quat}">
    <freejoint name="free_module_{module_id}"/>
    <inertial pos="0 0 0" mass="1e-08" diaginertia="1e-08 1e-08 1e-08"/>
    <body name="bodyBase_{module_id}" pos="0.000000000 0.000000000 0.000000000" quat="{module_quat}">
        {bodyBase_inertial}
        <geom name="geom_bodyBase_{module_id}" type="mesh" mesh="bodyBase" rgba="0.2 0.2 0.8 {trans_val}"/>
        <body name="connector2_{module_id}" pos="{connector2_pos}" quat="{connector2_quat}">
            {connector2_inertial}
            <geom name="geom_connector2_{module_id}" type="mesh" mesh="{connector2_mesh}" rgba="0 0 0 {trans_val}"/>
            <geom name="magnet_connector2_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009" mass="0"/>
        </body>
        <body name="connector3_{module_id}" pos="{connector3_pos}" quat="{connector3_quat}">
            {connector3_inertial}
            <geom name="geom_connector3_{module_id}" type="mesh" mesh="{connector3_mesh}" rgba="0.2 0.2 0.8 {trans_val}"/>
            <geom name="magnet_connector3_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009" mass="0"/>
        </body>
        <body name="bodyLink_{module_id}" pos="0 0 0" quat="1 0 0 0">
            {bodyLink_inertial}
            <joint name="joint_{module_id}" type="hinge" axis="1 0 0" pos="0 0.001 {joint_z}" range="{joint_range}" limited="true" armature="1e-04" damping="0"/>
            <!-- <geom name="joint_marker_bodyLink_{module_id}" type="cylinder" size="0.0002 0.008" pos="0 0.001 {joint_z}" quat="0.7071 0 0.7071 0" rgba="0 1 0 1" mass="0"/> -->
            {light_sensor_site}
            <geom name="geom_bodyLink_{module_id}" type="mesh" mesh="bodyLink" rgba="0.2 0.2 0.8 {trans_val}"/>
            <body name="connector1_{module_id}" pos="{connector1_pos}" quat="{connector1_quat}">
                {connector1_inertial}
                <geom name="geom_connector1_{module_id}" type="mesh" mesh="{connector1_mesh}" rgba="1 1 1 {trans_val}"/>
                <geom name="magnet_connector1_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009" mass="0"/>
            </body>
        </body>
    </body>
</body>
"""

    rigid_body_tpl = """
<body name="module_{module_id}" pos="{module_pos}" quat="{module_macro_quat}">
    <freejoint name="free_module_{module_id}"/>
    <inertial pos="0 0 0" mass="1e-08" diaginertia="1e-08 1e-08 1e-08"/>
    <body name="bodyRigid_{module_id}" pos="0.000000000 0.000000000 0.000000000" quat="{module_quat}">
        {bodyRigid_inertial}
        <geom name="geom_bodyRigid_{module_id}" type="mesh" mesh="bodyRigid" rgba="0.2 0.2 0.8 {trans_val}"/>
        <body name="connector1_{module_id}" pos="{connector1_pos}" quat="{connector1_quat}">
            {connector1_inertial}
            <geom name="geom_connector1_{module_id}" type="mesh" mesh="{connector1_mesh}" rgba="1 1 1 {trans_val}"/>
            <geom name="magnet_connector1_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009" mass="0"/>
        </body>
        <body name="connector2_{module_id}" pos="{connector2_pos}" quat="{connector2_quat}">
            {connector2_inertial}
            <geom name="geom_connector2_{module_id}" type="mesh" mesh="{connector2_mesh}" rgba="0 0 0 {trans_val}"/>
            <geom name="magnet_connector2_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009" mass="0"/>
        </body>
        <body name="connector3_{module_id}" pos="{connector3_pos}" quat="{connector3_quat}">
            {connector3_inertial}
            <geom name="geom_connector3_{module_id}" type="mesh" mesh="{connector3_mesh}" rgba="0.2 0.2 0.8 {trans_val}"/>
            <geom name="magnet_connector3_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009" mass="0"/>
        </body>
    </body>
</body>
"""

    base_xml_skeleton = f"""<?xml version="1.0" ?>
<mujoco model="FusionExportAssembly">
    <compiler meshdir="{meshdir}" autolimits="false"/>
    <size nconmax="500" njmax="1500" nstack="100000"/>
    <option timestep="0.01" integrator="implicitfast">
        <flag contact="enable"/>
    </option>
    <visual>
        <global offwidth="1920" offheight="1080"/>
    </visual>
    <asset>
        <mesh name="bodyBase" file="BodyFoldedSide1.stl" scale="0.001 0.001 0.001"/>
        <mesh name="bodyLink" file="BodyFoldedSide2.stl" scale="0.001 0.001 0.001"/>
        <mesh name="bodyRigid" file="Body.stl" scale="0.001 0.001 0.001"/>
        <mesh name="connectorA" file="SGA.stl" scale="0.001 0.001 0.001"/>
        <mesh name="connectorB" file="SGB.stl" scale="0.001 0.001 0.001"/>
        <mesh name="connectorC" file="SGX.stl" scale="0.001 0.001 0.001"/>
        <material name="silver" specular="1" shininess="0.8" rgba="0.85 0.85 0.9 1"/>
        <mesh name="Magnet" file="Magnet.stl" scale="1 1 1" inertia="shell"/>
        <texture name="grid" type="2d" builtin="checker" rgb1="1 1 1" rgb2="0.85 0.85 0.85" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="40 40" texuniform="true" specular="0" shininess="0"/>
        <!-- emission="1" (MuJoCo's max - self-illuminated on every face
             regardless of light direction, confirmed empirically; higher
             values render identically) so the luminance_sheet geom below
             reads as an actually glowing/luminous patch (a UV-excited
             fluorescent pheromone trace) rather than a flat-colored one -
             see roblet_simulator.py's _show_luminance_sheet(). -->
        <material name="luminance_glow" emission="1" specular="0.3" shininess="0.2" rgba="1 0.85 0 1"/>
    </asset>
    <worldbody>
        <light directional="true" ambient="{LIGHT_AMBIENT}" diffuse="{LIGHT_DIFFUSE}" specular="{LIGHT_SPECULAR}" pos="0 0 1" dir="0 0 -1" intensity="{LIGHT_INTENSITY_LUX}"/>
        <geom name="floor" type="plane" size="1 1 0.1" material="grid"
            friction="0.4 0.005 0.0001" solimp="0.9 0.95 0.001 0.5 2" solref="0.02 1" condim="3"/>
        <!-- Floor Boundaries / Perimeter Walls -->
        <!-- group="1": lets the wall-rangefinder raycast (mj_ray with
             geomgroup filtering, see roblet_simulator.py) see ONLY these 4
             geoms, ignoring the floor and the robot's own nearby modules. -->
        <geom name="wall_north" type="box" pos="0 1.0 0.1" size="1.05 0.02 0.1" group="1"/>
        <geom name="wall_south" type="box" pos="0 -1.0 0.1" size="1.05 0.02 0.1" group="1"/>
        <geom name="wall_east"  type="box" pos="1.0 0 0.1" size="0.02 1.05 0.1" group="1"/>
        <geom name="wall_west"  type="box" pos="-1.0 0 0.1" size="0.02 1.05 0.1" group="1"/>
        <!-- Placeholder for a floor-level "luminance sheet" (UV-excited
             fluorescent pheromone trace, per the wireless-pheromone-robot
             paper) - roblet_simulator.py's run_light_tests()/
             run_headless_light_tests() reposition, resize and recolor this
             at runtime (pos/size/quat/rgba are all just plugged in as
             initial placeholders here) to sit over whichever region a
             "left"/"front" stage is testing, since MuJoCo's geom count is
             fixed at compile time - see roblet_simulator._show_luminance_
             sheet(). contype/conaffinity "0": visual only, never a
             physical obstacle. rgba alpha 0: invisible until a stage
             actually shows it. material="luminance_glow": emissive, so
             the shown patch reads as actually glowing, not flat-colored. -->
        <geom name="luminance_sheet" type="box" pos="0 0 -1" size="0.001 0.001 0.0002"
            material="luminance_glow" rgba="1 0.85 0 0" contype="0" conaffinity="0" group="2"/>
    </worldbody>
</mujoco>
"""

    modules_info = {n: G.nodes[n]["module_type"] for n in G.nodes}
    fold_joints = [n.replace("module_", "") for n in sorted(G.nodes, key=lambda x: int(x.split("_")[1]))
                   if modules_info[n] != "non-foldable"]

    # -------------------------------------------------------------
    # 6. Flat, single-body-per-module construction (no recursion, no nesting)
    # -------------------------------------------------------------
    def build_module_element(node_id):
        num_id = node_id.replace("module_", "")
        m_type = G.nodes[node_id]["module_type"]

        c1_pos, c1_quat = CONN_POSITIONS[1], CONN_QUATS[1]
        c2_pos, c2_quat = CONN_POSITIONS[2], CONN_QUATS[2]
        c3_pos, c3_quat = CONN_POSITIONS[3], CONN_QUATS[3]

        c1_mesh = connector_assignments[node_id][1]
        c2_mesh = connector_assignments[node_id][2]
        c3_mesh = connector_assignments[node_id][3]

        m_quat = "1 0 0 0"
        joint_z = JOINT_Z_VALLEY_TOP if m_type == "valley fold" else JOINT_Z_MOUNTAIN_BOTTOM
        joint_range = "0 90" if m_type == "valley fold" else "-90 0"

        g_pos, g_R = global_pose.get(node_id, (np.zeros(3), np.eye(3)))
        module_pos = " ".join(f"{x:.9f}" for x in g_pos)
        module_macro_quat = " ".join(f"{x:.9f}" for x in matrix_to_quaternion(g_R))

        is_rigid = m_type == "non-foldable"
        site23 = "site23_rigid" if is_rigid else "site23_fold"

        # Design Variable 5 (light-sensitive joint selection): only a
        # foldable module carrying the light-sensitive PVC strip gets a
        # light_sensor_joint_* site at all - a rigid module never can
        # (no hinge, gated by is_rigid above), and a foldable module the
        # graph didn't select gets no site either, so
        # get_light_sensor_values()/set_angle_to_joint() simply never see
        # (and can never react to) light at that joint.
        is_light_sensitive = (not is_rigid) and G.nodes[node_id].get("light_sensitive", False)
        # The sensor site sits on the backside of the joint rather than
        # coinciding with it (fold_body_tpl's <joint pos="0 0.001
        # {joint_z}" .../>), on both axes:
        #  - z is unconditionally JOINT_Z_MOUNTAIN_BOTTOM, the lower of
        #    bodyLink's two faces, regardless of this module's own fold
        #    type. A mountain-fold joint's crease already sits at that
        #    same face (joint_z == JOINT_Z_MOUNTAIN_BOTTOM above), so its
        #    sensor was already on the hidden underside; a valley-fold
        #    joint's crease sits at the visually-exposed top face instead
        #    (joint_z == JOINT_Z_VALLEY_TOP), so its sensor needs pulling
        #    down to this same bottom face to end up equally hidden -
        #    using JOINT_Z_VALLEY_TOP here instead would do the opposite
        #    and expose the mountain-fold sensors that were already fine.
        #  - y is +0.0015 (not the joint's own +0.001, and NOT the -0.001
        #    "connector 2/3 side" this used to sit at) - +0.001 to +0.00233
        #    is where BodyFoldedSide2.stl has a real cutout/window (visible
        #    once a mountain joint folds open enough to expose that face -
        #    connector1's own mounting socket lives right there, per
        #    CONN_POSITIONS/CONNECTOR_MOUNT_OFFSET_Z above), and a sensor
        #    site placed inside it visibly floats in open air rather than
        #    sitting on solid material - confirmed empirically (multi-hit
        #    mj_ray sweeps + renders) at -0.001, -0.002, 0 and +0.0015; only
        #    +0.0015 lands outside that cutout at every fold angle tested.
        # quat="0 1 0 0" is a 180deg rotation about local X, flipping the
        # site's own +Z (its sensing normal - roblet_simulator.py's
        # get_light_sensor_values() reads the SITE's orientation, not the
        # owning body's) to point at the site's -Z, i.e. downward toward
        # the floor - a ground-facing sensor for a floor-level luminance
        # sheet/pheromone trail, matching where it's actually mounted,
        # rather than the body's own (upward) Z axis.
        light_sensor_site = (
            f'<site name="light_sensor_joint_{num_id}" type="box" size="0.0008 0.0008 0.0001" '
            f'pos="0 0.0015 {JOINT_Z_MOUNTAIN_BOTTOM}" quat="0 1 0 0" rgba="0 1 0 1"/>'
        ) if is_light_sensitive else ""

        tpl = rigid_body_tpl if is_rigid else fold_body_tpl
        body_xml = tpl.format(
            module_id=num_id, module_pos=module_pos, module_macro_quat=module_macro_quat,
            module_quat=m_quat,
            connector1_pos=c1_pos, connector1_quat=c1_quat, connector1_mesh=c1_mesh,
            connector2_pos=c2_pos, connector2_quat=c2_quat, connector2_mesh=c2_mesh,
            connector3_pos=c3_pos, connector3_quat=c3_quat, connector3_mesh=c3_mesh,
            trans_val=TRANSPARENCY, joint_z=joint_z, joint_range=joint_range,
            light_sensor_site=light_sensor_site,
            bodyBase_inertial=body_inertial("bodyBase"),
            bodyLink_inertial=body_inertial("bodyLink"),
            bodyRigid_inertial=body_inertial("bodyRigid"),
            connector1_inertial=connector_inertial(c1_mesh, "site1"),
            connector2_inertial=connector_inertial(c2_mesh, site23),
            connector3_inertial=connector_inertial(c3_mesh, site23),
        )
        elem = ET.fromstring(body_xml)

        # module_1 doubles as the control/sensor module: mount the IMU
        # (accelerometer + gyro) and 4 wall-facing rangefinders on it.
        # Rangefinders cast along their site's local +Z axis, so each site's
        # quat rotates +Z to point at the wall it's named after.
        if num_id == "1":
            main_body_name = f"bodyRigid_{num_id}" if is_rigid else f"bodyBase_{num_id}"
            main_body = elem.find(f"body[@name='{main_body_name}']")
            ET.SubElement(main_body, "site", name="rf_east", pos="0 0 0.001",
                          quat="0.70710678 0 0.70710678 0", rgba="0 0 0 0")
            ET.SubElement(main_body, "site", name="rf_west", pos="0 0 0.001",
                          quat="0.70710678 0 -0.70710678 0", rgba="0 0 0 0")
            ET.SubElement(main_body, "site", name="rf_north", pos="0 0 0.001",
                          quat="0.70710678 -0.70710678 0 0", rgba="0 0 0 0")
            ET.SubElement(main_body, "site", name="rf_south", pos="0 0 0.001",
                          quat="0.70710678 0.70710678 0 0", rgba="0 0 0 0")

        return elem

    root = ET.fromstring(base_xml_skeleton)
    worldbody = root.find("worldbody")
    for n in sorted(G.nodes, key=lambda x: int(x.split("_")[1])):
        worldbody.append(build_module_element(n))

    # Persist per-joint target angles as XML metadata so the simulator can
    # recover them directly from the generated MuJoCo model file.
    _write_joint_target_angles(root, G, fold_joints)

    # -------------------------------------------------------------
    # 7. Contact excludes + weld equality constraints (one weld per graph edge)
    # -------------------------------------------------------------
    contact_elem = ET.SubElement(root, "contact")
    equality_elem = ET.SubElement(root, "equality")
    actuator_elem = ET.SubElement(root, "actuator")

    # 7a. Internal excludes (hinge-adjacent bodies within one module) -- unchanged.
    for node_id, m_type in modules_info.items():
        num_id = node_id.replace("module_", "")
        if m_type != "non-foldable":
            ET.SubElement(contact_elem, "exclude", body1=f"bodyBase_{num_id}", body2=f"bodyLink_{num_id}")
            ET.SubElement(contact_elem, "exclude", body1=f"bodyLink_{num_id}", body2=f"connector1_{num_id}")
            ET.SubElement(contact_elem, "exclude", body1=f"bodyBase_{num_id}", body2=f"connector2_{num_id}")
            ET.SubElement(contact_elem, "exclude", body1=f"bodyBase_{num_id}", body2=f"connector3_{num_id}")
        else:
            ET.SubElement(contact_elem, "exclude", body1=f"bodyRigid_{num_id}", body2=f"connector1_{num_id}")
            ET.SubElement(contact_elem, "exclude", body1=f"bodyRigid_{num_id}", body2=f"connector2_{num_id}")
            ET.SubElement(contact_elem, "exclude", body1=f"bodyRigid_{num_id}", body2=f"connector3_{num_id}")

    def _connector_parent_body(node_id, c_idx):
        """The body a given connector is mounted on: connector1 hangs off
        bodyLink (the hinged half), connector2/3 off bodyBase -- both
        collapse to bodyRigid for a non-foldable module."""
        num_id = node_id.replace("module_", "")
        if modules_info[node_id] == "non-foldable":
            return f"bodyRigid_{num_id}"
        return f"bodyLink_{num_id}" if c_idx == 1 else f"bodyBase_{num_id}"

    # 7b. Cross-module excludes: since mating is done via weld (not contact),
    #     disable contact only in the immediate neighborhood of each mate --
    #     the two connector bodies the weld joins, AND each connector's own
    #     parent panel (it's flush-mounted on that panel's edge, so the two
    #     panels legitimately sit right next to each other at every normal
    #     joint too). That's a 2x2 set of 4 body pairs per edge, not the
    #     whole module: a bare connector-connector exclude is too narrow
    #     (falsely flags every ordinary joint, since the parent panels also
    #     touch by design), while excluding entire modules is too broad --
    #     see the two paragraphs below.
    #
    #     Every OTHER body pair between two mated modules -- e.g. one
    #     module's *other*, unrelated connector against the other module's
    #     bodyLink -- is deliberately left un-excluded. A module can be
    #     legitimately mated to two different neighbors at once (a graph
    #     cycle, e.g. a module welded to both a "left" and "right" parent),
    #     and in that case its own body can still collide with either
    #     parent's *other* geoms even though the two connector bodies are
    #     correctly mated -- excluding the whole module pair previously hid
    #     exactly that overlap.
    #     Likewise, module pairs with no edge at all are left un-excluded,
    #     so a real 3D overlap between two non-adjacent modules (e.g. a
    #     self-intersecting branch layout) still produces genuine MuJoCo
    #     contacts instead of being silently hidden -- see
    #     _assert_no_unintended_collisions() below, which turns any such
    #     contact into a build-time error.
    for u, v, d in G.edges(data=True):
        u_num, v_num = u.replace("module_", ""), v.replace("module_", "")
        c1, c2 = d["connector1"], d["connector2"]
        side_u = {f"connector{c1}_{u_num}", _connector_parent_body(u, c1)}
        side_v = {f"connector{c2}_{v_num}", _connector_parent_body(v, c2)}
        for b1 in side_u:
            for b2 in side_v:
                ET.SubElement(contact_elem, "exclude", body1=b1, body2=b2)

    # 7c. One weld equality constraint per graph edge = the actual physical
    #     magnetic mate between the two connector bodies. No relpose is given,
    #     so MuJoCo freezes the relative pose the two bodies have at compile
    #     time -- which is exactly right, since global_pose above was built
    #     from the same edge_local_transform that defines a correct mate.
    for u, v, d in G.edges(data=True):
        u_num, v_num = u.replace("module_", ""), v.replace("module_", "")
        c1, c2 = d["connector1"], d["connector2"]
        body1 = f"connector{c1}_{u_num}"
        body2 = f"connector{c2}_{v_num}"
        ET.SubElement(equality_elem, "weld", body1=body1, body2=body2,
                      solref=weld_solref, solimp=weld_solimp)

    # -------------------------------------------------------------
    # 8. Actuators: one per fold joint, all independently controllable
    # -------------------------------------------------------------
    for idx, num_id in enumerate(fold_joints, start=1):
        if G.nodes[f"module_{num_id}"]["module_type"] == "valley fold":
            ET.SubElement(
                actuator_elem,
                "position",
                name=f"ctrl_joint{idx}",
                joint=f"joint_{num_id}",
                kp="1",
                ctrlrange=f"0 {np.pi/2}",   # 0 to +90°
                ctrllimited="true",
                dampratio="1"
            )
        else:
            ET.SubElement(
                actuator_elem,
                "position",
                name=f"ctrl_joint{idx}",
                joint=f"joint_{num_id}",
                kp="1",
                ctrlrange=f"{-np.pi/2} 0",  # -90° to 0
                ctrllimited="true",
                dampratio="1"
            )

    xml_str = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="    ")
    clean_xml = "\n".join(line for line in pretty_xml.splitlines() if line.strip())

    with open(out_xml_path, "w") as f:
        f.write(clean_xml)

    if check_collisions:
        _assert_no_unintended_collisions(out_xml_path, G, fold_joints)

    n_modules = len(G.nodes)
    n_edges = len(G.edges)
    # print(f"[{graph_json_path}] {n_modules} independent free-body module(s) "
    #       f"({n_modules} freejoints), {len(fold_joints)} hinge joint(s), "
    #       f"{n_edges} weld equality constraint(s) (one per graph edge, {len(components)} "
    #       f"connected component(s)). Every fold joint is independently and "
    #       f"simultaneously actuatable.")

    print("MuJoCo assembly XML written created!")

    return {
        "graph": G, "global_pose": global_pose, "components": components,
        "n_modules": n_modules, "n_welds": n_edges,
    }


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "../graphs/assembly_graph.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "../models/assembly.xml"
    build_assembly(json_path, out_path)