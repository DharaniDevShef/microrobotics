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


def build_assembly(graph_json_path, out_xml_path, meshdir="../meshes",
                    weld_solref="0.01 1", weld_solimp="0.99 0.999 0.0001"):
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
    <asset>
        <mesh name="bodyBase" file="BodyFoldedSide1.stl" scale="0.001 0.001 0.001"/>
        <mesh name="bodyLink" file="BodyFoldedSide2.stl" scale="0.001 0.001 0.001"/>
        <mesh name="bodyRigid" file="Body.stl" scale="0.001 0.001 0.001"/>
        <mesh name="connectorA" file="SGA.stl" scale="0.001 0.001 0.001"/>
        <mesh name="connectorB" file="SGB.stl" scale="0.001 0.001 0.001"/>
        <mesh name="connectorC" file="SGX.stl" scale="0.001 0.001 0.001"/>
        <material name="silver" specular="1" shininess="0.8" rgba="0.85 0.85 0.9 1"/>
        <mesh name="Magnet" file="Magnet.stl" scale="1 1 1" inertia="shell"/>
        <material name="glass" rgba="0.6 0.8 0.9 0.4" shininess="0.9" specular="1"/>
    </asset>
    <worldbody>
        <light directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" pos="0 0 1" dir="0 0 -1"/>
        <geom name="glass_floor" type="plane" size="1 1 0.1" material="glass"
            friction="0.4 0.005 0.0001" solimp="0.9 0.95 0.001 0.5 2" solref="0.02 1" condim="3"/>
        <!-- Floor Boundaries / Perimeter Walls -->
        <!-- group="1": lets the wall-rangefinder raycast (mj_ray with
             geomgroup filtering, see roblet_simulator.py) see ONLY these 4
             geoms, ignoring the floor and the robot's own nearby modules. -->
        <geom name="wall_north" type="box" pos="0 1.0 0.1" size="1.05 0.02 0.1" group="1"/>
        <geom name="wall_south" type="box" pos="0 -1.0 0.1" size="1.05 0.02 0.1" group="1"/>
        <geom name="wall_east"  type="box" pos="1.0 0 0.1" size="0.02 1.05 0.1" group="1"/>
        <geom name="wall_west"  type="box" pos="-1.0 0 0.1" size="0.02 1.05 0.1" group="1"/>
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
        joint_z = "0.0005" if m_type == "valley fold" else "-0.0055"
        joint_range = "0 90" if m_type == "valley fold" else "-90 0"

        g_pos, g_R = global_pose.get(node_id, (np.zeros(3), np.eye(3)))
        module_pos = " ".join(f"{x:.9f}" for x in g_pos)
        module_macro_quat = " ".join(f"{x:.9f}" for x in matrix_to_quaternion(g_R))

        is_rigid = m_type == "non-foldable"
        site23 = "site23_rigid" if is_rigid else "site23_fold"

        tpl = rigid_body_tpl if is_rigid else fold_body_tpl
        body_xml = tpl.format(
            module_id=num_id, module_pos=module_pos, module_macro_quat=module_macro_quat,
            module_quat=m_quat,
            connector1_pos=c1_pos, connector1_quat=c1_quat, connector1_mesh=c1_mesh,
            connector2_pos=c2_pos, connector2_quat=c2_quat, connector2_mesh=c2_mesh,
            connector3_pos=c3_pos, connector3_quat=c3_quat, connector3_mesh=c3_mesh,
            trans_val=TRANSPARENCY, joint_z=joint_z, joint_range=joint_range,
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
            main_body = elem.find(f"body[@name='bodyRigid_{num_id}']")
            ET.SubElement(main_body, "site", name="rf_east", pos="0 0 0.001",
                          quat="0.70710678 0 0.70710678 0")
            ET.SubElement(main_body, "site", name="rf_west", pos="0 0 0.001",
                          quat="0.70710678 0 -0.70710678 0")
            ET.SubElement(main_body, "site", name="rf_north", pos="0 0 0.001",
                          quat="0.70710678 -0.70710678 0 0")
            ET.SubElement(main_body, "site", name="rf_south", pos="0 0 0.001",
                          quat="0.70710678 0.70710678 0 0")

        return elem

    root = ET.fromstring(base_xml_skeleton)
    worldbody = root.find("worldbody")
    for n in sorted(G.nodes, key=lambda x: int(x.split("_")[1])):
        worldbody.append(build_module_element(n))

    # -------------------------------------------------------------
    # 7. Contact excludes + weld equality constraints (one weld per graph edge)
    # -------------------------------------------------------------
    contact_elem = ET.SubElement(root, "contact")
    equality_elem = ET.SubElement(root, "equality")
    actuator_elem = ET.SubElement(root, "actuator")

    def module_bodies(node_id):
        """All geom-bearing body names belonging to a single module."""
        num_id = node_id.replace("module_", "")
        if modules_info[node_id] == "non-foldable":
            return [f"bodyRigid_{num_id}", f"connector1_{num_id}",
                    f"connector2_{num_id}", f"connector3_{num_id}"]
        return [f"bodyBase_{num_id}", f"bodyLink_{num_id}", f"connector1_{num_id}",
                f"connector2_{num_id}", f"connector3_{num_id}"]

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

    # 7b. Cross-module excludes: since mating is done via weld (not contact),
    #     disable contact between EVERY geom-bearing body of one module and
    #     every geom-bearing body of any other module. This avoids the weld
    #     fighting a simultaneous contact force at the same mating surface,
    #     and matches the old code's intent (it excluded whole-module pairs,
    #     but those wrapper bodies carried no geoms, so this version is the
    #     one that actually takes effect).
    node_list = sorted(G.nodes, key=lambda x: int(x.split("_")[1]))
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            for b1 in module_bodies(node_list[i]):
                for b2 in module_bodies(node_list[j]):
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

    n_modules = len(G.nodes)
    n_edges = len(G.edges)
    print(f"[{graph_json_path}] {n_modules} independent free-body module(s) "
          f"({n_modules} freejoints), {len(fold_joints)} hinge joint(s), "
          f"{n_edges} weld equality constraint(s) (one per graph edge, {len(components)} "
          f"connected component(s)). Every fold joint is independently and "
          f"simultaneously actuatable.")

    return {
        "graph": G, "global_pose": global_pose, "components": components,
        "n_modules": n_modules, "n_welds": n_edges,
    }


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "../graphs/assembly_graph.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "../models/assembly.xml"
    build_assembly(json_path, out_path)