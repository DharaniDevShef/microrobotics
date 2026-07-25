"""
Generic MuJoCo assembly builder for foldable-module graphs.

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


def build_assembly(graph_json_path, out_xml_path, meshdir="../meshes"):
    with open(graph_json_path, "r") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node in data["nodes"]:
        G.add_node(node["id"], module_type=node["module_type"],
                   hinge_angle=node["hinge_angle"], pos=node.get("_pos"))
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"],
                   connector1=edge["connector1"], connector2=edge["connector2"])

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


    FOLDABLE_TYPES = ("Mountain fold", "valley fold")

    TRANSPARENCY = 1

    # -------------------------------------------------------------
    # 3. Shared per-edge local-transform math
    #    (pose of `nbr` expressed in the module_{current} WRAPPER frame,
    #    i.e. before the valley-flip nesting correction that grow_tree
    #    applies on top of this when actually nesting bodies)
    # -------------------------------------------------------------
    edge_lookup = {}
    for u, v, d in G.edges(data=True):
        s_u, s_v = d["connector1"], d["connector2"]
        edge_lookup[frozenset((u, v))] = {"u": u, "v": v, "s_u": s_u, "s_v": s_v}

    def edge_info(a, b):
        return edge_lookup[frozenset((a, b))]

    def edge_local_transform(current, nbr):
        info = edge_info(current, nbr)
        cur_type = G.nodes[current].get("module_type", "non-foldable")
        nbr_type = G.nodes[nbr].get("module_type", "non-foldable")

        if info["u"] == current:
            parent_slot = info["s_u"]
            child_slot = info["s_v"]
        else:
            parent_slot = info["s_v"]
            child_slot = info["s_u"]

        # 1. Transform from parent center to parent's connector slot
        R_p_conn = PARENT_CONNECTOR_ROTATIONS[parent_slot]
        pos_p_conn = PARENT_CONNECTOR_REL_OFFSETS[parent_slot].copy()

        # # 2. Transform from child's connector slot back to child center
        R_c_conn = CHILD_CONNECTOR_ROTATIONS[child_slot]
        pos_c_conn = np.zeros(3)

        # 3. Combine rotations and positions (accounting for slot-to-slot mating)
        # R_rel = R_parent_slot @ R_child_slot.T
        local_R = R_p_conn @ R_c_conn.T
        local_pos = pos_p_conn - (local_R @ pos_c_conn)


        return local_pos, local_R, parent_slot

    # -------------------------------------------------------------
    # 4. forced_parent_of (global, over every edge)
    # -------------------------------------------------------------
    forced_parent_of = {}
    for u, v, d in G.edges(data=True):
        s_u, s_v = d["connector1"], d["connector2"]
        if G.nodes[v]["module_type"] in FOLDABLE_TYPES and s_v == 1:
            forced_parent_of[u] = v
        if G.nodes[u]["module_type"] in FOLDABLE_TYPES and s_u == 1:
            forced_parent_of[v] = u

    visited = set()
    tree_edges = {n: [] for n in G.nodes}
    incoming_edge = {}
    tree_edge_set = set()
    roots = []

    def grow_tree(start):
        visited.add(start)
        queue = [start]
        while queue:
            current = queue.pop(0)
            cur_type = G.nodes[current].get("module_type", "non-foldable")
            for nbr in list(G.successors(current)) + list(G.predecessors(current)):
                if nbr in visited:
                    continue
                fp = forced_parent_of.get(nbr)
                if fp is not None and fp != current:
                    continue

                local_pos, local_R, attach_slot = edge_local_transform(current, nbr)


                local_quat = matrix_to_quaternion(local_R)
                tree_edges[current].append({"child": nbr, "attach_slot": attach_slot})
                incoming_edge[nbr] = {"pos": local_pos, "quat": local_quat}
                info = edge_info(current, nbr)
                tree_edge_set.add((info["u"], info["v"]))
                visited.add(nbr)
                queue.append(nbr)

    sorted_nodes = sorted(G.nodes, key=lambda x: int(x.split("_")[1]))

    pass1_order = (["module_1"] if "module_1" in G.nodes else []) + sorted_nodes
    for start in pass1_order:
        if start in visited or start in forced_parent_of:
            continue
        roots.append(start)
        grow_tree(start)

    for start in sorted_nodes:
        if start in visited:
            continue
        roots.append(start)
        grow_tree(start)

    extra_edges = [(u, v, d) for u, v, d in G.edges(data=True) if (u, v) not in tree_edge_set]

    # -------------------------------------------------------------
    # 5. Global pose per connected component (covers ALL edges,
    #    including loop-closing ones, so every root within the same
    #    physical structure lands in its correct assembled place).
    # -------------------------------------------------------------
    UG = G.to_undirected()
    components = list(nx.connected_components(UG))
    global_pose = {}
    COMPONENT_SPACING = 0.5  # meters; only matters if >1 disconnected assembly

    for comp_idx, comp in enumerate(components):
        comp_nodes = sorted(comp, key=lambda x: int(x.split("_")[1]))
        comp_start = "module_1" if "module_1" in comp else comp_nodes[0]
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

    # -------------------------------------------------------------
    # 6. XML Templates
    # -------------------------------------------------------------
    fold_body_tpl = """
<body name="module_{module_id}" pos="{module_pos}" quat="{module_macro_quat}">
{freejoint}
    <inertial pos="0 0 0" mass="1e-05" diaginertia="1e-08 1e-08 1e-08"/>
    <body name="bodyBase_{module_id}" pos="0.000000000 0.000000000 0.000000000" quat="{module_quat}">
        <inertial pos="-0.000000 -0.002067 -0.002280" mass="0.000089" diaginertia="1.651610e-09 1.384291e-09 1.198679e-09"/>
        <geom name="geom_bodyBase_{module_id}" type="mesh" mesh="bodyBase" rgba="0.2 0.2 0.8 {trans_val}" />
        <body name="connector2_{module_id}" pos="{connector2_pos}" quat="{connector2_quat}">
            <inertial pos="0.000000 -0.000000 0.001102" mass="0.000037" diaginertia="1.661707e-10 1.661707e-10 1.944327e-10"/>
            <geom name="geom_connector2_{module_id}" type="mesh" mesh="{connector2_mesh}" rgba="0 0 0 {trans_val}"/>
            <geom name="magnet_connector2_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009"/>
        </body>
        <body name="connector3_{module_id}" pos="{connector3_pos}" quat="{connector3_quat}">
            <inertial pos="0.000000 -0.000000 0.001102" mass="0.000037" diaginertia="1.661707e-10 1.661707e-10 1.944327e-10"/>
            <geom name="geom_connector3_{module_id}" type="mesh" mesh="{connector3_mesh}" rgba="0.2 0.2 0.8 {trans_val}"/>
            <geom name="magnet_connector3_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009"/>
        </body>
        <body name="bodyLink_{module_id}" pos="0 0 0" quat="1 0 0 0">
            <inertial pos="-0.000000 0.002372 -0.002389" mass="0.000078" diaginertia="1.262893e-09 1.530211e-09 1.260111e-09"/>
            <joint name="joint_{module_id}" type="hinge" axis="1 0 0" pos="0 0.001 {joint_z}" range="{joint_range}" limited="true" armature="0.001" damping="0"/>
            <geom name="joint_marker_bodyLink_{module_id}" type="cylinder" size="0.0002 0.008" pos="0 0.001 {joint_z}" quat="0.7071 0 0.7071 0" rgba="0 1 0 1" mass="0"/>
            <geom name="geom_bodyLink_{module_id}" type="mesh" mesh="bodyLink" rgba="0.2 0.2 0.8 {trans_val}" />
            <body name="connector1_{module_id}" pos="{connector1_pos}" quat="{connector1_quat}">
                <inertial pos="-0.000000 -0.000000 0.001027" mass="0.000035" diaginertia="1.315291e-10 1.315224e-10 1.494471e-10"/>
                <geom name="geom_connector1_{module_id}" type="mesh" mesh="{connector1_mesh}" rgba="1 1 1 {trans_val}"/>
                <geom name="magnet_connector1_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009"/>
            </body>
        </body>
    </body>
</body>
"""

    rigid_body_tpl = """
<body name="module_{module_id}" pos="{module_pos}" quat="{module_macro_quat}">
{freejoint}
    <inertial pos="0 0 0" mass="1e-05" diaginertia="1e-08 1e-08 1e-08"/>
    <body name="bodyRigid_{module_id}" pos="0.000000000 0.000000000 0.000000000" quat="{module_quat}">
        <inertial pos="-0.000000 0.000000 -0.002331" mass="0.000167" diaginertia="2.914502e-09 2.914502e-09 2.458790e-09"/>
        <geom name="geom_bodyRigid_{module_id}" type="mesh" mesh="bodyRigid" rgba="0.2 0.2 0.8 {trans_val}" />
        <body name="connector1_{module_id}" pos="{connector1_pos}" quat="{connector1_quat}">
            <inertial pos="-0.000000 -0.000000 0.001027" mass="0.000035" diaginertia="1.315291e-10 1.315224e-10 1.494471e-10"/>
            <geom name="geom_connector1_{module_id}" type="mesh" mesh="{connector1_mesh}" rgba="1 1 1 {trans_val}"/>
            <geom name="magnet_connector1_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009"/>
        </body>
        <body name="connector2_{module_id}" pos="{connector2_pos}" quat="{connector2_quat}">
            <inertial pos="0.000000 0.000000 0.001471" mass="0.000057" diaginertia="3.133186e-10 3.133186e-10 2.768016e-10"/>
            <geom name="geom_connector2_{module_id}" type="mesh" mesh="{connector2_mesh}" rgba="0 0 0 {trans_val}"/>
             <geom name="magnet_connector2_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009"/>
        </body>
        <body name="connector3_{module_id}" pos="{connector3_pos}" quat="{connector3_quat}">
            <inertial pos="0.000000 0.000000 0.001471" mass="0.000057" diaginertia="3.133186e-10 3.133186e-10 2.768016e-10"/>
            <geom name="geom_connector3_{module_id}" type="mesh" mesh="{connector3_mesh}" rgba="0.2 0.2 0.8 {trans_val}"/>
             <geom name="magnet_connector3_{module_id}" type="mesh" mesh="Magnet" material="silver" rgba="0.75 0.75 0.78 1" pos="0 0 0.0009"/>
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
    </worldbody>
</mujoco>
"""

    modules_info = {n: G.nodes[n]["module_type"] for n in G.nodes}
    fold_joints = [n.replace("module_", "") for n in sorted(G.nodes, key=lambda x: int(x.split("_")[1]))
                   if modules_info[n] != "non-foldable"]

    # -------------------------------------------------------------
    # 7. Recursive nested-body construction
    # -------------------------------------------------------------
    def get_attach_body(element, num_id, m_type, attach_slot):
        if m_type == "non-foldable":
            return element.find(f".//body[@name='bodyRigid_{num_id}']")
        if attach_slot == 1:
            return element.find(f".//body[@name='bodyLink_{num_id}']")
        return element.find(f".//body[@name='bodyBase_{num_id}']")

    def build_module_element(node_id, is_root):
        num_id = node_id.replace("module_", "")
        m_type = G.nodes[node_id]["module_type"]

        c1_idx, c2_idx, c3_idx = 1, 2, 3

        c1_pos, c1_quat = CONN_POSITIONS[c1_idx], CONN_QUATS[c1_idx]
        c2_pos, c2_quat = CONN_POSITIONS[c2_idx], CONN_QUATS[c2_idx]
        c3_pos, c3_quat = CONN_POSITIONS[c3_idx], CONN_QUATS[c3_idx]

        c1_mesh = connector_assignments[node_id][1]
        c2_mesh = connector_assignments[node_id][2]
        c3_mesh = connector_assignments[node_id][3]

        m_quat = "1 0 0 0"
        joint_z = "0.0005" if m_type == "valley fold" else "-0.0055"
        joint_range = "0 90" if m_type == "valley fold" else "-90 0"

        if is_root:
            g_pos, g_R = global_pose.get(node_id, (np.zeros(3), np.eye(3)))
            module_pos = " ".join(f"{x:.9f}" for x in g_pos)
            module_macro_quat = " ".join(f"{x:.9f}" for x in matrix_to_quaternion(g_R))
            freejoint = f'    <freejoint name="free_module_{num_id}"/>'
        else:
            edge = incoming_edge[node_id]
            module_pos = " ".join(f"{x:.9f}" for x in edge["pos"])
            module_macro_quat = " ".join(f"{x:.9f}" for x in edge["quat"])
            freejoint = ""

        tpl = rigid_body_tpl if m_type == "non-foldable" else fold_body_tpl
        body_xml = tpl.format(
            module_id=num_id, module_pos=module_pos, module_macro_quat=module_macro_quat,
            module_quat=m_quat, freejoint=freejoint,
            connector1_pos=c1_pos, connector1_quat=c1_quat, connector1_mesh=c1_mesh,
            connector2_pos=c2_pos, connector2_quat=c2_quat, connector2_mesh=c2_mesh,
            connector3_pos=c3_pos, connector3_quat=c3_quat, connector3_mesh=c3_mesh,
            trans_val=TRANSPARENCY, joint_z=joint_z, joint_range=joint_range
        )
        element = ET.fromstring(body_xml)

        for child_info in tree_edges[node_id]:
            child_elem = build_module_element(child_info["child"], is_root=False)
            attach_point = get_attach_body(element, num_id, m_type, child_info["attach_slot"])
            if attach_point is None:
                raise RuntimeError(f"Could not find attach body for module_{num_id} slot {child_info['attach_slot']}")
            attach_point.append(child_elem)

        return element

    root = ET.fromstring(base_xml_skeleton)
    worldbody = root.find("worldbody")
    for r in roots:
        worldbody.append(build_module_element(r, is_root=True))

    # -------------------------------------------------------------
    # 8. Contact excludes + welds
    # -------------------------------------------------------------
    contact_elem = ET.SubElement(root, "contact")
    equality_elem = ET.SubElement(root, "equality")
    actuator_elem = ET.SubElement(root, "actuator")

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

    for u, v, d in G.edges(data=True):
        u_num, v_num = u.replace("module_", ""), v.replace("module_", "")
        c1, c2 = d["connector1"], d["connector2"]
        ET.SubElement(contact_elem, "exclude", body1=f"connector{c1}_{u_num}", body2=f"connector{c2}_{v_num}")

        if (u, v) not in tree_edge_set:
            body1, body2 = f"connector{c1}_{u_num}", f"connector{c2}_{v_num}"
            if G.nodes[u].get("module_type") in ["Mountain fold", "valley fold"] and c1 == 1:
                body1 = f"bodyBase_{u_num}"
            if G.nodes[v].get("module_type") in ["Mountain fold", "valley fold"] and c2 == 1:
                body2 = f"bodyBase_{v_num}"
            ET.SubElement(equality_elem, "weld", body1=body1, body2=body2,
                          solref="0.002 1", solimp="0.99 0.999 0.0001")

    module_ids = list(modules_info.keys())
    for i in range(len(module_ids)):
        for j in range(i + 1, len(module_ids)):
            ET.SubElement(contact_elem, "exclude", body1=module_ids[i], body2=module_ids[j])

    for idx, num_id in enumerate(fold_joints, start=1):
        if G.nodes[f"module_{num_id}"]["module_type"] == "valley fold":
            ET.SubElement(actuator_elem, "position", name=f"ctrl_joint{idx}", joint=f"joint_{num_id}",
                          kp="1", ctrlrange="0 1.5708", ctrllimited="true")
        else:
            ET.SubElement(actuator_elem, "position", name=f"ctrl_joint{idx}", joint=f"joint_{num_id}",
                          kp="1", ctrlrange="-1.5708 0", ctrllimited="true")

    xml_str = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="    ")
    clean_xml = "\n".join(line for line in pretty_xml.splitlines() if line.strip())

    with open(out_xml_path, "w") as f:
        f.write(clean_xml)

    print(f"[{graph_json_path}] Roots={roots} ({len(roots)} freejoint tree(s), "
          f"{len(components)} connected component(s)). "
          f"{len(fold_joints)} hinge joints. {len(extra_edges)} loop-closing weld(s).")

    return {
        "graph": G, "roots": roots, "global_pose": global_pose,
        "tree_edges": tree_edges, "components": components,
    }


if __name__ == "__main__":
    json_path = sys.argv[1] if len(sys.argv) > 1 else "../graphs/beetle.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "../models/assembly.xml"
    build_assembly(json_path, out_path)
